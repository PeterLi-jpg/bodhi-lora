"""LoRA SFT on (Med)Gemma3 via Google's tunix + qwix path (Stage 3).

This is the new trainer entrypoint that replaces the custom MaxText glue
in ``scripts/train_lora_maxtext.py``. It uses upstream tunix/qwix:

  * Gemma3 model loading via
    ``tunix.models.gemma3.params_safetensors.create_model_from_safe_tensors``
    (HF safetensors -> tunix nnx params; no orbax conversion step).
  * LoRA injection via ``qwix.apply_lora_to_model`` with a single
    ``LoraRule`` whose ``module_path`` regex targets the q_einsum / kv_einsum
    weights inside Gemma3's attention blocks.
  * SFT loop driven by ``tunix.sft.peft_trainer.PeftTrainer`` with the
    canonical ``gen_model_input_fn`` from ``tunix.cli.peft_main``.

Pipeline:
    1. Argparse + YAML load.
    2. Lazy-import jax / tunix / qwix / optax.
    3. Build a 1-D JAX mesh (axis 'fsdp', size = num_devices) so tunix
       can FSDP-shard params and data.
    4. Resolve the right ``ModelConfig`` for the requested model.
    5. ``create_model_from_safe_tensors`` -> base nnx model on the mesh.
    6. ``qwix.apply_lora_to_model`` -> LoRA-wrapped model.
    7. Build the dataset iterators via the existing
       ``scripts.maxtext_lora.dataset_loader.build_iterators``. Prefer
       ``output_format='tunix'`` if U4 has landed; otherwise wrap the
       legacy dict batches into ``TrainingInput`` inline.
    8. Build ``optax.adamw`` from the YAML's learning_rate.
    9. ``PeftTrainer.train(train_iter, eval_iter)`` inside the mesh.

CLI surface (mirrors what U6's launcher invokes per-VM):

    python scripts/train_lora_tunix.py \
      --config configs/lora_medgemma27b_tunix_smoke.yaml \
      --seed 42 \
      --output-dir checkpoints/seed_42

This file is additive — it does not modify the existing maxtext trainer.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple


# Map the friendly model name (HF id or its short alias) onto the matching
# ``ModelConfig`` factory. MedGemma-27B is Gemma3-27B with a SFT'd init, so
# its architecture config is identical to gemma3_27b_it. Keep this small —
# we only need the models we actually train.
_MODEL_CONFIG_FACTORIES: Dict[str, str] = {
    "google/medgemma-27b-text-it": "gemma3_27b_it",
    "google/gemma-3-27b-it": "gemma3_27b_it",
    "google/gemma-3-12b-it": "gemma3_12b_it",
    "google/gemma-3-4b-it": "gemma3_4b_it",
    "google/gemma-3-1b-it": "gemma3_1b_it",
    "google/gemma-3-270m-it": "gemma3_270m_it",
}


def _set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy so dataset shuffles + jax PRNG keys are
    reproducible across reruns. JAX PRNG keys are derived from this seed
    inside ``_train``."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _load_config(path: str) -> dict:
    """Read the YAML config and validate the keys this trainer expects.

    Schema (defined here, populated by U3's actual config file):

        seed: int
        model:
          name: str          # HF model id, e.g. "google/medgemma-27b-text-it"
          hf_path: str       # local safetensors directory
          dtype: str         # "bfloat16" or "float32"
        lora:
          r: int
          alpha: float
          dropout: float
          target_modules: [str, ...]   # used to build the LoRA module_path regex
        training:
          per_device_batch_size: int
          gradient_accumulation_steps: int
          learning_rate: float
          max_steps: int
          eval_interval: int
          max_seq_length: int
          bf16: bool
        data:
          dataset_dir: str
        paths:
          checkpoint_root_directory: str
          best_dir: str
    """
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    required_sections = ("model", "lora", "training", "data", "paths")
    for section in required_sections:
        if section not in cfg:
            raise ValueError(
                f"config {path!r}: missing required section {section!r}. "
                "See the U3 tunix config (configs/lora_medgemma27b_tunix*.yaml) "
                "for the expected shape."
            )
    if "name" not in cfg["model"]:
        raise ValueError(
            f"config {path!r}: model.name is required."
        )
    # hf_path is OPTIONAL; null/missing means resolve via snapshot_download at runtime.
    return cfg


def _expand(path: str) -> str:
    """``~/...`` and ``$VAR`` expansion so configs are portable between hosts."""
    return os.path.expanduser(os.path.expandvars(path))


def _resolve_model_config(model_name: str, dtype):
    """Map ``model_name`` -> a ``tunix.models.gemma3.model.ModelConfig``.

    Raises if the name isn't in the supported set rather than silently
    falling back, so a typo in YAML doesn't mis-shape the model.
    """
    from tunix.models.gemma3 import model as gemma3_model

    factory_name = _MODEL_CONFIG_FACTORIES.get(model_name)
    if factory_name is None:
        raise ValueError(
            f"Unsupported model.name {model_name!r}. Supported: "
            f"{sorted(_MODEL_CONFIG_FACTORIES)}. Add a new entry to "
            "_MODEL_CONFIG_FACTORIES if you need another size."
        )
    factory = getattr(gemma3_model.ModelConfig, factory_name)
    cfg = factory()
    # Honor the YAML-requested dtype rather than the ModelConfig default,
    # so configs that ask for float32 (debugging) actually get it.
    cfg.param_dtype = dtype
    return cfg


def _build_lora_module_path_regex(target_modules) -> str:
    """Build the qwix ``module_path`` regex from the YAML's
    ``lora.target_modules`` list.

    qwix's ``module_path`` uses '/' as the nesting separator (not '.'),
    matching ``flax_util.get_current_module_path()`` which joins NNX
    module names with '/'. Tunix's Gemma3 attention exposes the targets
    as ``layers/<i>/attn/q_einsum`` and ``layers/<i>/attn/kv_einsum``;
    the U7 CPU smoke confirmed this is the canonical path format. We
    anchor on the ``attn`` parent + the einsum module name to avoid
    accidentally matching unrelated submodules.
    """
    if not target_modules:
        raise ValueError(
            "lora.target_modules is empty; specify e.g. "
            "['q_einsum', 'kv_einsum'] for attention LoRA."
        )
    # re.escape each module name so a future name with special chars
    # (e.g. dots) doesn't blow up the regex.
    alts = "|".join(re.escape(m) for m in target_modules)
    return rf".*/attn/({alts})"


def _make_dummy_inputs(global_batch_size: int, max_seq_length: int):
    """Construct dummy inputs that match what ``model.__call__`` will see at
    runtime so qwix can trace the LoRA targets correctly.

    Shape MUST be the global batch (per-device * num-devices), not 1.
    Earlier iterations (v23-v27) hit subtle shape-mismatch bugs because
    LoRA was traced with B=1 then run with B=global.
    """
    import jax.numpy as jnp
    from tunix.sft import utils as sft_utils

    input_tokens = jnp.zeros((global_batch_size, max_seq_length), dtype=jnp.int32)
    # mirror the gen_model_input_fn used at training time so the traced
    # call shape is identical. positions / attention_mask are derived
    # from a non-pad mask of all True (worst-case dense attention).
    pad_mask = input_tokens != 0  # all False in this dummy, but the shape is what matters
    positions = sft_utils.build_positions_from_mask(pad_mask)
    attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
    return input_tokens, positions, attention_mask


def _wrap_legacy_batches(
    dict_iter: Iterable[Dict[str, Any]],
):
    """Forward-compat wrapper: convert the existing dict batches
    (``input_ids`` / ``labels`` / ``loss_mask``) to ``TrainingInput`` so
    this file works BEFORE U4 lands the ``output_format='tunix'`` kwarg.

    After U4 lands, ``build_iterators`` returns ``TrainingInput`` directly
    and this wrapper is unused. We keep it on as a fallback so a re-order
    of unit landings doesn't break ``--smoke``.
    """
    from tunix.sft import peft_trainer as _pt

    for batch in dict_iter:
        # labels == -100 marks "ignore in loss". The trainer's input_mask
        # is "True where loss should be computed", which is exactly the
        # negation of the ignore-label mask.
        yield _pt.TrainingInput(
            input_tokens=batch["input_ids"],
            input_mask=(batch["labels"] != -100),
        )


def _build_iterators(
    cfg: dict,
    seed: int,
) -> Tuple[Iterator[Any], Iterator[Any]]:
    """Build (train_iter, eval_iter) using the existing dataset_loader.

    Try ``output_format='tunix'`` first (the contract U4 will add). If
    the running version of dataset_loader doesn't accept that kwarg yet
    (TypeError), fall back to the legacy dict path and wrap inline.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.maxtext_lora import dataset_loader

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    common_kwargs = dict(
        dataset_dir=_expand(data_cfg["dataset_dir"]),
        train_file=data_cfg.get("train_file"),
        val_file=data_cfg.get("val_file"),
        per_device_batch_size=int(train_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_seq_length=int(train_cfg["max_seq_length"]),
        seed=seed,
    )

    try:
        train_iter, eval_iter, _steps = dataset_loader.build_iterators(
            output_format="tunix", **common_kwargs
        )
        return train_iter, eval_iter
    except TypeError:
        # U4 hasn't landed yet — fall through to the legacy dict path.
        pass

    train_iter, eval_iter, _steps = dataset_loader.build_iterators(**common_kwargs)
    return _wrap_legacy_batches(train_iter), _wrap_legacy_batches(eval_iter)


def _train(cfg: dict, seed: int, output_dir: str) -> None:
    """Run the LoRA SFT loop end-to-end via tunix's PeftTrainer."""
    # Lazy heavy imports — keeps `--help` light.
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import qwix
    from flax import nnx
    from tunix.models.gemma3 import params_safetensors as gemma3_params
    from tunix.sft import peft_trainer
    from tunix.sft import utils as sft_utils

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    # --- Mesh: 2-D (fsdp, tp) -------------------------------------------------
    # tunix's gemma3 model declares its param sharding as P('tp', 'fsdp'), so
    # the runtime mesh MUST expose both axes by name even when one of them is
    # size 1. v29 crashed at model load with:
    #   "Resource axis: tp of P('tp', 'fsdp') is not found in mesh: ('fsdp',)"
    # because we used a 1-D mesh ('fsdp',). For v6e-8 (8 chips) the simplest
    # split is fsdp=num_devices, tp=1: full FSDP across all chips, no tensor
    # parallelism. Configurable via training.tp_size in YAML for hybrid splits.
    devices = jax.devices()
    if not devices:
        raise RuntimeError(
            "jax.devices() returned an empty list. tunix needs at least one "
            "device (TPU, GPU, or CPU)."
        )
    n_devices = len(devices)
    tp_size = int(train_cfg.get("tp_size", 1))
    if n_devices % tp_size != 0:
        raise ValueError(
            f"training.tp_size={tp_size} does not divide num_devices={n_devices}; "
            "pick a tp_size that divides cleanly (e.g. 1, 2, 4, 8 on v6e-8)."
        )
    fsdp_size = n_devices // tp_size
    device_mesh = np.asarray(devices).reshape(fsdp_size, tp_size)
    mesh = jax.sharding.Mesh(device_mesh, ("fsdp", "tp"))
    print(f"[tunix] mesh: fsdp={fsdp_size} tp={tp_size} ({n_devices} devices)", flush=True)

    # --- dtype mapping --------------------------------------------------------
    dtype_str = str(model_cfg.get("dtype", "bfloat16")).lower()
    dtype_map = {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "fp32": jnp.float32}
    if dtype_str not in dtype_map:
        raise ValueError(
            f"model.dtype={dtype_str!r} is not supported; use 'bfloat16' or 'float32'."
        )
    dtype = dtype_map[dtype_str]

    # --- Base model from HF safetensors ---------------------------------------
    model_config = _resolve_model_config(model_cfg["name"], dtype)
    # hf_path: explicit local dir > $HF_HOME snapshot > snapshot_download (last
    # resort, downloads from HF). The smoke YAML sets hf_path: null deliberately
    # to mean "resolve at runtime via the HF cache or snapshot_download." v28
    # crashed here because the prior code unconditionally _expand()'d a None.
    raw_hf_path = model_cfg.get("hf_path")
    if raw_hf_path is None or raw_hf_path == "":
        from huggingface_hub import snapshot_download
        print(
            f"[tunix] hf_path is null; resolving {model_cfg['name']!r} via "
            f"huggingface_hub.snapshot_download (HF cache at {os.environ.get('HF_HOME', '~/.cache/huggingface')!r})",
            flush=True,
        )
        hf_path = snapshot_download(
            repo_id=model_cfg["name"],
            allow_patterns=["*.safetensors", "*.json", "*.model"],
        )
    else:
        hf_path = _expand(raw_hf_path)
        if not Path(hf_path).is_dir():
            raise FileNotFoundError(
                f"model.hf_path {hf_path!r} does not exist. Set hf_path: null "
                "in the YAML to auto-resolve via snapshot_download, or download "
                "with `huggingface-cli download` first."
            )
    print(f"[tunix] loading {model_cfg['name']} from {hf_path}", flush=True)
    model = gemma3_params.create_model_from_safe_tensors(
        file_dir=hf_path,
        config=model_config,
        mesh=mesh,
        dtype=dtype,
    )

    # --- LoRA injection via qwix ----------------------------------------------
    rank = int(lora_cfg["r"])
    alpha = float(lora_cfg["alpha"])
    dropout = float(lora_cfg.get("dropout", 0.0))
    module_path = _build_lora_module_path_regex(lora_cfg["target_modules"])
    print(
        f"[tunix] applying LoRA: rank={rank} alpha={alpha} dropout={dropout} "
        f"module_path={module_path!r}",
        flush=True,
    )
    lora_provider = qwix.LoraProvider(
        rules=[
            qwix.LoraRule(
                module_path=module_path,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        ]
    )
    # Dummy inputs MUST match the global batch shape used during training,
    # otherwise qwix's traced LoRA shapes diverge from runtime — see v23-v27
    # bugs called out in the unit brief.
    per_device = int(train_cfg["per_device_batch_size"])
    global_batch_size = per_device * len(devices)
    max_seq_length = int(train_cfg["max_seq_length"])
    dummy_tokens, dummy_positions, dummy_attn = _make_dummy_inputs(
        global_batch_size, max_seq_length
    )
    model = qwix.apply_lora_to_model(
        model,
        lora_provider,
        dummy_tokens,
        dummy_positions,
        None,  # cache (no kv cache during training)
        dummy_attn,
        rngs=nnx.Rngs(seed),
    )

    # --- Optimizer ------------------------------------------------------------
    learning_rate = float(train_cfg["learning_rate"])
    optimizer = optax.adamw(learning_rate=learning_rate)

    # --- Datasets -------------------------------------------------------------
    train_iter, eval_iter = _build_iterators(cfg, seed=seed)

    # --- TrainingConfig -------------------------------------------------------
    out_dir = Path(_expand(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    # paths.checkpoint_root_directory may already include the seed; if it
    # contains "<N>" treat that as a placeholder. Otherwise nest under
    # output_dir for the launcher's per-seed layout.
    ckpt_root_raw = paths_cfg.get("checkpoint_root_directory") or str(out_dir / "orbax")
    ckpt_root_substituted = _expand(ckpt_root_raw).replace("<N>", str(seed))
    # Orbax's tensorstore kvstore rejects relative paths with
    #   ValueError: Checkpoint path should be absolute. Got <relative>
    # at first save (we hit this on v34 — training succeeded, ckpt save
    # crashed). Resolve to an absolute path here regardless of how the
    # YAML / launcher passed it. We use `Path.resolve(strict=False)`
    # because the directory does not yet exist on first save.
    ckpt_root = str(Path(ckpt_root_substituted).resolve())
    print(f"[tunix] checkpoint root: {ckpt_root}", flush=True)

    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=int(train_cfg.get("eval_interval", 1)),
        max_steps=int(train_cfg["max_steps"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        checkpoint_root_directory=ckpt_root,
    )

    trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)

    # gen_model_input_fn: convert TrainingInput -> kwargs that the loss_fn
    # / model expect. Mirrors tunix.cli.peft_main.PeftPipeline.run_peft_trainer.
    def gen_model_input_fn(x: peft_trainer.TrainingInput):
        pad_mask = x.input_tokens != 0
        positions = sft_utils.build_positions_from_mask(pad_mask)
        attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
        return {
            "input_tokens": x.input_tokens,
            "input_mask": x.input_mask,
            "positions": positions,
            "attention_mask": attention_mask,
        }

    trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

    # --- Train ----------------------------------------------------------------
    # tunix's PeftTrainer runs an INITIAL eval at step 0 whenever
    # eval_ds is not None (peft_trainer.py:619-620), independent of
    # eval_every_n_steps. v32 OOMed on the eval JIT compile with 27B
    # params + qwix LoRA on v6e-8. When the YAML's eval_interval is
    # greater than max_steps, the user has effectively asked for "no
    # eval"; respect that by passing eval_ds=None so the initial eval
    # doesn't fire either.
    eval_interval = int(train_cfg.get("eval_interval", 1))
    eval_arg = eval_iter if eval_interval <= int(train_cfg["max_steps"]) else None
    if eval_arg is None:
        print(
            f"[tunix] eval_interval={eval_interval} > max_steps="
            f"{int(train_cfg['max_steps'])}: skipping eval entirely "
            "(passing eval_ds=None to trainer.train).",
            flush=True,
        )

    print(
        f"[tunix] starting training: max_steps={training_config.max_steps} "
        f"global_batch_size={global_batch_size} grad_accum="
        f"{training_config.gradient_accumulation_steps} lr={learning_rate}",
        flush=True,
    )
    with mesh:
        trainer.train(train_iter, eval_arg)
    print("[tunix] training finished.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA SFT entrypoint via tunix + qwix (Stage 3, new path)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to a tunix LoRA YAML config "
             "(see configs/lora_medgemma27b_tunix*.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the seed in the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="directory used as the default checkpoint root if "
             "paths.checkpoint_root_directory is unset.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(
        cfg.get("seed", cfg.get("training", {}).get("seed", 42))
    )
    _set_global_seeds(seed)

    _train(cfg, seed=seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
