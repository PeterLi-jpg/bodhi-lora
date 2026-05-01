"""LoRA SFT on MedGemma-27B using a forked MaxText (Stage 3, Phase 1).

This is the MaxText counterpart of scripts/train_lora.py — same training
semantics (5 epochs, eval at epoch boundaries, cosine LR with warmup,
load_best_model_at_end), but the model lives in Flax and the training
loop runs through MaxText's SFT pipeline instead of HF Trainer + torch_xla.

Pipeline overview:
    1. Argparse + YAML config load (no JAX/MaxText needed — keeps --help
       importable on dev boxes that don't have the full TPU stack).
    2. Lazy-import orbax + MaxText, then load the base-model checkpoint
       produced by Unit 4 from `paths.maxtext_orbax_checkpoint`.
    3. Build the Gemma-3-27B Flax model from `third_party/maxtext`.
    4. Apply Unit 3's LoRA injector with `lora.target_modules` from config.
    5. Build the train/val iterator over the MaxText-formatted shards
       produced by Unit 5 at `paths.maxtext_dataset_dir`.
    6. Run MaxText's SFT loop with only the LoRA params marked trainable
       (everything else gets `optax.set_to_zero()` so the optimizer state
       stays small — the analogue of the requires_grad-only optimizer
       trick we used on torch_xla, see #39795).
    7. Periodic orbax checkpoint of the LoRA params under
       `<output-dir>/orbax/`.
    8. At end of training, hand the latest orbax checkpoint to Unit 6's
       PEFT exporter to produce `<output-dir>/best/` — a directory the
       Stage 4 vLLM eval can load via the existing PEFT adapter path.

`--help` is argparse-only and does NOT require JAX, MaxText, orbax, or
the sibling `scripts/maxtext_lora/` package to be importable.  All heavy
imports happen inside helper functions so unit-7 can ship + acceptance-
check (--help, py_compile, pytest) before sibling units land.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _set_global_seeds(seed: int) -> None:
    """Seed Python + NumPy.  JAX PRNG keys are derived inside _train()."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        # NumPy is a hard dep of MaxText, but allow --help to work
        # on boxes that don't have it installed.
        pass


def _load_config(path: str) -> dict:
    """Read the YAML config and validate the keys train_lora_maxtext expects.

    Validation here is intentionally light — we just confirm the top-level
    sections exist so a typo doesn't surface as a confusing KeyError 2 hours
    into a TPU run.
    """
    import yaml  # local import: PyYAML is everywhere, but keep main import-light
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "lora", "training", "data", "paths"):
        if section not in cfg:
            raise ValueError(
                f"config {path!r}: missing required section {section!r}. "
                f"See configs/lora_medgemma27b_maxtext.yaml for the expected shape."
            )
    for key in ("maxtext_orbax_checkpoint", "maxtext_dataset_dir"):
        if key not in cfg["paths"]:
            raise ValueError(
                f"config {path!r}: paths.{key} is required.  "
                "It points to the output of Unit 4 (orbax base-model checkpoint) "
                "or Unit 5 (MaxText-formatted dataset dir)."
            )
    return cfg


def _expand(path: str) -> str:
    """`~/...` and `$VAR` expansion so configs are portable between VMs."""
    return os.path.expanduser(os.path.expandvars(path))


def _train(cfg: dict, seed: int, output_dir: str) -> None:
    """Run the MaxText SFT training loop.

    All heavy imports happen here (not at module top level) so `--help` and
    py_compile work on a dev box that doesn't have JAX/MaxText/orbax.
    """
    # --- Lazy heavy imports ----------------------------------------------------
    # JAX + MaxText come from the third_party/maxtext fork (Unit 1).
    # scripts.maxtext_lora.* are sibling units (3 = LoRA injector,
    # 5 = dataset converter, 6 = PEFT exporter).  None of these are needed
    # for argparse --help, so they import here rather than at module top.
    import jax

    # third_party/maxtext is vendored by Unit 1.  Add to sys.path before
    # importing — MaxText uses a `MaxText/` package directory layout rather
    # than a pip-installable package.
    repo_root = Path(__file__).resolve().parent.parent
    maxtext_root = repo_root / "third_party" / "maxtext"
    if str(maxtext_root) not in sys.path:
        sys.path.insert(0, str(maxtext_root))

    from MaxText import pyconfig
    from MaxText.experimental.sft import sft_trainer

    from scripts.maxtext_lora import lora_inject
    from scripts.maxtext_lora import dataset_loader
    from scripts.maxtext_lora import export_peft

    # --- Resolve config sections + paths --------------------------------------
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    orbax_ckpt = _expand(paths_cfg["maxtext_orbax_checkpoint"])
    dataset_dir = _expand(paths_cfg["maxtext_dataset_dir"])
    train_file = _expand(data_cfg["train_file"])
    val_file = _expand(data_cfg["val_file"])

    if not Path(orbax_ckpt).exists():
        raise FileNotFoundError(
            f"orbax base-model checkpoint not found at {orbax_ckpt}.  "
            "Run Unit 4's HF -> orbax converter first "
            "(scripts/maxtext_lora/convert_hf_to_orbax.py)."
        )
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"MaxText dataset dir not found at {dataset_dir}.  "
            "Run Unit 5's dataset converter first "
            "(scripts/maxtext_lora/convert_dataset.py)."
        )

    output_dir = _expand(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    orbax_out = str(Path(output_dir) / "orbax")
    best_out = str(Path(output_dir) / "best")

    # --- Build MaxText config + base model ------------------------------------
    # MaxText's pyconfig translates a list of CLI-style overrides into a
    # frozen training config.  We feed in the keys the SFT trainer expects;
    # anything not explicitly set falls back to MaxText's defaults for
    # gemma3-27b.  Mirrors what the launcher (Unit 8) would otherwise pass
    # on the command line — keeping it in one place here means the YAML is
    # the single source of truth for hyperparameters.
    mt_argv = [
        "model_name=gemma3-27b",
        f"load_parameters_path={orbax_ckpt}",
        f"dataset_path={dataset_dir}",
        f"per_device_batch_size={train_cfg['per_device_train_batch_size']}",
        # MaxText counts steps directly, not epochs, so derive from epochs *
        # steps_per_epoch inside the trainer call below.  We forward the
        # raw fields so SFT trainer can size its scheduler correctly.
        f"num_epochs={train_cfg['num_epochs']}",
        f"gradient_accumulation_steps={train_cfg['gradient_accumulation_steps']}",
        f"learning_rate={train_cfg['learning_rate']}",
        f"warmup_steps_fraction={train_cfg['warmup_ratio']}",
        f"learning_rate_schedule={train_cfg['lr_scheduler_type']}",
        f"max_target_length={train_cfg['max_seq_length']}",
        f"weight_dtype={'bfloat16' if train_cfg.get('bf16', True) else 'float32'}",
        f"dtype={'bfloat16' if train_cfg.get('bf16', True) else 'float32'}",
        f"base_output_directory={orbax_out}",
        f"run_name=lora_seed_{seed}",
        f"data_seed={seed}",
        f"init_weights_seed={seed}",
        # Logging cadence mirrors HF Trainer's logging_steps.
        f"log_period={train_cfg.get('logging_steps', 5)}",
        # save / eval at epoch boundary — see translation in sft_trainer.
        f"checkpoint_period_strategy={train_cfg.get('save_strategy', 'epoch')}",
        f"eval_period_strategy={train_cfg.get('eval_strategy', 'epoch')}",
    ]
    mt_cfg = pyconfig.initialize(mt_argv)

    print(f"MaxText config initialized: model={model_cfg['name']!r} "
          f"seed={seed} output_dir={output_dir!r}")

    # --- LoRA injection (Unit 3) ----------------------------------------------
    # The injector returns a (model, params, lora_param_filter) triple where
    # lora_param_filter is a pytree mask flagging only the LoRA delta params.
    # MaxText's optimizer takes this mask via optax.masked() so base weights
    # stay frozen and the optimizer state size is bounded by the LoRA rank
    # (~12M params for r=8 + 2 targets — about 50 MB of fp32 m+v state).
    # This is the JAX equivalent of the requires_grad-only optimizer
    # workaround we use on torch_xla in train_lora.py.
    model, params, lora_param_filter = lora_inject.apply_lora(
        mt_cfg,
        target_modules=lora_cfg["target_modules"],
        rank=lora_cfg["r"],
        alpha=lora_cfg["lora_alpha"],
        dropout=lora_cfg.get("lora_dropout", 0.05),
        variant=lora_cfg.get("variant", "standard"),
        seed=seed,
    )

    n_lora = sum(int(x) for x in jax.tree_util.tree_leaves(lora_param_filter))
    print(f"LoRA injection done: {n_lora:,} trainable params "
          f"(target_modules={lora_cfg['target_modules']}, r={lora_cfg['r']})")

    # --- Dataset iterator (Unit 5) --------------------------------------------
    train_iter, eval_iter, steps_per_epoch = dataset_loader.build_iterators(
        dataset_dir=dataset_dir,
        train_file=train_file,
        val_file=val_file,
        per_device_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_seq_length=train_cfg["max_seq_length"],
        seed=seed,
    )
    total_steps = int(train_cfg["num_epochs"]) * int(steps_per_epoch)
    print(f"Dataset built: {steps_per_epoch} steps/epoch x "
          f"{train_cfg['num_epochs']} epochs = {total_steps} total steps")

    # --- Training loop (MaxText SFT) ------------------------------------------
    # sft_trainer.train() runs the SFT loop, applies grad updates only to
    # params matching lora_param_filter, evaluates on eval_iter at the
    # cadence in eval_period_strategy, and writes orbax checkpoints to
    # base_output_directory/orbax at the cadence in checkpoint_period_strategy.
    # Returns the final orbax checkpoint directory (e.g.
    # <output-dir>/orbax/checkpoints/<step>) which feeds the PEFT exporter.
    #
    # The "first step landed in N min" line is parsed by the launcher
    # (Unit 8) to detect compile-done — once the first step has flushed,
    # XLA compile finished and subsequent steps will be fast.
    t_start = time.monotonic()
    first_step_printed = False

    def _on_first_step(_state):
        """Fired by sft_trainer once the first training step has flushed."""
        nonlocal first_step_printed
        if first_step_printed:
            return
        elapsed_min = (time.monotonic() - t_start) / 60.0
        print(f"first step landed in {elapsed_min:.1f} min", flush=True)
        first_step_printed = True

    final_ckpt = sft_trainer.train(
        config=mt_cfg,
        model=model,
        params=params,
        train_iter=train_iter,
        eval_iter=eval_iter,
        trainable_param_filter=lora_param_filter,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        on_first_step=_on_first_step,
    )

    # Fallback for older MaxText forks whose sft_trainer.train() ignores the
    # on_first_step kwarg — emit one line so the launcher's compile-done
    # detector still has something to grep.
    if not first_step_printed:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        print(f"first step landed in {elapsed_min:.1f} min "
              "(post-train fallback)", flush=True)

    # --- PEFT export (Unit 6) -------------------------------------------------
    # export_peft.write_adapter() reads the final orbax checkpoint, pulls out
    # only the LoRA delta params, and writes them in the HuggingFace PEFT
    # adapter layout (adapter_model.safetensors + adapter_config.json) under
    # <output-dir>/best/ so Stage 4 eval can load it via the existing
    # PEFT path with no MaxText dependency.
    Path(best_out).mkdir(parents=True, exist_ok=True)
    export_peft.write_adapter(
        orbax_checkpoint=final_ckpt,
        output_dir=best_out,
        base_model_name=model_cfg["name"],
        target_modules=lora_cfg["target_modules"],
        rank=lora_cfg["r"],
        alpha=lora_cfg["lora_alpha"],
        dropout=lora_cfg.get("lora_dropout", 0.05),
        variant=lora_cfg.get("variant", "standard"),
    )

    # Drop a small JSON next to best/ so the launcher (and Stage 4) can
    # confirm the run finished without parsing trainer state.  Mirrors the
    # role of trainer_state.json on the torch_xla path.
    summary = {
        "seed": seed,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "final_orbax_checkpoint": str(final_ckpt),
        "model_name": model_cfg["name"],
        "lora": {
            "r": lora_cfg["r"],
            "alpha": lora_cfg["lora_alpha"],
            "dropout": lora_cfg.get("lora_dropout", 0.05),
            "target_modules": list(lora_cfg["target_modules"]),
            "variant": lora_cfg.get("variant", "standard"),
        },
    }
    with open(Path(best_out) / "trainer_state.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved adapter to {best_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA SFT of MedGemma-27B via MaxText (Stage 3, Phase 1)."
    )
    parser.add_argument("--config", required=True,
                        help="path to a MaxText LoRA YAML config "
                             "(see configs/lora_medgemma27b_maxtext.yaml).")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed in the YAML config (useful for "
                             "multi-seed runs where one YAML is reused with "
                             "different seeds per invocation).")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="directory to save orbax checkpoints + best adapter. "
                             "<output-dir>/orbax/  -> periodic MaxText checkpoints; "
                             "<output-dir>/best/   -> final HF/PEFT adapter for "
                             "Stage 4 eval.  Override per seed in multi-seed runs, "
                             "e.g. checkpoints/seed_42 .")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(
        cfg.get("seed", cfg.get("training", {}).get("seed", 42))
    )
    _set_global_seeds(seed)

    _train(cfg, seed=seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
