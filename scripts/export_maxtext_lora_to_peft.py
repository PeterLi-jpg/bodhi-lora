"""Convert a MaxText orbax LoRA checkpoint to a HuggingFace PEFT adapter directory.

Stage 3 of the BOHDI pipeline used to fine-tune via PyTorch + torch_xla and
write a PEFT adapter directly through ``PeftModel.save_pretrained``.  Migrating
to MaxText keeps Stage 4 (vLLM eval, ``XLALoRAEngine.merge_and_unload``)
unchanged only if the trained adapter still ends up on disk in the exact
PEFT layout.  This script is the bridge: it reads an orbax checkpoint
containing trained LoRA A/B matrices and writes a directory with
``adapter_config.json`` and ``adapter_model.safetensors`` that
``peft.PeftModel.from_pretrained`` can load against the original base model.

Conventions assumed of the MaxText fork's checkpoint
----------------------------------------------------
The orbax pytree is restored as a nested dict.  After flattening with
``"/"`` joiners, every LoRA leaf path must contain:

* a layer-index segment matching ``layers_<i>`` or ``layer_<i>`` (any case).
* a projection-name segment that maps to one of the HF target modules
  (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``,
  ``up_proj``, ``down_proj``).  MaxText's own names (``query``, ``key``,
  ``value``, ``out`` and ``ffw_gating``, ``ffw_up``, ``ffw_down``) are
  aliased automatically.
* a final segment matching ``lora_a`` or ``lora_b`` (case-insensitive,
  ``A``/``B`` and ``lora_A.kernel`` / ``lora_b.weight`` variants are also
  accepted).

Anything else in the pytree (optimizer state, base weights, step counters)
is silently ignored — we only export the trained adapter.

Shape handling
--------------
PEFT stores ``lora_A`` with shape ``(r, in_features)`` and ``lora_B`` with
shape ``(out_features, r)``.  Flax/JAX convention typically stores
``lora_a`` as ``(in_features, r)`` and ``lora_b`` as ``(r, out_features)``.
This script auto-detects which axis is the rank by comparing matrix
dimensions to the configured ``r`` and transposes if needed, so checkpoints
written either way round trip cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from peft import LoraConfig
from safetensors.torch import save_file


# Map MaxText-style projection names -> HF Gemma-3 target_modules names.
# Both keys (MaxText alias) and values (HF canonical) are treated as valid
# inputs; we normalise to the HF name in the output adapter file.
_PROJECTION_ALIASES: Dict[str, str] = {
    # attention
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "out": "o_proj",
    "out_proj": "o_proj",
    # MLP
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "ffw_gating": "gate_proj",
    "ffw_up": "up_proj",
    "ffw_down": "down_proj",
    "wi_0": "gate_proj",
    "wi_1": "up_proj",
    "wo": "down_proj",
}

# All HF projection names that live under ``self_attn`` vs. ``mlp`` in
# Gemma-3 / MedGemma.  Used to build the right PEFT key prefix.
_ATTN_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj"}
_MLP_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}

# Path-segment patterns we recognise.
_LAYER_RE = re.compile(r"^layers?_(\d+)$", re.IGNORECASE)
_LORA_A_RE = re.compile(r"^lora[_-]?a$", re.IGNORECASE)
_LORA_B_RE = re.compile(r"^lora[_-]?b$", re.IGNORECASE)
# Some MaxText forks nest the actual array under a leaf dict like
# ``{"kernel": array}`` (Flax linen) or ``{"weight": array}`` (NNX).  We
# strip these terminal wrappers when walking the tree.  "value" is
# deliberately NOT here even though Flax NNX wraps params in ``{"value":
# array}`` — it collides with the v_proj alias and would silently
# drop attention paths.  NNX checkpoints need a separate handler if
# they ever land here.
_LEAF_KEYS = {"kernel", "weight", "params"}


def _flatten_pytree(
    tree: Any,
    prefix: Tuple[str, ...] = (),
) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    """Yield (path, leaf) for every leaf in a nested dict / list / tuple.

    Treats numpy arrays, torch tensors, and anything with a ``shape``
    attribute as terminal leaves.  Strings inside the tree are left alone
    (they show up as leaves but later filtering drops them).
    """
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten_pytree(v, prefix + (str(k),))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _flatten_pytree(v, prefix + (str(i),))
    else:
        yield prefix, tree


def _looks_like_array(leaf: Any) -> bool:
    """Cheap duck-type check — orbax may return numpy / jax / torch arrays."""
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _to_numpy(leaf: Any) -> np.ndarray:
    """Best-effort conversion of an orbax leaf to a numpy array.

    Avoids importing jax at module import time — only resolved when we
    actually have to convert a jax.Array.  numpy arrays and torch tensors
    pass through directly.
    """
    if isinstance(leaf, np.ndarray):
        return leaf
    if isinstance(leaf, torch.Tensor):
        return leaf.detach().cpu().numpy()
    # jax.Array, jnp.ndarray, ml_dtypes scalar, etc. — np.asarray handles
    # all of them via __array__.
    return np.asarray(leaf)


def _peft_key(layer_idx: int, hf_proj: str, ab: str) -> str:
    """Return the safetensors key PEFT would have written for this slot.

    PEFT prefixes adapters with ``base_model.model.`` and then mirrors the
    base model's parameter path.  For Gemma-3 / MedGemma, attention
    projections live at
    ``model.layers.<i>.self_attn.<proj>`` and MLP projections at
    ``model.layers.<i>.mlp.<proj>``.
    """
    if hf_proj in _ATTN_PROJECTIONS:
        bucket = "self_attn"
    elif hf_proj in _MLP_PROJECTIONS:
        bucket = "mlp"
    else:  # pragma: no cover — guarded earlier in the pipeline
        raise ValueError(f"unrecognised projection name {hf_proj!r}")
    return (
        f"base_model.model.model.layers.{layer_idx}."
        f"{bucket}.{hf_proj}.lora_{ab}.weight"
    )


def _identify_lora_leaf(
    path: Tuple[str, ...],
) -> Optional[Tuple[int, str, str]]:
    """Decide whether ``path`` names a LoRA A or B matrix.

    Returns ``(layer_idx, hf_projection_name, "A" | "B")`` if so,
    or ``None`` if the path doesn't look like a LoRA leaf we should export.

    The search is permissive about ordering — the path can be
    ``("params", "decoder", "layers_5", "self_attention", "q_proj", "lora_a", "kernel")``
    or ``("model", "layer_5", "attention", "query_lora_A", "weight")`` —
    we match the first ``layers_<i>``-shaped segment, the first known
    projection alias, and the last ``lora_a``/``lora_b`` segment we see.
    """
    layer_idx: Optional[int] = None
    hf_proj: Optional[str] = None
    ab: Optional[str] = None

    for segment in path:
        # Strip trailing wrapper keys ("kernel", "weight", "params") to
        # expose the lora_a / lora_b token they wrap, e.g. the path
        # (..., "lora_a", "kernel") is really a lora_a leaf.
        if segment.lower() in _LEAF_KEYS:
            continue

        # Some checkpoints concatenate proj+lora into one segment, e.g.
        # "q_proj_lora_A" or "query_lora_a".  Split on the first lora_
        # token so both halves get matched below.
        sub_segments: List[str] = [segment]
        m = re.match(
            r"^(?P<head>.+?)[._-]?(?P<tail>lora[_-]?[ab])$",
            segment,
            re.IGNORECASE,
        )
        if m:
            sub_segments = [m.group("head"), m.group("tail")]

        for sub in sub_segments:
            if not sub:
                continue
            # Layer index?
            mm = _LAYER_RE.match(sub)
            if mm and layer_idx is None:
                layer_idx = int(mm.group(1))
                continue
            # Projection alias?
            normalised = sub.lower()
            if normalised in _PROJECTION_ALIASES and hf_proj is None:
                hf_proj = _PROJECTION_ALIASES[normalised]
                continue
            # LoRA A / B?
            if _LORA_A_RE.match(sub):
                ab = "A"
                continue
            if _LORA_B_RE.match(sub):
                ab = "B"
                continue

    if layer_idx is None or hf_proj is None or ab is None:
        return None
    return layer_idx, hf_proj, ab


def _orient_lora_matrix(
    arr: np.ndarray,
    *,
    ab: str,
    lora_r: int,
) -> np.ndarray:
    """Return ``arr`` with axes in PEFT's convention.

    PEFT layout (matches PyTorch ``nn.Linear.weight``):
      - lora_A.weight: ``(r, in_features)``
      - lora_B.weight: ``(out_features, r)``

    Flax/MaxText typical layout:
      - lora_a kernel: ``(in_features, r)``
      - lora_b kernel: ``(r, out_features)``

    We detect the rank axis by matching against ``lora_r`` and transpose
    if the matrix is in the Flax orientation.  Non-2D arrays are returned
    unchanged (the test below catches the obvious failure case).
    """
    if arr.ndim != 2:
        return arr
    rows, cols = arr.shape
    if ab == "A":
        # PEFT wants r as axis-0.  If rows == r we're already correct;
        # if cols == r we need to transpose.  If neither matches we fall
        # through and let the loader complain — better than silently
        # mis-shaping the weight.
        if rows == lora_r:
            return arr
        if cols == lora_r:
            return arr.T
    else:  # ab == "B"
        # PEFT wants r as axis-1.
        if cols == lora_r:
            return arr
        if rows == lora_r:
            return arr.T
    return arr


def _torch_dtype_from_numpy(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor preserving bf16 when present.

    bf16 isn't a native numpy dtype; orbax/jax usually exposes it via
    ``ml_dtypes.bfloat16``.  ``np.asarray`` preserves that, but
    ``torch.from_numpy`` doesn't know about it — go through a uint16
    bit-pattern view in that case.  Plain float32 / float16 / int dtypes
    pass through ``torch.from_numpy`` directly.
    """
    dtype_str = str(arr.dtype)
    if dtype_str == "bfloat16":
        # Reinterpret as uint16, copy into a torch bf16 tensor.
        u16 = arr.view(np.uint16)
        return torch.from_numpy(np.ascontiguousarray(u16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def collect_lora_weights(
    pytree: Any,
    *,
    lora_r: int,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Walk ``pytree`` and return (PEFT-keyed weights, sorted target_modules).

    Returns a dict of PEFT-formatted safetensors keys -> torch tensors,
    and the list of HF projection names actually present (so we can stash
    that in the adapter_config target_modules field).
    """
    weights: Dict[str, torch.Tensor] = {}
    seen_projections: set[str] = set()

    for path, leaf in _flatten_pytree(pytree):
        if not _looks_like_array(leaf):
            continue
        ident = _identify_lora_leaf(path)
        if ident is None:
            continue
        layer_idx, hf_proj, ab = ident
        arr = _to_numpy(leaf)
        arr = _orient_lora_matrix(arr, ab=ab, lora_r=lora_r)
        key = _peft_key(layer_idx, hf_proj, ab)
        if key in weights:
            raise ValueError(
                f"duplicate LoRA leaf for {key!r} in checkpoint — "
                f"clashing source path: {'/'.join(path)}"
            )
        weights[key] = _torch_dtype_from_numpy(arr).contiguous()
        seen_projections.add(hf_proj)

    if not weights:
        raise ValueError(
            "no LoRA leaves found in the orbax pytree. Either the path is "
            "empty, or the MaxText fork uses naming this script doesn't "
            "recognise. Check that param paths contain layer indices, "
            "projection names (q_proj/k_proj/...), and lora_a/lora_b "
            "segments. Add aliases to _PROJECTION_ALIASES if needed."
        )

    return weights, sorted(seen_projections)


def load_orbax_checkpoint(path: Path) -> Any:
    """Restore an orbax checkpoint into a nested dict.

    We use ``PyTreeCheckpointer`` because it imposes the fewest
    assumptions on the saved structure — any pytree can come back, and
    we walk it generically.  Modern MaxText writes via the
    ``CheckpointManager`` API but the on-disk format remains compatible
    with this restore path for plain pytrees (no Composite handlers).
    """
    import orbax.checkpoint as ocp

    abspath = str(path.resolve())
    restorer = ocp.PyTreeCheckpointer()
    return restorer.restore(abspath)


def write_peft_adapter(
    *,
    output_dir: Path,
    weights: Dict[str, torch.Tensor],
    target_modules: List[str],
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    task_type: str,
    use_dora: bool,
    use_rslora: bool,
) -> None:
    """Materialise ``adapter_config.json`` + ``adapter_model.safetensors``.

    The config is built via PEFT's ``LoraConfig`` so the JSON schema (key
    set, default fields the running PEFT version cares about) is exactly
    what ``PeftModel.from_pretrained`` validates against.  Hard-coding a
    JSON dict here would silently bit-rot when peft adds / renames
    fields — rebuilding through the dataclass keeps the schema honest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        use_dora=use_dora,
        use_rslora=use_rslora,
        base_model_name_or_path=base_model,
    )
    cfg.save_pretrained(str(output_dir))

    # save_file accepts a flat str -> Tensor dict.  Tensors must be
    # contiguous + on CPU; collect_lora_weights guarantees both.
    save_file(weights, str(output_dir / "adapter_model.safetensors"))


def _resolve_lora_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Pull r / alpha / dropout / target_modules / variant / task_type.

    Precedence: explicit CLI flag > YAML config > sensible defaults.
    target_modules from CLI / YAML is *advisory*; the actual list written
    to adapter_config.json is the intersection of "what the YAML asked
    for" and "what we found in the orbax tree" — so a YAML that requested
    7 targets but the run only trained 2 (the TPU diag-run case in
    configs/lora_medgemma27b_tpu.yaml) ends up with the 2 actually-trained
    modules in the output.
    """
    settings: Dict[str, Any] = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": None,
        "task_type": "CAUSAL_LM",
        "variant": "standard",
    }

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        lora_cfg = cfg.get("lora", {})
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules",
                    "task_type", "variant"):
            if key in lora_cfg:
                settings[key] = lora_cfg[key]

    if args.lora_r is not None:
        settings["r"] = args.lora_r
    if args.lora_alpha is not None:
        settings["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        settings["lora_dropout"] = args.lora_dropout
    if args.target_modules:
        settings["target_modules"] = list(args.target_modules)
    if args.task_type:
        settings["task_type"] = args.task_type
    if args.lora_variant:
        settings["variant"] = args.lora_variant

    variant = str(settings["variant"]).lower()
    if variant not in ("standard", "dora", "rslora"):
        raise ValueError(
            f"lora.variant={variant!r} not recognised. "
            "Use one of: standard, dora, rslora."
        )
    settings["use_dora"] = variant == "dora"
    settings["use_rslora"] = variant == "rslora"
    return settings


def export(
    *,
    orbax_path: Path,
    output_dir: Path,
    base_model: str,
    settings: Dict[str, Any],
) -> None:
    """End-to-end: load orbax, collect LoRA weights, write the PEFT dir."""
    pytree = load_orbax_checkpoint(orbax_path)
    weights, found_targets = collect_lora_weights(
        pytree, lora_r=int(settings["r"])
    )

    requested = settings.get("target_modules")
    if requested:
        # Honour the YAML's intent — drop anything we didn't ask for and
        # warn loudly about modules that were configured but not found
        # (almost always a bug in the training run).
        requested_set = set(requested)
        target_modules = [m for m in found_targets if m in requested_set]
        missing = sorted(requested_set - set(found_targets))
        extra = sorted(set(found_targets) - requested_set)
        if missing:
            print(
                f"WARNING: target_modules in config but not in checkpoint: "
                f"{missing}. The trained run skipped these projections; "
                "downstream merge will not modify those weights.",
                file=sys.stderr,
            )
        if extra:
            print(
                f"WARNING: target_modules in checkpoint but not in config: "
                f"{extra}. Dropping these from the exported adapter.",
                file=sys.stderr,
            )
            # Filter the weights dict to match target_modules so we don't
            # silently ship adapters the YAML never asked for.
            weights = {
                k: v for k, v in weights.items()
                if any(f".{m}.lora_" in k for m in target_modules)
            }
    else:
        target_modules = found_targets

    if not target_modules:
        raise ValueError(
            "No LoRA target modules left after intersecting checkpoint "
            "contents with the configured target_modules. The run likely "
            "trained different modules than the config requested."
        )

    write_peft_adapter(
        output_dir=output_dir,
        weights=weights,
        target_modules=target_modules,
        base_model=base_model,
        lora_r=int(settings["r"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        task_type=str(settings["task_type"]),
        use_dora=bool(settings["use_dora"]),
        use_rslora=bool(settings["use_rslora"]),
    )

    # Concise summary for the launcher logs — Stage 4 eval reads this
    # directory unchanged, so it's useful to confirm the contents at
    # export time rather than deep inside the eval run.
    n_tensors = len(weights)
    print(
        f"Exported {n_tensors} LoRA tensors covering "
        f"{len(target_modules)} target modules ({', '.join(target_modules)}) "
        f"to {output_dir}",
        flush=True,
    )


def write_adapter(
    *,
    orbax_checkpoint: Any,
    output_dir: Any,
    base_model_name: str,
    target_modules: List[str],
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    variant: str = "standard",
    task_type: str = "CAUSAL_LM",
) -> None:
    """Trainer-facing wrapper for ``export``.

    Stage 3b's trainer (``scripts/train_lora_maxtext.py``) calls this with
    the per-run hyperparameters it already has in scope (rank, alpha,
    target_modules from the YAML's lora section; orbax_checkpoint from
    the final save; output_dir = checkpoints/seed_<N>/best/). The
    underlying ``export`` function takes a ``settings`` dict; we build
    that here so the trainer doesn't have to know about the dict
    plumbing or the optional DoRA / rsLoRA flags (we leave them off —
    the trainer has no path to enable either today).
    """
    settings: Dict[str, Any] = {
        "r": int(rank),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "target_modules": list(target_modules) if target_modules else None,
        "task_type": task_type,
        "variant": variant,
        "use_dora": False,
        "use_rslora": False,
    }
    export(
        orbax_path=Path(orbax_checkpoint),
        output_dir=Path(output_dir),
        base_model=base_model_name,
        settings=settings,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a MaxText orbax LoRA checkpoint to a HuggingFace PEFT "
            "adapter directory loadable via PeftModel.from_pretrained."
        ),
    )
    p.add_argument(
        "--orbax-path",
        required=True,
        type=Path,
        help="path to the orbax checkpoint directory (the leaf step "
             "directory; not the parent CheckpointManager root).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="destination directory for adapter_config.json + "
             "adapter_model.safetensors. Created if missing.",
    )
    p.add_argument(
        "--base-model",
        required=True,
        help="HF id of the base model the LoRA was trained against, "
             "e.g. google/medgemma-27b-text-it. Stored in "
             "adapter_config.json as base_model_name_or_path.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="optional YAML config (e.g. configs/lora_medgemma27b_tpu.yaml) "
             "to read r / lora_alpha / lora_dropout / target_modules / "
             "variant / task_type from. CLI flags override YAML values.",
    )
    p.add_argument("--lora-r", type=int, default=None, help="override lora.r")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="override lora.lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=None,
                   help="override lora.lora_dropout")
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="override lora.target_modules (HF projection names, e.g. "
             "q_proj v_proj). The exported adapter is the intersection "
             "of this list and the modules actually found in the orbax "
             "checkpoint.",
    )
    p.add_argument(
        "--task-type",
        default=None,
        help="override lora.task_type (default CAUSAL_LM).",
    )
    p.add_argument(
        "--lora-variant",
        default=None,
        choices=["standard", "dora", "rslora"],
        help="override lora.variant. Sets use_dora / use_rslora flags in "
             "adapter_config.json so PEFT loads the same forward path "
             "the trainer used.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    settings = _resolve_lora_settings(args)
    export(
        orbax_path=args.orbax_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        settings=settings,
    )


if __name__ == "__main__":
    main()
