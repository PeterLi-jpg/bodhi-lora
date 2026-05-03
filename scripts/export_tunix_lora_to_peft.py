"""Convert a tunix + qwix LoRA orbax checkpoint to a HuggingFace PEFT adapter.

Stage 3 of the BOHDI pipeline now trains LoRA via Google's tunix on top of
qwix-wrapped Gemma-3 models.  Stage 4 (vLLM-TPU eval) hasn't moved with us
and still expects the standard PEFT layout: a directory with an
``adapter_config.json`` and an ``adapter_model.safetensors`` keyed under
``base_model.model.model.layers.<i>.self_attn.<proj>.lora_{A,B}.weight``.
This script bridges the two: it reads a tunix orbax checkpoint, walks the
restored pytree for the qwix-injected LoRA factors, and writes a PEFT
adapter directory.

Layout assumptions (qwix-on-tunix-Gemma-3)
------------------------------------------
qwix wraps the base weight ``self.w`` of every targeted Einsum / DotGeneral
module by registering two extra params ``<weight>_lora_a`` and
``<weight>_lora_b``.  In tunix's Gemma-3 attention block the relevant
einsums are:

* ``layers.<i>.attn.q_einsum.w``    (einsum ``BTD,NDH->BTNH``)
* ``layers.<i>.attn.kv_einsum.w``   (einsum ``BSD,CKDH->CBSKH``)
* ``layers.<i>.attn.qkv_einsum.w``  (only when ``num_heads == num_kv_heads``)

Per qwix's einsum LoRA decomposition (see ``qwix._src.providers.lora.
_parse_einsum_str_for_lora``), the contracting axis becomes the lora_a
input dim and every remaining rhs axis ends up on lora_b.  Concretely the
shapes that come out of an orbax restore are:

* ``q_einsum.w_lora_a``    : ``(D, R)``       (D = embed_dim)
* ``q_einsum.w_lora_b``    : ``(R, N, H)``    (N = num_heads, H = head_dim)
* ``kv_einsum.w_lora_a``   : ``(D, R)``
* ``kv_einsum.w_lora_b``   : ``(R, 2, K, H)`` (axis 1 is [k, v], K = num_kv_heads)
* ``qkv_einsum.w_lora_a``  : ``(D, R)``
* ``qkv_einsum.w_lora_b``  : ``(R, 3, N, H)`` (axis 1 is [q, k, v])

The kv_einsum and qkv_einsum cases share a single lora_a across the
packed projections and split the output along axis 1 of lora_b.  Splitting
into separate HF q_proj / k_proj / v_proj adapters means the lora_A matrix
is duplicated across the unpacked adapters (identical weights) and the
lora_B matrix is sliced.  The resulting per-adapter LoRA is mathematically
equivalent to the packed update at inference time.

Output layout
-------------
PEFT stores ``lora_A.weight`` as ``(r, in_features)`` and ``lora_B.weight``
as ``(out_features, r)`` to match ``torch.nn.Linear.weight`` layout.  We
transpose lora_a from qwix convention ``(D, R) -> (R, D)`` and reshape +
transpose lora_b ``(R, N, H) -> (N*H, R)``.

This module deliberately avoids importing ``torch`` and ``peft`` so it can
run inside the same TPU venv that runs the trainer (which today does not
ship torch).  Tensors are written via ``safetensors.numpy.save_file``;
the on-disk container is identical to ``safetensors.torch.save_file``,
so PEFT loads it the same way.  bf16 round-trips intact via
``ml_dtypes.bfloat16`` arrays, which safetensors recognises natively.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# Orbax / qwix / tunix produce paths like
# ``layers.<i>.attn.q_einsum.w_lora_a`` once flattened with "/".  We accept
# any wrapping prefix (``params/``, ``model/``, etc.); only the layer
# index and einsum tag matter.
_LAYER_RE = re.compile(r"^layers?[_.]?(\d+)$", re.IGNORECASE)

# qwix names its params ``<weight>_lora_a`` / ``<weight>_lora_b``.  In the
# tunix Einsum the weight is ``w``, so the leaves are ``w_lora_a`` and
# ``w_lora_b``.  We allow a few terminal-wrapper variants (``kernel`` /
# ``value`` / ``raw_value``) since orbax may surface NNX boxed params.
_LORA_LEAF_RE = re.compile(r"^w_lora_(a|b)$", re.IGNORECASE)
_TERMINAL_WRAPPERS = {"value", "raw_value", "kernel", "weight", "params"}

# Einsum tags we recognise.  ``q_einsum`` is q-only; ``kv_einsum`` packs
# k+v on axis 0 of its weight; ``qkv_einsum`` packs q+k+v.
_EINSUM_TAGS = {"q_einsum", "kv_einsum", "qkv_einsum"}


def _flatten_pytree(
    tree: Any,
    prefix: Tuple[str, ...] = (),
) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    """Yield ``(path, leaf)`` for every leaf in nested dict/list/tuple."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten_pytree(v, prefix + (str(k),))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _flatten_pytree(v, prefix + (str(i),))
    else:
        yield prefix, tree


def _looks_like_array(leaf: Any) -> bool:
    """Cheap duck-type check for orbax-restored array leaves."""
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _to_numpy(leaf: Any) -> np.ndarray:
    """Best-effort conversion to numpy.  Avoids a hard jax import."""
    if isinstance(leaf, np.ndarray):
        return leaf
    return np.asarray(leaf)


def _identify_einsum_lora_leaf(
    path: Tuple[str, ...],
) -> Optional[Tuple[int, str, str]]:
    """Recognise a tunix einsum LoRA leaf in a flattened orbax path.

    Returns ``(layer_idx, einsum_tag, "a" | "b")`` if ``path`` names one of
    ``layers.<i>.attn.<einsum_tag>.w_lora_<ab>`` (with any wrapping
    prefix), or ``None`` otherwise.

    Wrapping leaves like ``("..., 'w_lora_a', 'value')`` are tolerated:
    NNX boxed params land that way after a default orbax restore.
    """
    layer_idx: Optional[int] = None
    einsum_tag: Optional[str] = None
    ab: Optional[str] = None
    # Set after seeing a bare "layers" / "layer" segment; tunix's orbax
    # save writes the layer index as a SEPARATE segment after that, e.g.
    # ``("layers", "0", "attn", "q_einsum", "w_lora_a", "value")``. The
    # original ``_LAYER_RE`` only matched compound segments like
    # ``"layer_0"`` / ``"layer.0"`` and missed this layout.
    expect_bare_index = False

    for seg in path:
        # Skip terminal wrappers (NNX 'value', Flax 'kernel', etc.).  None
        # of these collide with the 'w_lora_<a|b>' lora-leaf token, so a
        # plain set membership check is enough.
        if seg.lower() in _TERMINAL_WRAPPERS:
            continue

        # tunix layout: bare "layers" segment followed by a digit segment.
        if seg.lower() in ("layers", "layer") and layer_idx is None:
            expect_bare_index = True
            continue
        if expect_bare_index and layer_idx is None:
            try:
                layer_idx = int(seg)
                expect_bare_index = False
                continue
            except ValueError:
                expect_bare_index = False
                # fall through to other checks for this seg

        # Compound layer segment (legacy / other layouts: "layer_0", "layers.0").
        m = _LAYER_RE.match(seg)
        if m and layer_idx is None:
            layer_idx = int(m.group(1))
            continue

        # Einsum tag?  (Lower-cased so a future ``Q_einsum`` doesn't slip
        # through silently.)
        seg_lower = seg.lower()
        if seg_lower in _EINSUM_TAGS and einsum_tag is None:
            einsum_tag = seg_lower
            continue

        # LoRA factor leaf?
        m = _LORA_LEAF_RE.match(seg)
        if m and ab is None:
            ab = m.group(1).lower()
            continue

    if layer_idx is None or einsum_tag is None or ab is None:
        return None
    return layer_idx, einsum_tag, ab


def _peft_key(layer_idx: int, hf_proj: str, ab: str) -> str:
    """Return the safetensors key PEFT writes for a given LoRA slot."""
    return (
        f"base_model.model.model.layers.{layer_idx}."
        f"self_attn.{hf_proj}.lora_{ab}.weight"
    )


def _convert_lora_a(arr: np.ndarray) -> np.ndarray:
    """Convert qwix lora_a ``(D, R)`` to PEFT lora_A.weight ``(R, D)``."""
    if arr.ndim != 2:
        raise ValueError(
            f"expected lora_a to be 2-D (D, R); got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr.T)


def _convert_q_lora_b(arr: np.ndarray) -> np.ndarray:
    """qwix q_einsum lora_b ``(R, N, H)`` -> PEFT lora_B.weight ``(N*H, R)``."""
    if arr.ndim != 3:
        raise ValueError(
            f"expected q_einsum lora_b to be 3-D (R, N, H); got shape {arr.shape}"
        )
    r, n, h = arr.shape
    # (R, N, H) -> (N, H, R) -> (N*H, R)
    transposed = np.transpose(arr, (1, 2, 0)).reshape(n * h, r)
    return np.ascontiguousarray(transposed)


def _split_kv_lora_b(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split qwix kv_einsum lora_b ``(R, 2, K, H)`` -> (k, v) PEFT lora_B.

    Returns ``(k_lora_B, v_lora_B)``, each shaped ``(K*H, R)``.  Axis 1 of
    the input is ``[k, v]`` to match tunix's ``kv_einsum`` weight whose
    rhs einsum chars are ``CKDH`` with C=2 the k/v split (see
    ``tunix.models.gemma3.model.Attention.kv_einsum``).
    """
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError(
            f"expected kv_einsum lora_b to be 4-D (R, 2, K, H); "
            f"got shape {arr.shape}"
        )
    r, _two, k, h = arr.shape
    k_part = np.transpose(arr[:, 0], (1, 2, 0)).reshape(k * h, r)
    v_part = np.transpose(arr[:, 1], (1, 2, 0)).reshape(k * h, r)
    return np.ascontiguousarray(k_part), np.ascontiguousarray(v_part)


def _split_qkv_lora_b(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split qwix qkv_einsum lora_b ``(R, 3, N, H)`` -> (q, k, v) PEFT lora_B.

    Used only when the model has ``num_heads == num_kv_heads`` and tunix
    fuses q/k/v into a single ``qkv_einsum``.  Axis 1 is ``[q, k, v]`` to
    match the ``jnp.stack([q, k, v], axis=0)`` packing in
    ``tunix.models.gemma3.params_safetensors._make_preprocess_fn``.
    """
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(
            f"expected qkv_einsum lora_b to be 4-D (R, 3, N, H); "
            f"got shape {arr.shape}"
        )
    r, _three, n, h = arr.shape
    q_part = np.transpose(arr[:, 0], (1, 2, 0)).reshape(n * h, r)
    k_part = np.transpose(arr[:, 1], (1, 2, 0)).reshape(n * h, r)
    v_part = np.transpose(arr[:, 2], (1, 2, 0)).reshape(n * h, r)
    return (
        np.ascontiguousarray(q_part),
        np.ascontiguousarray(k_part),
        np.ascontiguousarray(v_part),
    )


def collect_lora_weights(
    pytree: Any,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Walk ``pytree`` and return ``(weights, target_modules)``.

    ``weights`` maps PEFT-formatted safetensors keys to numpy arrays.
    ``target_modules`` is the sorted list of HF projection names actually
    present in the checkpoint (used to populate ``adapter_config.json``).
    """
    # Collect the matched leaves first so we can verify lora_a / lora_b
    # arrive in pairs and emit clear errors if not.
    found: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}

    for path, leaf in _flatten_pytree(pytree):
        if not _looks_like_array(leaf):
            continue
        ident = _identify_einsum_lora_leaf(path)
        if ident is None:
            continue
        layer_idx, einsum_tag, ab = ident
        slot = found.setdefault((layer_idx, einsum_tag), {})
        if ab in slot:
            raise ValueError(
                f"duplicate lora_{ab} for layer {layer_idx} "
                f"einsum {einsum_tag!r}; clashing source path: "
                f"{'/'.join(path)}"
            )
        slot[ab] = _to_numpy(leaf)

    if not found:
        raise ValueError(
            "no qwix LoRA leaves found in the orbax pytree. The path is "
            "either empty or uses naming this script doesn't recognise. "
            "Expected leaves at "
            "layers.<i>.attn.{q_einsum,kv_einsum,qkv_einsum}.w_lora_{a,b}."
        )

    weights: Dict[str, np.ndarray] = {}
    seen_projections: set[str] = set()

    for (layer_idx, einsum_tag), slot in sorted(found.items()):
        if "a" not in slot or "b" not in slot:
            missing = "a" if "a" not in slot else "b"
            raise ValueError(
                f"layer {layer_idx} {einsum_tag!r}: missing lora_{missing} "
                f"factor (have {sorted(slot.keys())}). "
                "Both factors are required."
            )
        a = slot["a"]
        b = slot["b"]

        if einsum_tag == "q_einsum":
            a_peft = _convert_lora_a(a)
            b_peft = _convert_q_lora_b(b)
            weights[_peft_key(layer_idx, "q_proj", "A")] = a_peft
            weights[_peft_key(layer_idx, "q_proj", "B")] = b_peft
            seen_projections.add("q_proj")

        elif einsum_tag == "kv_einsum":
            # Same lora_a feeds both k and v adapters; lora_b splits along
            # axis 1.  Ship two independent copies of the lora_A weight so
            # PEFT can load each adapter without referring across modules.
            a_peft = _convert_lora_a(a)
            k_b, v_b = _split_kv_lora_b(b)
            weights[_peft_key(layer_idx, "k_proj", "A")] = a_peft
            weights[_peft_key(layer_idx, "k_proj", "B")] = k_b
            weights[_peft_key(layer_idx, "v_proj", "A")] = a_peft.copy()
            weights[_peft_key(layer_idx, "v_proj", "B")] = v_b
            seen_projections.update({"k_proj", "v_proj"})

        elif einsum_tag == "qkv_einsum":
            a_peft = _convert_lora_a(a)
            q_b, k_b, v_b = _split_qkv_lora_b(b)
            weights[_peft_key(layer_idx, "q_proj", "A")] = a_peft
            weights[_peft_key(layer_idx, "q_proj", "B")] = q_b
            weights[_peft_key(layer_idx, "k_proj", "A")] = a_peft.copy()
            weights[_peft_key(layer_idx, "k_proj", "B")] = k_b
            weights[_peft_key(layer_idx, "v_proj", "A")] = a_peft.copy()
            weights[_peft_key(layer_idx, "v_proj", "B")] = v_b
            seen_projections.update({"q_proj", "k_proj", "v_proj"})

        else:  # pragma: no cover -- _identify guard already filters
            raise ValueError(f"unrecognised einsum tag {einsum_tag!r}")

    return weights, sorted(seen_projections)


def _resolve_step_candidates(path: Path) -> List[Path]:
    """Return candidate leaf step dirs in newest-first order.

    tunix's PeftTrainer uses ``orbax.CheckpointManager``, which writes
    ``<root>/<step>/`` subdirectories (one per saved step, integer-named).
    The exporter's underlying ``PyTreeCheckpointer.restore`` needs the
    leaf step dir, not the parent root.

    Returns:
      - If ``path`` is itself a leaf (no int-named children), a single-element
        list ``[path]``.
      - If ``path`` contains int-named subdirs (the CheckpointManager layout),
        the list of those leaves in DECREASING step order. Caller can try
        them one by one if loading the highest fails -- this is the
        "incomplete final save" defence: when training ends right at a save
        boundary, the trainer may exit before orbax finalizes the manifest,
        leaving the highest-numbered checkpoint structurally invalid even
        though all earlier saves are complete.
    """
    abspath = Path(path).resolve()
    if not abspath.is_dir():
        raise FileNotFoundError(
            f"--orbax-dir {abspath} does not exist or is not a directory."
        )

    int_steps: List[int] = []
    for child in abspath.iterdir():
        if child.is_dir():
            try:
                int_steps.append(int(child.name))
            except ValueError:
                continue

    if int_steps:
        # Newest first; caller falls back to older steps if newest is broken.
        int_steps.sort(reverse=True)
        return [abspath / str(s) for s in int_steps]

    # No integer-named children -- assume ``path`` is already the leaf.
    return [abspath]


def _resolve_step_dir(path: Path) -> Path:
    """Pick the highest int-named step (or the leaf if there's only one).

    Backward-compat wrapper around ``_resolve_step_candidates`` for callers
    that want a single Path rather than a list.
    """
    return _resolve_step_candidates(path)[0]


def load_orbax_checkpoint(path: Path) -> Any:
    """Restore a tunix orbax checkpoint into a nested dict.

    ``PyTreeCheckpointer`` is used because tunix saves a plain pytree of
    params with no Composite / metadata handlers, so the simplest restore
    path round-trips cleanly.

    If the highest-numbered step is broken (tunix's "incomplete final
    save" -- trainer exits before orbax finalizes the manifest), fall back
    to the next-highest step. Earlier saves were taken DURING training and
    have had time to fully flush, so they're structurally sound.
    """
    import orbax.checkpoint as ocp

    candidates = _resolve_step_candidates(path)
    restorer = ocp.PyTreeCheckpointer()
    last_error: Optional[Exception] = None
    for leaf in candidates:
        # tunix's PeftTrainer saves a COMPOSITE checkpoint:
        #   <leaf>/
        #     _CHECKPOINT_METADATA
        #     model_params/  <- has manifest.ocdbt, the actual pytree
        #     optimizer_state/  (we don't need this for inference export)
        # PyTreeCheckpointer.restore wants the inner pytree dir (model_params),
        # not the composite root. Earlier code passed the composite root and
        # hit "No structure could be identified" because the loader couldn't
        # find manifest.ocdbt at the top level. If model_params subdir exists,
        # use it; otherwise fall back to the leaf (legacy flat layout).
        params_dir = leaf / "model_params"
        target = params_dir if params_dir.is_dir() else leaf
        try:
            print(f"[exporter] loading orbax checkpoint from {target}", flush=True)
            return restorer.restore(str(target))
        except FileNotFoundError as exc:
            # Common tunix failure: highest step's manifest never finalized
            # because the trainer exited at end-of-training right after
            # save_checkpoint() was queued. Earlier saves are complete.
            print(
                f"[exporter] {target} is not a complete checkpoint "
                f"({type(exc).__name__}: {exc}); trying earlier step...",
                flush=True,
            )
            last_error = exc
            continue
    raise FileNotFoundError(
        f"No loadable orbax checkpoint at {path}. Tried {len(candidates)} "
        f"candidate step dirs (newest first); all failed. Last error: {last_error}"
    )


def build_adapter_config(
    *,
    base_model: str,
    target_modules: List[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Dict[str, Any]:
    """Build the PEFT ``adapter_config.json`` body.

    Hand-rolling the dict (rather than going through ``peft.LoraConfig``)
    keeps this script importable without torch/peft installed; the
    trainer's TPU venv ships neither.  The field set matches what
    ``peft.LoraConfig.save_pretrained`` writes for the standard LoRA
    case, which is what ``PeftModel.from_pretrained`` validates against.
    """
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": int(r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": sorted(target_modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "modules_to_save": None,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
    }


def write_peft_adapter(
    *,
    output_dir: Path,
    weights: Dict[str, np.ndarray],
    target_modules: List[str],
    base_model: str,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    """Write ``adapter_config.json`` + ``adapter_model.safetensors``."""
    from safetensors.numpy import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_adapter_config(
        base_model=base_model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    (output_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    )

    # safetensors requires contiguous arrays; ``collect_lora_weights``
    # already enforces that via ``np.ascontiguousarray``.
    save_file(weights, str(output_dir / "adapter_model.safetensors"))


def export(
    *,
    orbax_dir: Path,
    output_dir: Path,
    base_model: str,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    """End-to-end: load orbax, collect LoRA weights, write the PEFT dir."""
    pytree = load_orbax_checkpoint(orbax_dir)
    weights, target_modules = collect_lora_weights(pytree)

    if not target_modules:
        raise ValueError(
            "checkpoint contained no recognised LoRA modules. "
            "Expected q_proj / k_proj / v_proj derived from "
            "q_einsum / kv_einsum / qkv_einsum leaves."
        )

    write_peft_adapter(
        output_dir=output_dir,
        weights=weights,
        target_modules=target_modules,
        base_model=base_model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    print(
        f"exported {len(weights)} LoRA tensors covering "
        f"{len(target_modules)} target modules "
        f"({', '.join(target_modules)}) to {output_dir}",
        flush=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a tunix + qwix LoRA orbax checkpoint to a HuggingFace "
            "PEFT adapter directory loadable via PeftModel.from_pretrained."
        ),
    )
    p.add_argument(
        "--orbax-dir",
        required=True,
        type=Path,
        help="path to the tunix orbax checkpoint. Accepts either the leaf "
             "step directory (e.g. checkpoints/seed_42/orbax/2) or the "
             "parent CheckpointManager root (e.g. checkpoints/seed_42/orbax); "
             "if the root is given the exporter picks the highest-numbered "
             "step subdirectory automatically.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="destination directory for adapter_config.json + "
             "adapter_model.safetensors. Created if missing.",
    )
    p.add_argument(
        "--base-model-name",
        required=True,
        help="HF id of the base model the LoRA was trained against, e.g. "
             "google/medgemma-27b-text-it. Stored in adapter_config.json "
             "as base_model_name_or_path.",
    )
    p.add_argument("--r", type=int, default=8, help="LoRA rank (default 8).")
    p.add_argument(
        "--alpha",
        type=int,
        default=16,
        help="LoRA alpha (default 16).",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="LoRA dropout used during training (default 0.0).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    export(
        orbax_dir=args.orbax_dir,
        output_dir=args.output_dir,
        base_model=args.base_model_name,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
