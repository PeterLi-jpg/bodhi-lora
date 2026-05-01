"""HuggingFace -> MaxText/orbax converter for MedGemma-27B.

MedGemma-27B-text-it (google/medgemma-27b-text-it) is a Gemma-3-27B fine-tune
trained on medical text. It loads via `Gemma3ForCausalLM`, which gives a
state_dict with parameter names like `model.layers.{i}.self_attn.q_proj.weight`
and `model.embed_tokens.weight` (no `language_model.` prefix).

MaxText's existing `gemma3-27b` converter targets the multimodal
`Gemma3ForConditionalGeneration` checkpoint, where the same parameters are
nested under a `language_model.` prefix (e.g. `model.language_model.layers.X`).
So we wrap MaxText's converter and re-prefix the loaded HF state_dict before
the conversion's parameter mapping runs.

Layer-mapping summary:
    text-only HF (this script's input)        ->  multimodal HF (MaxText expects)
    model.embed_tokens.weight                  ->  model.language_model.embed_tokens.weight
    model.norm.weight                          ->  model.language_model.norm.weight
    model.layers.{i}.<x>                       ->  model.language_model.layers.{i}.<x>

Vision-tower / multi-modal-projector keys are absent from the text-only
checkpoint, which is expected. MaxText's mapping for `gemma3-27b` will look
for those keys, so we run conversion with `use_multimodal=false` to disable
the vision head (Gemma3 supports a text-only inference variant in MaxText).

CLI:
    python scripts/convert_medgemma_to_maxtext.py \
        --hf-path google/medgemma-27b-text-it \
        --output  ~/.cache/maxtext/medgemma-27b/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List


# The text-only -> multimodal prefix remap MaxText needs in order to reuse
# its existing gemma3-27b parameter mapping.
TEXT_ONLY_TO_MULTIMODAL_PREFIXES = (
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("model.layers.", "model.language_model.layers."),
)


def remap_text_only_to_multimodal(state_dict: dict) -> dict:
    """Re-prefix a `Gemma3ForCausalLM` state_dict so MaxText's mapping accepts it.

    Only `model.layers.*`, `model.embed_tokens.*`, and `model.norm.*` get
    rewritten; `lm_head.weight` and any other top-level keys pass through
    unchanged. This matches the structure produced by
    `Gemma3ForConditionalGeneration` minus the vision-tower / projector keys.
    """
    remapped: dict = {}
    for old_key, value in state_dict.items():
        new_key = old_key
        for src, dst in TEXT_ONLY_TO_MULTIMODAL_PREFIXES:
            if new_key.startswith(src):
                new_key = dst + new_key[len(src):]
                break
        remapped[new_key] = value
    return remapped


def _ensure_maxtext_importable() -> None:
    """Add `third_party/maxtext/src` to sys.path if MaxText isn't installed.

    Unit 1 vendored MaxText under `third_party/maxtext/`. We don't require
    `pip install` of MaxText into this project's main env (it has heavy JAX/
    TPU deps); instead we make its `src/` importable when present.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "third_party" / "maxtext" / "src"
    if candidate.is_dir():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _maxtext_base_config_path() -> str:
    """Resolve the absolute path to MaxText's base.yml under third_party/.

    Falls back to the relative form if vendoring isn't in place yet, so
    `_build_maxtext_args` stays unit-testable without the vendored tree.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "third_party" / "maxtext" / "src" / "maxtext" / "configs" / "base.yml"
    if candidate.is_file():
        return str(candidate)
    return "src/maxtext/configs/base.yml"


def _build_maxtext_args(
    model_name: str,
    output: str,
    hf_token: str | None,
    extra: Iterable[str],
) -> List[str]:
    """Assemble the args MaxText's `to_maxtext.main()` expects.

    MaxText's main parses these as `key=value` overrides on top of a base
    config file (`maxtext/configs/base.yml`, the canonical default).
    """
    args: List[str] = [
        sys.argv[0],  # MaxText's pyconfig keeps arg[0] for argparse parity
        _maxtext_base_config_path(),
        f"model_name={model_name}",
        f"base_output_directory={output}",
        # CPU host conversion. TPU/GPU not required and would fail on a laptop.
        "hardware=cpu",
        "skip_jax_distributed_system=True",
        # Unstacked layers: training entrypoint (Unit 7) expects the per-layer
        # form rather than the scanned-stacked form. Inference loaders also
        # default to this.
        "scan_layers=False",
        # Text-only: skip the SigLIP vision tower / mm projector.
        "use_multimodal=False",
    ]
    if hf_token:
        args.append(f"hf_access_token={hf_token}")
    args.extend(extra)
    return args


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a MedGemma-27B HuggingFace checkpoint into a MaxText/orbax "
            "checkpoint by reusing MaxText's gemma3-27b mapping with a thin "
            "text-only -> multimodal key remap."
        ),
    )
    parser.add_argument(
        "--hf-path",
        default="google/medgemma-27b-text-it",
        help=(
            "HuggingFace repo ID or local directory of the MedGemma checkpoint. "
            "Defaults to google/medgemma-27b-text-it."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination for the orbax checkpoint. Local path or gs:// URI. "
            "MaxText writes the actual checkpoint under {output}/0/items."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="gemma3-27b",
        help=(
            "MaxText model_name to drive the architecture config + parameter "
            "mapping. Override only if you know what you're doing (MedGemma "
            "27B is a Gemma-3-27B fine-tune)."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HF_AUTH_TOKEN"),
        help=(
            "HuggingFace access token. Falls back to $HF_TOKEN, then "
            "$HF_AUTH_TOKEN. Needed for gated MedGemma weights unless the "
            "host is already authenticated via `huggingface-cli login`."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Output orbax checkpoint dtype.",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help=(
            "Use MaxText's lazy-load path. Lower RAM, but unsupported when "
            "use_multimodal=True (we leave it off here since text-only is the "
            "expected mode)."
        ),
    )
    parser.add_argument(
        "--simulated-cpu-devices",
        type=int,
        default=16,
        help=(
            "Number of virtual CPU devices for orbax sharding. 16 matches "
            "MaxText's default and tends to load cleanly on a v5e/v6e TPU pod."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional key=value overrides forwarded to MaxText pyconfig.",
    )
    return parser.parse_args(argv)


def run(ns: argparse.Namespace) -> int:
    """Drive the conversion. Returns the exit code MaxText would have used."""
    _ensure_maxtext_importable()

    # Imported lazily so `--help` works even on a host without JAX/MaxText.
    try:
        from maxtext.checkpoint_conversion import to_maxtext  # type: ignore[import-not-found]
        from maxtext.checkpoint_conversion.utils import utils as ckpt_utils  # type: ignore[import-not-found]
    except ImportError as e:
        sys.stderr.write(
            "ERROR: MaxText is not importable. Make sure third_party/maxtext "
            "is vendored (Unit 1) or install MaxText into the active env.\n"
            f"Underlying ImportError: {e}\n",
        )
        return 2

    # Wrap MaxText's HF loader so the state_dict it returns has multimodal-
    # style keys, matching what MaxText's gemma3-27b mapping expects.
    original_loader = ckpt_utils.load_hf_dict_from_transformers

    def _patched_loader(model_id, token, revision=None, dtype="auto"):
        sd = original_loader(model_id, token=token, revision=revision, dtype=dtype)
        return remap_text_only_to_multimodal(sd)

    ckpt_utils.load_hf_dict_from_transformers = _patched_loader
    # to_maxtext imports the helper by name at module-load time (it does
    # `from ...utils.utils import load_hf_dict_from_transformers`), so the
    # rebinding above wouldn't reach the call site. Patch the bound name too.
    if hasattr(to_maxtext, "load_hf_dict_from_transformers"):
        to_maxtext.load_hf_dict_from_transformers = _patched_loader

    args = _build_maxtext_args(
        model_name=ns.model_name,
        output=ns.output,
        hf_token=ns.hf_token,
        extra=[a for a in ns.extra if a],
    )

    try:
        to_maxtext.main(
            args=args,
            lazy_load_tensors=ns.lazy_load,
            eager_load_method="transformers",  # required for gemma3 mapping
            hf_model_path=ns.hf_path,
            save_dtype=ns.save_dtype,
            simulated_cpu_devices_count=ns.simulated_cpu_devices,
        )
    finally:
        # Restore the loader so re-imports during testing aren't sticky.
        ckpt_utils.load_hf_dict_from_transformers = original_loader
        if hasattr(to_maxtext, "load_hf_dict_from_transformers"):
            to_maxtext.load_hf_dict_from_transformers = original_loader

    return 0


def main(argv: List[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
