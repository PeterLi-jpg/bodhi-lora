"""Pre-stage a HuggingFace gemma3 model's safetensors into the local HF cache.

The tunix launcher (tpu/launch_5seeds_tunix.sh) calls this BEFORE the trainer
runs so safetensors land on the TPU's tmpfs (HF_HOME=/dev/shm/hf) up front.
The MaxText pipeline used to get this caching as a side-effect of
convert_medgemma_to_maxtext.py's `from_pretrained` call; the tunix path drops
that conversion step, so without an explicit prefetch the trainer would block
on a multi-GB download mid-pipeline.

We only fetch text-only weights and metadata (`*.safetensors`, `*.json`,
`*.model`). Vision-tower binaries and tokenizer extras we don't need are
skipped via allow_patterns.

Usage:
    python scripts/prefetch_gemma_safetensors.py \
        --model google/medgemma-27b-text-it
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download


# Text-only LoRA training only needs the model weights, the config / tokenizer
# JSON, and the SentencePiece `.model` file. Vision-tower files (preprocessor,
# image embeddings) are intentionally excluded.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="google/medgemma-27b-text-it",
        help="HuggingFace model id to prefetch (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (branch / tag / commit). Defaults to the repo's default branch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # HF_TOKEN is required for gated repos like google/medgemma-*. We pass it
    # explicitly rather than relying on `HfFolder` so a missing token fails
    # loudly with the env-var name in the error.
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print(
            "error: HF_TOKEN environment variable is not set; "
            "gated gemma3 repos require a token.",
            file=sys.stderr,
        )
        return 1

    snapshot_path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
        token=token,
    )

    # Print the local snapshot path on stdout so the launcher can capture it
    # if it needs to reference the on-disk location directly.
    print(snapshot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
