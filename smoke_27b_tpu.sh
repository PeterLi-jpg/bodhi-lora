#!/bin/bash
# Smoke test (27B / TPU): boot the full pipeline on the actual MedGemma-27B
# target on a v6e-8 (or compatible) TPU host and run a handful of training
# steps end-to-end in <30 min.
#
# Why this exists in addition to smoke.sh:
#   smoke.sh runs gemma-3n-E4B-it (4B) on a local box. A passing 4B smoke
#   says "the pipeline does not crash on a small model" but does NOT
#   exercise the failure modes that only appear at the 27B scale:
#     - FSDP + LoRA sharding edge cases
#     - optimizer state for trainable LoRA params (#39795-class bugs)
#     - quantization / dtype behavior at 27B
#     - tensor-parallel topology on real TPU chips
#   This script catches those before a full multi-hour cluster run.
#
# Required env:
#   HF_TOKEN              MedGemma is gated; must accept terms first.
#   TPU access            PJRT_DEVICE=TPU set, or /dev/vfio populated
#                         (i.e. running on a Cloud TPU VM, e.g. v6e-8).
#
# This is a single-shot variant of the production multi-seed launcher
# (tpu/launch_multiseed.sh), not a replacement. Per #66, eval is skipped
# (only the training loop needs to complete), save_strategy is "no" so we
# don't burn time on checkpointing, and the grader is a small ungated
# model so the smoke doesn't need a second gated access for filter_traces.
#
# Runs in ~30 min on v6e-8.

set -euo pipefail

cd "$(dirname "$0")"

# ── Preflight: hard-fail if we're not on a TPU host. ─────────────────────────
# This script burns 27B-scale time on a TPU; running it on a CPU box would
# either OOM or sit forever in XLA compile. Catch the misuse up front rather
# than 20 minutes into a run.
if [ "${PJRT_DEVICE:-}" != "TPU" ] && [ ! -d /dev/vfio ]; then
    echo "ERROR: smoke_27b_tpu.sh requires a TPU host (v6e-8 or compatible)."
    echo "For 4B local smoke, run smoke.sh instead."
    exit 1
fi

MODEL="google/medgemma-27b-text-it"
# Small ungated grader so the smoke doesn't need a second gated access.
GRADER="Qwen/Qwen2.5-0.5B-Instruct"
N_EXAMPLES="${N_EXAMPLES:-5}"
RUNTIME_CONFIG="data/sft/smoke_27b/runtime_train_config.yaml"

# MedGemma is gated; same check as smoke.sh.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required for gated model $MODEL"
    echo "  1. Accept terms at https://huggingface.co/$MODEL"
    echo "  2. export HF_TOKEN=hf_..."
    echo "  3. rerun bash smoke_27b_tpu.sh"
    exit 1
fi

echo "=== smoke_27b_tpu | $(date) ==="
echo "model:   $MODEL"
echo "grader:  $GRADER"
echo "samples: $N_EXAMPLES"
echo

mkdir -p data/sft/smoke_27b logs

# Generate a fast-smoke runtime config from the production 27B TPU template.
# Override only the fields that matter for "did the loop boot and step": one
# epoch, short seq, no grad accum, log every step, no checkpoint saves, no eval.
# Everything else (LoRA r/targets, batch sizing, sharding) we keep identical
# to production so 27B-specific bugs still surface here.
python - <<PY
from pathlib import Path

import yaml

cfg_path = Path("configs/lora_medgemma27b_tpu.yaml")
out_path = Path("$RUNTIME_CONFIG")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["training"]["num_epochs"] = 1
cfg["training"]["max_seq_length"] = 512
cfg["training"]["gradient_accumulation_steps"] = 1
cfg["training"]["logging_steps"] = 1
cfg["training"]["save_strategy"] = "no"
cfg["training"]["eval_strategy"] = "no"

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote runtime config -> {out_path}")
PY

echo "--- 0/3: preflight ---"
python scripts/preflight.py --models "$MODEL" "$GRADER"

echo "--- 1/3: download data ---"
python scripts/download_data.py

echo "--- 2/3: generate $N_EXAMPLES BOHDI traces ---"
python scripts/generate_traces.py \
    --model "$MODEL" \
    --datasets healthbench_hard \
    --output data/sft/smoke_27b/raw_traces.jsonl \
    --use-bodhi \
    --max-examples "$N_EXAMPLES"

echo "--- 3a/3: grade and filter (production threshold to exercise the gate) ---"
# Match run_multi_seed.sh's MIN_SCORE=0.4 and VAL_RATIO=0.1 so the smoke
# actually exercises the same gate + split as production. With small
# N_EXAMPLES it's possible 0 traces survive; that's fine as smoke output
# (train_lora.py errors loudly on empty train.jsonl) and surfaces grader
# regressions before cluster time burns.
python scripts/filter_traces.py \
    --input data/sft/smoke_27b/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl \
    --grader-model "$GRADER" \
    --output-dir data/sft/smoke_27b \
    --min-score 0.4 \
    --val-ratio 0.1

echo "--- 3b/3: train a few steps on the smoke set ---"
# Same rationale as smoke.sh: do NOT use `accelerate launch`. accelerate's
# tpu_launcher always xmp.spawn()s addressable_device_count() processes,
# which is incompatible with our single-process SPMD design (one python
# process drives all 8 chips, sharding via mark_sharding). Plain `python`
# with PJRT_DEVICE=TPU lets HF Trainer's native XLA path pick up SPMD.
#
# --train-file / --val-file pin the smoke to its own filtered output.
# Without them, train_lora.py reads data.train_file from the runtime
# yaml, which inherits configs/lora_medgemma27b_tpu.yaml's production
# paths (data/sft/{train,val}.jsonl) and silently trains on whatever
# stale full-pipeline data happens to be on disk — observed on
# bohdi-lora-v4 where 838 leftover examples were used instead of the
# 4-example smoke set.
python scripts/train_lora.py \
    --config "$RUNTIME_CONFIG" \
    --train-file data/sft/smoke_27b/train.jsonl \
    --val-file data/sft/smoke_27b/val.jsonl

echo
echo "=== smoke_27b_tpu PASSED | $(date) ==="
echo "artifacts:"
echo "  $RUNTIME_CONFIG"
echo "  data/sft/smoke_27b/raw_traces.jsonl"
echo "  data/sft/smoke_27b/{train,val}.jsonl"
