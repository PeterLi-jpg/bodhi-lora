#!/bin/bash
# Run filter -> train -> eval across multiple seeds, reusing one set of
# generated traces. Trace generation is the expensive stage (~40h on H100 for
# the 27B run), so we pay that cost once and vary only the downstream
# randomness: filter shuffle, LoRA initialization, training order.
#
# Usage:
#   SEEDS="42 7 13 99 101" bash scripts/run_multi_seed.sh
# or (defaults to 5 seeds):
#   bash scripts/run_multi_seed.sh
#
# Expects data/sft/raw_traces.jsonl to already exist (run
# scripts/generate_traces.py first, or slurm/generate_traces.sh).

set -euo pipefail
cd "${BOHDI_DIR:-$(dirname "$0")/..}"

SEEDS="${SEEDS:-42 7 13 99 101}"
CONFIG="${CONFIG:-configs/lora_medgemma27b.yaml}"
MODEL="${MODEL:-google/medgemma-27b-text-it}"
IDS="${IDS:-data/raw/hard_200_sample_ids.json}"
GRADER="${GRADER:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
MIN_SCORE="${MIN_SCORE:-0.4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

RAW_TRACES="data/sft/raw_traces.jsonl"
if [ ! -f "$RAW_TRACES" ]; then
    echo "ERROR: $RAW_TRACES not found. Run scripts/generate_traces.py first." >&2
    exit 1
fi

# --- preflight: dataset leakage gate (issue #60) ---
#
# The eval protocol bootstraps 5 independent 200-sample draws from the 1K Hard
# set (seeds 0..4 — see scripts/check_dataset_overlap.py and RESULTS.md). If
# raw_traces.jsonl was generated with only --exclude-ids hard_200_sample_ids.json
# (the legacy default in tpu/launch_stage1.sh and slurm/generate_traces.sh),
# ~80% of every per-seed eval draw is contaminated by the training pool and
# the headline scores measure memorization, not generalization.
#
# check_dataset_overlap.py exits non-zero on any prompt_id leakage between the
# training pool and any of the 5 per-seed draws, so set -e here aborts the
# whole multi-seed run before we burn cluster time on a contaminated experiment.
#
# Skip with SKIP_OVERLAP_CHECK=1 only for audit/replay runs where you have
# already accounted for the contamination in writing.
SKIP_OVERLAP_CHECK="${SKIP_OVERLAP_CHECK:-0}"
if [ "$SKIP_OVERLAP_CHECK" = "1" ]; then
    echo "WARNING: SKIP_OVERLAP_CHECK=1 — skipping dataset leakage gate (issue #60)"
else
    echo "--- preflight: dataset leakage gate (issue #60) ---"
    python scripts/check_dataset_overlap.py \
        --train-jsonl "$RAW_TRACES" \
        --seeds 5
    echo
fi

echo "Multi-seed run:"
echo "  seeds:     $SEEDS"
echo "  config:    $CONFIG"
echo "  grader:    $GRADER"
echo "  min_score: $MIN_SCORE"
echo "  val_ratio: $VAL_RATIO"
echo

for SEED in $SEEDS; do
    SFT_DIR="data/sft/seed_${SEED}"
    CKPT_DIR="checkpoints/seed_${SEED}"
    EVAL_DIR="eval/seed_${SEED}"
    mkdir -p "$SFT_DIR" "$CKPT_DIR" "$EVAL_DIR"

    echo "=========================================================="
    echo "SEED $SEED"
    echo "=========================================================="

    echo "--- filter (seed $SEED) ---"
    python scripts/filter_traces.py \
        --input "$RAW_TRACES" \
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --grader-model "$GRADER" \
        --output-dir "$SFT_DIR" \
        --min-score "$MIN_SCORE" \
        --val-ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "--- train (seed $SEED) ---"
    python scripts/train_lora.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --train-file "$SFT_DIR/train.jsonl" \
        --val-file "$SFT_DIR/val.jsonl" \
        --output-dir "$CKPT_DIR"

    echo "--- eval 4 configs (seed $SEED) ---"
    python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$IDS" \
        --output "$EVAL_DIR/base_no_wrapper.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$IDS" \
        --output "$EVAL_DIR/base_bodhi.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --sample-ids "$IDS" \
        --output "$EVAL_DIR/lora_no_wrapper.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --use-bodhi --sample-ids "$IDS" \
        --output "$EVAL_DIR/lora_bodhi.json" --seed "$SEED"
done

echo
echo "=========================================================="
echo "aggregate across seeds"
echo "=========================================================="
python scripts/aggregate_seeds.py \
    --seed-dirs eval/seed_* \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/multi_seed_summary.json

echo
echo "Multi-seed run done."
