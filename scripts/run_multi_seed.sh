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
# Optional: cross-grader pass for the paper's bias-control story.
#   SECOND_GRADER_MODEL=Qwen/Qwen2.5-14B-Instruct  # off by default; ~12h H100 if set
# When set, each seed re-grades the four eval configs with this second
# grader and writes per-seed Spearman correlation to
# eval/seed_<N>/cross_grader/<tag>/correlation.json.
#
# Expects data/sft/raw_traces.jsonl to already exist (run
# scripts/generate_traces.py first, or slurm/generate_traces.sh).

set -euo pipefail
cd "${BOHDI_DIR:-$(dirname "$0")/..}"

SEEDS="${SEEDS:-42 7 13 99 101}"
CONFIG="${CONFIG:-configs/lora_medgemma27b.yaml}"
MODEL="${MODEL:-google/medgemma-27b-text-it}"
# Bootstrap eval: each seed gets its own 200-prompt random draw from the 1000
# HealthBench Hard prompts (issue #60). Per-seed files live at
# data/raw/hard_seed_<N>.json and are generated below if missing.
GRADER="${GRADER:-meta-llama/Llama-3.1-8B-Instruct}"
MIN_SCORE="${MIN_SCORE:-0.4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# Optional second-pass grader for the cross-grader bias-control sweep.
# When unset (the default) we skip the extra ~12h H100 of grader compute.
# Recommended secondary: Qwen/Qwen2.5-14B-Instruct (different family from
# the primary Llama grader, breaking the "graded by your own evaluator"
# critique).
SECOND_GRADER_MODEL="${SECOND_GRADER_MODEL:-}"

RAW_TRACES="data/sft/raw_traces.jsonl"
if [ ! -f "$RAW_TRACES" ]; then
    echo "ERROR: $RAW_TRACES not found. Run scripts/generate_traces.py first." >&2
    exit 1
fi

echo "Multi-seed run:"
echo "  seeds:        $SEEDS"
echo "  config:       $CONFIG"
echo "  grader:       $GRADER"
echo "  min_score:    $MIN_SCORE"
echo "  val_ratio:    $VAL_RATIO"
echo "  2nd grader:   ${SECOND_GRADER_MODEL:-(off — set SECOND_GRADER_MODEL to enable cross-grader pass)}"
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
    # Defensive --exclude-ids drops any HealthBench Hard rows that may
    # have survived in a legacy raw_traces.jsonl (issue #60).
    python scripts/filter_traces.py \
        --input "$RAW_TRACES" \
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
        --grader-model "$GRADER" \
        --output-dir "$SFT_DIR" \
        --min-score "$MIN_SCORE" \
        --val-ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "--- preflight: leakage gate (seed $SEED) ---"
    # Aborts non-zero if any HealthBench Hard prompt landed in train.jsonl.
    # Issue #60 invariant: all 1K Hard must be excluded so bootstrap eval is honest.
    # Skip with SKIP_OVERLAP_CHECK=1 only for audit/replay runs that have
    # already documented the contamination in writing (per Max's #124 escape hatch).
    if [ "${SKIP_OVERLAP_CHECK:-0}" = "1" ]; then
        echo "WARNING: SKIP_OVERLAP_CHECK=1 — skipping leakage gate (issue #60)"
    else
        python scripts/check_dataset_overlap.py \
            --train-jsonl "$SFT_DIR/train.jsonl" \
            --tag-overlap
    fi

    echo "--- train (seed $SEED) ---"
    python scripts/train_lora.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --train-file "$SFT_DIR/train.jsonl" \
        --val-file "$SFT_DIR/val.jsonl" \
        --output-dir "$CKPT_DIR"

    echo "--- eval 4 configs (seed $SEED) ---"
    # Per-seed bootstrap eval draw (issue #60): a deterministic 200-prompt
    # subset of the 1000 HealthBench Hard prompts, unique to this seed.
    SEED_IDS="data/raw/hard_seed_${SEED}.json"
    python scripts/make_bootstrap_eval_ids.py \
        --healthbench-jsonl data/raw/healthbench_hard.jsonl \
        --seed "$SEED" \
        --output "$SEED_IDS"
    python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$SEED_IDS" \
        --output "$EVAL_DIR/base_no_wrapper.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$SEED_IDS" \
        --output "$EVAL_DIR/base_bodhi.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --sample-ids "$SEED_IDS" \
        --output "$EVAL_DIR/lora_no_wrapper.json" --seed "$SEED"
    python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --use-bodhi --sample-ids "$SEED_IDS" \
        --output "$EVAL_DIR/lora_bodhi.json" --seed "$SEED"

    # Optional cross-grader second pass — same 4 configs, same prompt
    # IDs, but graded by SECOND_GRADER_MODEL. Lets us report Spearman
    # correlation between the primary (Qwen) grader and a different
    # family (e.g. Llama-3.1-70B-Instruct) so the paper can't be
    # dismissed with "you optimized for your own evaluator". Only runs
    # when SECOND_GRADER_MODEL is set (significant grader compute).
    if [ -n "$SECOND_GRADER_MODEL" ]; then
        SECOND_GRADER_TAG="${SECOND_GRADER_TAG:-$(printf '%s' "$SECOND_GRADER_MODEL" | tr '/:' '__')}"
        SECOND_GRADER_DIR="$EVAL_DIR/cross_grader/$SECOND_GRADER_TAG"
        mkdir -p "$SECOND_GRADER_DIR"

        echo "--- cross-grader pass: $SECOND_GRADER_MODEL (seed $SEED) ---"
        python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$IDS" \
            --grader-model "$SECOND_GRADER_MODEL" \
            --output "$SECOND_GRADER_DIR/base_no_wrapper.json" --seed "$SEED"
        python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$IDS" \
            --grader-model "$SECOND_GRADER_MODEL" \
            --output "$SECOND_GRADER_DIR/base_bodhi.json" --seed "$SEED"
        python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --sample-ids "$IDS" \
            --grader-model "$SECOND_GRADER_MODEL" \
            --output "$SECOND_GRADER_DIR/lora_no_wrapper.json" --seed "$SEED"
        python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$CKPT_DIR/best" --use-bodhi --sample-ids "$IDS" \
            --grader-model "$SECOND_GRADER_MODEL" \
            --output "$SECOND_GRADER_DIR/lora_bodhi.json" --seed "$SEED"

        echo "--- grader correlation (seed $SEED) ---"
        python scripts/grader_correlation.py \
            --reference-jsons \
                "$EVAL_DIR/base_no_wrapper.json" \
                "$EVAL_DIR/base_bodhi.json" \
                "$EVAL_DIR/lora_no_wrapper.json" \
                "$EVAL_DIR/lora_bodhi.json" \
            --candidate-jsons \
                "$SECOND_GRADER_DIR/base_no_wrapper.json" \
                "$SECOND_GRADER_DIR/base_bodhi.json" \
                "$SECOND_GRADER_DIR/lora_no_wrapper.json" \
                "$SECOND_GRADER_DIR/lora_bodhi.json" \
            --output "$SECOND_GRADER_DIR/correlation.json"
    fi
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
