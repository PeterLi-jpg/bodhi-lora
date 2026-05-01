#!/bin/bash
# Paper figure-3 ablation infrastructure (issue #69).
#
# BODHI bundles several distinct components (domain framing, calibration
# prompt, multi-turn questions, safe-recommendations / abstention). The paper
# claims the LoRA internalizes "epistemic virtues", but a reviewer will ask
# which component actually drove the shift. This script runs the full pipeline
# (generate -> filter -> train -> eval) once per ablation tag for ONE seed,
# producing per-component eval numbers we can plot side-by-side with the
# baseline.
#
# Each ablation gets its OWN trace pool: the wrapper modification changes
# what the teacher writes, so we can't share traces across ablations the way
# run_multi_seed.sh shares one pool across seeds.
#
# Cost warning: per #69, one full ablation pipeline takes ~24h on H100; five
# of them serialized = ~120h. This script is designed to run on a TPU/GPU pod,
# NOT on a CPU dev box.
#
# Usage:
#   SEED=42 N_EXAMPLES=300 bash scripts/run_ablation_seeds.sh
#
# Override defaults via env vars:
#   SEED=42                               # which seed to use for filter+train
#   N_EXAMPLES=300                        # how many traces per ablation
#   TRAIN_CONFIG=configs/lora_medgemma27b_tpu.yaml
#   MODEL=google/medgemma-27b-text-it
#   IDS=data/raw/hard_200_sample_ids.json
#   GRADER=Qwen/Qwen2.5-14B-Instruct       # filter-side grader (eval is Llama-3.1-8B)
#   MIN_SCORE=0.4
#   VAL_RATIO=0.1
#
# Note on eval: we eval each LoRA with the *production* BODHI wrapper
# (--use-bodhi, no --ablate-component flag), because eval_healthbench.py
# doesn't yet support component ablation and the figure-3 question is "what
# does the LoRA learn from training-time wrapper component X"; eval-time
# ablation is a separate experiment.

set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
N_EXAMPLES="${N_EXAMPLES:-300}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lora_medgemma27b_tpu.yaml}"
MODEL="${MODEL:-google/medgemma-27b-text-it}"
IDS="${IDS:-data/raw/hard_200_sample_ids.json}"
GRADER="${GRADER:-Qwen/Qwen2.5-14B-Instruct}"
MIN_SCORE="${MIN_SCORE:-0.4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

COMPONENTS=(none no_calibration no_questions no_abstention no_domain_framing)

echo "BODHI ablation sweep (issue #69)"
echo "  seed:        $SEED"
echo "  n_examples:  $N_EXAMPLES per ablation"
echo "  components:  ${COMPONENTS[*]}"
echo "  config:      $TRAIN_CONFIG"
echo "  model:       $MODEL"
echo

for COMP in "${COMPONENTS[@]}"; do
    SFT_DIR="data/sft/ablation/${COMP}"
    CKPT_DIR="checkpoints/ablation/${COMP}"
    EVAL_DIR="eval/ablation/${COMP}"
    RAW_TRACES="${SFT_DIR}/raw_traces.jsonl"
    mkdir -p "$SFT_DIR" "$CKPT_DIR" "$EVAL_DIR"

    echo "=========================================================="
    echo "ablation: $COMP"
    echo "=========================================================="

    echo "--- generate traces ($COMP) ---"
    python scripts/generate_traces.py \
        --model "$MODEL" \
        --use-bodhi \
        --ablate-component "$COMP" \
        --output "$RAW_TRACES" \
        --max-examples "$N_EXAMPLES" \
        --seed "$SEED"

    echo "--- filter ($COMP) ---"
    # Defensive --exclude-ids drops any HealthBench Hard rows that may have
    # survived in a legacy raw_traces.jsonl (issue #60).
    python scripts/filter_traces.py \
        --input "$RAW_TRACES" \
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
        --grader-model "$GRADER" \
        --output-dir "$SFT_DIR" \
        --min-score "$MIN_SCORE" \
        --val-ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "--- train ($COMP) ---"
    python scripts/train_lora.py \
        --config "$TRAIN_CONFIG" \
        --seed "$SEED" \
        --train-file "$SFT_DIR/train.jsonl" \
        --val-file "$SFT_DIR/val.jsonl" \
        --output-dir "$CKPT_DIR"

    echo "--- eval ($COMP) ---"
    # Eval uses the production wrapper (no --ablate-component), since
    # eval_healthbench.py doesn't support that flag yet (see top-of-file
    # note). The interesting question for figure 3 is what the trained LoRA
    # learned from each ablated training set, evaluated under the standard
    # eval-time wrapper.
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --lora-path "$CKPT_DIR/best" \
        --use-bodhi \
        --sample-ids "$IDS" \
        --output "$EVAL_DIR/lora.json" \
        --seed "$SEED"
done

echo
echo "=========================================================="
echo "ablation sweep done. per-component results in eval/ablation/*/lora.json"
echo "=========================================================="
