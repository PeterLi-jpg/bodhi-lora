#!/bin/bash
# launch_multiseed.sh (GPU) — run the full pipeline on one GPU machine.
#
# Stages:
#   1 — generate BOHDI traces (MedGemma-27B inference)
#   2 — grade + filter traces (Qwen2.5-14B-Instruct, bfloat16)
#   3 — LoRA SFT, one run per seed (multi-GPU DDP or single-GPU)
#   4 — evaluate all 4 configs per seed + U-shape + rubric diff
#   5 — aggregate across seeds (mean/std/CI tables)
#
# GPU memory requirements:
#   Full-precision LoRA (configs/lora_medgemma27b.yaml):
#     needs ~60-80 GB VRAM — use 2x A100-40GB or 1x A100-80GB/H100-80GB
#   4-bit QLoRA (configs/lora_medgemma27b_qlora.yaml):
#     needs ~18-22 GB VRAM — fits on a single A100-40GB or RTX 4090
#
# Usage (run from repo root):
#   bash gpu/launch_multiseed.sh                          # seeds 42 123 456
#   SEEDS="42 123" bash gpu/launch_multiseed.sh
#   CONFIG=configs/lora_medgemma27b_qlora.yaml bash gpu/launch_multiseed.sh
#   MAX_EXAMPLES=100 bash gpu/launch_multiseed.sh         # quick test run
#   (HF_TOKEN auto-loaded from .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
ENV_FILE="${REPO_ROOT}/.env"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

SEEDS="${SEEDS:-42 123 456}"
CONFIG="${CONFIG:-configs/lora_medgemma27b.yaml}"
RESULTS_DIR="${REPO_ROOT}/results"
MODEL="google/medgemma-27b-text-it"
SAMPLE_IDS="data/raw/hard_200_sample_ids.json"

# Detect GPU count and pick the right accelerate config
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: no CUDA GPUs found. Check nvidia-smi."
    exit 1
elif [ "$GPU_COUNT" -eq 1 ]; then
    ACCEL_CFG="${SCRIPT_DIR}/accelerate_config_1gpu.yaml"
    echo "Single GPU detected — using 1-GPU config"
else
    ACCEL_CFG="${SCRIPT_DIR}/accelerate_config_multi.yaml"
    echo "${GPU_COUNT} GPUs detected — using multi-GPU DDP config"
    echo "  (edit gpu/accelerate_config_multi.yaml if num_processes doesn't match)"
fi

echo "Model:   $MODEL"
echo "Config:  $CONFIG"
echo "Seeds:   $SEEDS"
echo "Results: $RESULTS_DIR"
echo ""

cd "$REPO_ROOT"
export HF_TOKEN
mkdir -p data/raw data/sft "${RESULTS_DIR}"

# ── Stage 1: generate traces ──────────────────────────────────────────────────
echo "========================================"
echo "  Stage 1: generate BOHDI traces"
echo "========================================"

if [ -f "data/sft/raw_traces.jsonl" ]; then
    echo "  data/sft/raw_traces.jsonl already exists — skipping generation."
    echo "  Delete it to regenerate: rm data/sft/raw_traces.jsonl"
else
    python scripts/generate_traces.py \
        --model "$MODEL" \
        --datasets healthbench_hard healthbench \
        --output data/sft/raw_traces.jsonl \
        --use-bodhi \
        --max-examples "${MAX_EXAMPLES:-4800}"
fi

# ── Stage 2: grade + filter ───────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Stage 2: grade and filter traces"
echo "========================================"

if [ -f "data/sft/train.jsonl" ] && [ -f "data/sft/val.jsonl" ]; then
    echo "  data/sft/train.jsonl + val.jsonl already exist — skipping filter."
    echo "  Delete them to re-filter: rm data/sft/train.jsonl data/sft/val.jsonl"
else
    python scripts/filter_traces.py \
        --input data/sft/raw_traces.jsonl \
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --output-dir data/sft \
        --min-score 0.4
fi

echo "Training data: $(wc -l < data/sft/train.jsonl) train, $(wc -l < data/sft/val.jsonl) val"

# ── Stages 3 + 4 per seed ─────────────────────────────────────────────────────
for SEED in $SEEDS; do
    echo ""
    echo "========================================"
    echo "  Stage 3: training seed $SEED"
    echo "========================================"

    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    CKPT_DIR="${SEED_DIR}/checkpoints"
    EVAL_DIR="${SEED_DIR}/eval"
    FIG_DIR="${SEED_DIR}/figures"
    LORA="${CKPT_DIR}/best"
    mkdir -p "$CKPT_DIR" "$EVAL_DIR" "$FIG_DIR"

    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train_lora.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --output-dir "$CKPT_DIR"

    echo "Seed $SEED training done — checkpoints in $CKPT_DIR/"

    # ── Stage 4: evaluate 4 configurations ───────────────────────────────────
    echo ""
    echo "  Stage 4: evaluation (seed $SEED)"
    echo ""

    echo "  -- base, no wrapper --"
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --sample-ids "$SAMPLE_IDS" \
        --seed "$SEED" \
        --output "${EVAL_DIR}/base_no_wrapper.json"

    echo "  -- base + BOHDI wrapper --"
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --use-bodhi \
        --sample-ids "$SAMPLE_IDS" \
        --seed "$SEED" \
        --output "${EVAL_DIR}/base_bodhi.json"

    echo "  -- LoRA, no wrapper --"
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --lora-path "$LORA" \
        --sample-ids "$SAMPLE_IDS" \
        --seed "$SEED" \
        --output "${EVAL_DIR}/lora_no_wrapper.json"

    echo "  -- LoRA + BOHDI wrapper --"
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --lora-path "$LORA" \
        --use-bodhi \
        --sample-ids "$SAMPLE_IDS" \
        --seed "$SEED" \
        --output "${EVAL_DIR}/lora_bodhi.json"

    echo "  -- U-shape stratification --"
    python scripts/eval_ushape.py \
        --eval-jsons \
            "${EVAL_DIR}/base_no_wrapper.json" \
            "${EVAL_DIR}/base_bodhi.json" \
            "${EVAL_DIR}/lora_no_wrapper.json" \
            "${EVAL_DIR}/lora_bodhi.json" \
        --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --output "${EVAL_DIR}/ushape.json"

    echo "  -- plots --"
    python scripts/plot_ushape.py \
        --input "${EVAL_DIR}/ushape.json" \
        --eval-jsons \
            "${EVAL_DIR}/base_no_wrapper.json" \
            "${EVAL_DIR}/base_bodhi.json" \
            "${EVAL_DIR}/lora_no_wrapper.json" \
            "${EVAL_DIR}/lora_bodhi.json" \
        --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
        --n-bins 10 \
        --out-dir "$FIG_DIR"

    if [ -f "${LORA}/trainer_state.json" ]; then
        python scripts/plot_training.py \
            --trainer-state "${LORA}/trainer_state.json" \
            --output "${FIG_DIR}/training_loss.png"
    fi

    echo "  -- rubric diff (base_no_wrapper → lora_bodhi) --"
    python scripts/rubric_diff.py \
        "${EVAL_DIR}/base_no_wrapper.json" \
        "${EVAL_DIR}/lora_bodhi.json" \
        --output "${EVAL_DIR}/rubric_diff.json"

    echo "Seed $SEED complete — results in $SEED_DIR/"
done

# ── Stage 5: aggregate across all seeds ──────────────────────────────────────
echo ""
echo "========================================"
echo "  Stage 5: multi-seed aggregate"
echo "========================================"

SEED_DIR_ARGS=""
for SEED in $SEEDS; do
    SEED_DIR_ARGS="$SEED_DIR_ARGS ${RESULTS_DIR}/seed_${SEED}/eval"
done

python scripts/aggregate_seeds.py \
    --seed-dirs $SEED_DIR_ARGS \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output "${RESULTS_DIR}/multi_seed_summary.json"

echo ""
echo "All seeds done: $SEEDS"
echo "Results in $RESULTS_DIR/"
echo "Multi-seed summary: ${RESULTS_DIR}/multi_seed_summary.json"
