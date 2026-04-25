#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --job-name=bohdi_eval
#SBATCH --mem=250G

set -euo pipefail

module load miniforge/24.3.0-0
conda activate bohdi  # change to your env name
cd "${BOHDI_DIR:-${SLURM_SUBMIT_DIR:?ERROR: neither BOHDI_DIR nor SLURM_SUBMIT_DIR is set (needed to find the repo root)}}"

# pick up HF_TOKEN from a local .env if the login shell didn't export it
# shellcheck source=/dev/null
[ -f .env ] && source .env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. MedGemma requires gated access."
    echo "Run: export HF_TOKEN=hf_..."
    exit 1
fi
export HF_TOKEN
RUN_DIR="${1:-${RUN_DIR:-results/manual_run}}"
SECOND_GRADER_MODEL="${SECOND_GRADER_MODEL:-}"

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/figures"

echo "$(date) | starting eval on $(hostname)"
nvidia-smi --list-gpus

# Fail fast on missing deps / gated-access / no-GPU before the 4-config sweep.
python scripts/preflight.py

python scripts/download_data.py

MODEL="google/medgemma-27b-text-it"
IDS="data/raw/hard_200_sample_ids.json"
LORA="$RUN_DIR/checkpoints/best"

echo "--- base, no wrapper ---"
python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$IDS" --output "$RUN_DIR/eval/base_no_wrapper.json"

echo "--- base + bodhi ---"
python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$IDS" --output "$RUN_DIR/eval/base_bodhi.json"

echo "--- lora, no wrapper ---"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --sample-ids "$IDS" --output "$RUN_DIR/eval/lora_no_wrapper.json"

echo "--- lora + bodhi ---"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --use-bodhi --sample-ids "$IDS" --output "$RUN_DIR/eval/lora_bodhi.json"

if [ -n "$SECOND_GRADER_MODEL" ]; then
    SECOND_GRADER_TAG="${SECOND_GRADER_TAG:-$(printf '%s' "$SECOND_GRADER_MODEL" | tr '/:' '__')}"
    SECOND_GRADER_DIR="$RUN_DIR/eval/cross_grader/$SECOND_GRADER_TAG"
    mkdir -p "$SECOND_GRADER_DIR"

    echo "--- second grader pass: $SECOND_GRADER_MODEL ---"
    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --sample-ids "$IDS" \
        --grader-model "$SECOND_GRADER_MODEL" \
        --output "$SECOND_GRADER_DIR/base_no_wrapper.json"

    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --use-bodhi \
        --sample-ids "$IDS" \
        --grader-model "$SECOND_GRADER_MODEL" \
        --output "$SECOND_GRADER_DIR/base_bodhi.json"

    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --lora-path "$LORA" \
        --sample-ids "$IDS" \
        --grader-model "$SECOND_GRADER_MODEL" \
        --output "$SECOND_GRADER_DIR/lora_no_wrapper.json"

    python scripts/eval_healthbench.py \
        --model "$MODEL" \
        --lora-path "$LORA" \
        --use-bodhi \
        --sample-ids "$IDS" \
        --grader-model "$SECOND_GRADER_MODEL" \
        --output "$SECOND_GRADER_DIR/lora_bodhi.json"

    echo "--- grader correlation report ---"
    python scripts/grader_correlation.py \
        --reference-jsons \
            "$RUN_DIR/eval/base_no_wrapper.json" \
            "$RUN_DIR/eval/base_bodhi.json" \
            "$RUN_DIR/eval/lora_no_wrapper.json" \
            "$RUN_DIR/eval/lora_bodhi.json" \
        --candidate-jsons \
            "$SECOND_GRADER_DIR/base_no_wrapper.json" \
            "$SECOND_GRADER_DIR/base_bodhi.json" \
            "$SECOND_GRADER_DIR/lora_no_wrapper.json" \
            "$SECOND_GRADER_DIR/lora_bodhi.json" \
        --output "$SECOND_GRADER_DIR/correlation.json"
fi

echo "--- U-shape stratified analysis ---"
python scripts/eval_ushape.py \
    --eval-jsons "$RUN_DIR/eval/base_no_wrapper.json" "$RUN_DIR/eval/base_bodhi.json" "$RUN_DIR/eval/lora_no_wrapper.json" "$RUN_DIR/eval/lora_bodhi.json" \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output "$RUN_DIR/eval/ushape.json"

echo "--- plot U-shape figures ---"
python scripts/plot_ushape.py \
    --input "$RUN_DIR/eval/ushape.json" \
    --eval-jsons "$RUN_DIR/eval/base_no_wrapper.json" "$RUN_DIR/eval/base_bodhi.json" "$RUN_DIR/eval/lora_no_wrapper.json" "$RUN_DIR/eval/lora_bodhi.json" \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --n-bins 10 \
    --out-dir "$RUN_DIR/figures"

echo "--- plot training loss (if trainer_state.json exists) ---"
if [ -f "$LORA/trainer_state.json" ]; then
    python scripts/plot_training.py \
        --trainer-state "$LORA/trainer_state.json" \
        --output "$RUN_DIR/figures/training_loss.png"
else
    echo "trainer_state.json not found at $LORA — skipping training loss plot"
fi

echo "$(date) | done"
