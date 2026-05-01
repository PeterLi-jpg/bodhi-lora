#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/generalization_train_eval_%j.out
#SBATCH --error=logs/generalization_train_eval_%j.err
#SBATCH --job-name=bohdi_generalization
#SBATCH --mem=250G

set -euo pipefail

module load miniforge/24.3.0-0
conda activate bohdi
cd "${BOHDI_DIR:-${SLURM_SUBMIT_DIR:?ERROR: neither BOHDI_DIR nor SLURM_SUBMIT_DIR is set}}"

# shellcheck source=/dev/null
[ -f .env ] && source .env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. MedGemma requires gated access."
    exit 1
fi
export HF_TOKEN

RUN_DIR="${1:-${RUN_DIR:-results/manual_generalization}}"
mkdir -p "$RUN_DIR/eval/generalization" "$RUN_DIR/figures/generalization"

python scripts/preflight.py
python scripts/download_data.py

python scripts/filter_traces.py \
    --input data/sft/raw_traces_healthbench_only.jsonl \
    --healthbench-data data/raw/healthbench.jsonl \
    --grader-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --output-dir data/sft/generalization \
    --min-score 0.4 \
    --val-ratio 0.1 \
    --graded-output data/sft/generalization/all_graded.jsonl

python scripts/train_lora.py \
    --config configs/lora_medgemma27b.yaml \
    --train-file data/sft/generalization/train.jsonl \
    --val-file data/sft/generalization/val.jsonl \
    --output-dir "$RUN_DIR/checkpoints/generalization"

MODEL="google/medgemma-27b-text-it"
IDS="data/raw/hard_200_sample_ids.json"
LORA="$RUN_DIR/checkpoints/generalization/best"

python scripts/eval_healthbench.py --model "$MODEL" --sample-ids "$IDS" --output "$RUN_DIR/eval/generalization/base_no_wrapper.json"
python scripts/eval_healthbench.py --model "$MODEL" --use-bodhi --sample-ids "$IDS" --output "$RUN_DIR/eval/generalization/base_bodhi.json"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --sample-ids "$IDS" --output "$RUN_DIR/eval/generalization/lora_no_wrapper.json"
python scripts/eval_healthbench.py --model "$MODEL" --lora-path "$LORA" --use-bodhi --sample-ids "$IDS" --output "$RUN_DIR/eval/generalization/lora_bodhi.json"
