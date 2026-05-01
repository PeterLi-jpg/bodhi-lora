#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/generalization_generate_%j.out
#SBATCH --error=logs/generalization_generate_%j.err
#SBATCH --job-name=bohdi_gen_generalization
#SBATCH --mem=200G

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

python scripts/preflight.py
python scripts/download_data.py
python scripts/check_dataset_overlap.py

python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench \
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces_healthbench_only.jsonl \
    --use-bodhi \
    --resume-from data/sft/raw_traces_healthbench_only.jsonl
