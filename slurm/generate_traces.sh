#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/generate_traces_%j.out
#SBATCH --error=logs/generate_traces_%j.err
#SBATCH --job-name=bohdi_gen
#SBATCH --mem=200G

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

echo "$(date) | starting trace generation on $(hostname)"
nvidia-smi --list-gpus

# Fail fast on missing deps / gated-access / no-GPU before burning queue time.
python scripts/preflight.py

python scripts/download_data.py

python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --exclude-ids data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl \
    --use-bodhi \
    --resume-from data/sft/raw_traces.jsonl

echo "$(date) | done"
