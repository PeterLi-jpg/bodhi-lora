#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --job-name=bohdi_train
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
export WANDB_MODE=offline
RUN_DIR="${1:-${RUN_DIR:-results/manual_run}}"

mkdir -p "$RUN_DIR"

echo "$(date) | starting lora training on $(hostname)"
nvidia-smi --list-gpus

# Fail fast on missing deps / gated-access / no-GPU before loading MedGemma-27B.
python scripts/preflight.py

python scripts/train_lora.py \
    --config configs/lora_medgemma27b.yaml \
    --output-dir "$RUN_DIR/checkpoints"

echo "$(date) | done"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
