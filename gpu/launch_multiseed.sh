#!/bin/bash
# launch_multiseed.sh (GPU) — run multiple seeds sequentially on one GPU machine.
#
# Works on any Linux machine with CUDA GPUs: cloud VMs (Lambda, RunPod, Vast.ai)
# or a local workstation. For SLURM clusters, use slurm/train_lora.sh instead.
#
# GPU memory requirements:
#   Full-precision LoRA (configs/lora_medgemma27b.yaml):
#     needs ~60-80 GB VRAM — use 2x A100-40GB or 1x A100-80GB/H100-80GB
#   4-bit QLoRA (configs/lora_medgemma27b_qlora.yaml):
#     needs ~18-22 GB VRAM — fits on a single A100-40GB or RTX 4090
#
# Usage (run from repo root):
#   bash gpu/launch_multiseed.sh                   # seeds 42 123 456, full-precision
#   SEEDS="42 123" bash gpu/launch_multiseed.sh
#   CONFIG=configs/lora_medgemma27b_qlora.yaml bash gpu/launch_multiseed.sh
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

echo "Config:  $CONFIG"
echo "Seeds:   $SEEDS"
echo ""

cd "$REPO_ROOT"

export HF_TOKEN

for SEED in $SEEDS; do
    echo "========================================"
    echo "  Training seed $SEED"
    echo "========================================"

    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train_lora.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --output-dir "${SEED_DIR}/checkpoints"

    echo "Seed $SEED done — checkpoints in ${SEED_DIR}/checkpoints/"
    echo ""
done

echo "All seeds done: $SEEDS"
echo "Results in $RESULTS_DIR/"
