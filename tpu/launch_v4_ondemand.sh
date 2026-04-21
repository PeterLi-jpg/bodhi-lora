#!/bin/bash
# launch_v4_ondemand.sh — on-demand v4-32 TPU VM fallback.
#
# Use this if the v6e spot VMs are preempted or unavailable.
# On-demand = never preempted, but slower than v6e.
#
# Quota: 32 on-demand Cloud TPU v4 chips in zone us-central2-b
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash tpu/launch_v4_ondemand.sh

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"

PROJECT="tokyo-micron-494016-s9"
TPU_NAME="bohdi-lora-v4"
TPU_TYPE="v4-32"
TPU_RUNTIME="tpu-vm-pt-2.4"   # PyTorch 2.4 + torch_xla, stable for v4
ZONE="us-central2-b"

gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$TPU_TYPE" \
    --version="$TPU_RUNTIME" \
    --project="$PROJECT"
# NOTE: no --spot flag — this is on-demand

echo "TPU VM $TPU_NAME created (on-demand, zone $ZONE)"

trap 'echo "Cleaning up TPU VM..."; \
      gcloud compute tpus tpu-vm delete "$TPU_NAME" \
          --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null; \
      echo "VM deleted."' EXIT

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="$(cat <<REMOTE
set -euo pipefail

git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
cd ~/bohdi-lora

bash tpu/setup_tpu.sh

mkdir -p data/sft eval checkpoints logs
export HF_TOKEN="${HF_TOKEN}"

# v4-32 uses 32 cores; batch size is smaller (less HBM per chip than v6e)
accelerate launch \
    --config_file tpu/accelerate_config_v4_32.yaml \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --output-dir checkpoints

REMOTE
)"

echo "Copying checkpoints from TPU VM..."
gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone="$ZONE" \
    --project="$PROJECT" \
    "${TPU_NAME}:~/bohdi-lora/checkpoints" ./checkpoints_tpu

echo "Done."
