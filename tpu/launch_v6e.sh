#!/bin/bash
# launch_v6e.sh — create a v6e-64 spot TPU VM, run training, delete VM on exit.
#
# Quota source: TRC grant, project tokyo-micron-494016-s9
# Primary zone:  us-east1-d  (v6e-64 spot)
# Fallback zone: europe-west4-a (v6e-64 spot)
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash tpu/launch_v6e.sh
#
# The script streams logs in real time and deletes the VM on exit (success or
# failure). If the spot VM is preempted mid-run, re-launch or use launch_v4.sh
# for the on-demand v4-32 fallback.

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"

PROJECT="tokyo-micron-494016-s9"
TPU_NAME="bohdi-lora-v6e"
TPU_TYPE="v6e-64"
# tpu-vm-pt-2.4 is the latest stable PyTorch runtime with torch_xla for v6e
TPU_RUNTIME="v2-alpha-tpuv6e"

# Primary zone, fallback to europe if quota is exhausted
PRIMARY_ZONE="us-east1-d"
FALLBACK_ZONE="europe-west4-a"

# Try primary zone first; if create fails, use fallback
ZONE="$PRIMARY_ZONE"
if ! gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$TPU_TYPE" \
    --version="$TPU_RUNTIME" \
    --project="$PROJECT" \
    --spot 2>/dev/null; then
  echo "Primary zone $PRIMARY_ZONE unavailable, trying $FALLBACK_ZONE..."
  ZONE="$FALLBACK_ZONE"
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --zone="$ZONE" \
      --accelerator-type="$TPU_TYPE" \
      --version="$TPU_RUNTIME" \
      --project="$PROJECT" \
      --spot
fi

echo "TPU VM $TPU_NAME created in $ZONE"

# Always delete the VM on exit, even on error or Ctrl-C
trap 'echo "Cleaning up TPU VM..."; \
      gcloud compute tpus tpu-vm delete "$TPU_NAME" \
          --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null; \
      echo "VM deleted."' EXIT

# Copy SSH key and run setup + training remotely
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="$(cat <<REMOTE
set -euo pipefail

echo "=== Cloning repo ==="
git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
cd ~/bohdi-lora

echo "=== Setting up deps ==="
bash tpu/setup_tpu.sh

echo "=== Making data dirs ==="
mkdir -p data/sft eval checkpoints logs

echo "=== Running training (v6e-64, 64 TPU chips) ==="
export HF_TOKEN="${HF_TOKEN}"

accelerate launch \
    --config_file tpu/accelerate_config_v6e64.yaml \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --output-dir checkpoints

echo "=== Training done — copying outputs ==="
REMOTE
)"

# Retrieve outputs via gcloud scp after training
echo "Copying checkpoints from TPU VM..."
gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone="$ZONE" \
    --project="$PROJECT" \
    "${TPU_NAME}:~/bohdi-lora/checkpoints" ./checkpoints_tpu

echo "Done. Checkpoints in ./checkpoints_tpu/"
