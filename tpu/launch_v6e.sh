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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

PROJECT="tokyo-micron-494016-s9"
TPU_NAME="bohdi-lora-v6e"
TPU_TYPE="v6e-64"
# tpu-vm-base is a plain Ubuntu image. setup_tpu.sh installs torch + torch_xla
# from Google's TPU wheel server (2.5.0), which supports v6e (Trillium).
# Note: tpu-vm-pt-* images only go up to 2.0 and do not support v6e.
TPU_RUNTIME="tpu-vm-base"

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
trap '
    echo "=== Saving checkpoints before VM deletion ==="
    mkdir -p "./checkpoints_tpu"
    gcloud compute tpus tpu-vm scp \
        --recurse --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/checkpoints" "./checkpoints_tpu/" 2>/dev/null \
        && echo "Checkpoints saved to ./checkpoints_tpu/" \
        || echo "Nothing to copy or copy failed."
    echo "=== Deleting TPU VM ==="
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null
    echo "VM deleted."
' EXIT

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

# Training data (data/sft/train.jsonl + val.jsonl) is not in the repo.
# Download it from a GCS bucket if GCS_DATA_PATH is set, otherwise
# the user must scp the files to the VM before training.
if [ -n "${GCS_DATA_PATH:-}" ]; then
    echo "=== Downloading training data from ${GCS_DATA_PATH} ==="
    gsutil -m cp "${GCS_DATA_PATH}/train.jsonl" data/sft/train.jsonl
    gsutil -m cp "${GCS_DATA_PATH}/val.jsonl"   data/sft/val.jsonl
    echo "Data downloaded: $(wc -l < data/sft/train.jsonl) train, $(wc -l < data/sft/val.jsonl) val examples"
else
    echo "WARNING: GCS_DATA_PATH not set — data/sft/train.jsonl and val.jsonl must"
    echo "already be on this VM (e.g. scp'd in). Training will fail if they are absent."
fi

echo "=== Running training (v6e-64, 64 TPU chips) ==="
export HF_TOKEN="${HF_TOKEN}"

accelerate launch \
    --config_file tpu/accelerate_config_v6e64.yaml \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --output-dir checkpoints

echo "=== Training done ==="
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
