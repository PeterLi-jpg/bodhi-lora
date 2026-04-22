#!/bin/bash
# launch_multiseed.sh — run multiple seeds sequentially on one on-demand v4-32 VM.
#
# Quota: 32 on-demand Cloud TPU v4 chips in zone us-central2-b (never preempted).
#
# The VM stays alive for all seeds. The HF model download and XLA graph
# compilation only happen once, so per-seed overhead after the first is small.
# Estimated total time: ~1.5 - 2 hours for 3 seeds.
#
# Usage:
#   bash tpu/launch_multiseed.sh                   # default: seeds 42 123 456
#   SEEDS="42 123" bash tpu/launch_multiseed.sh    # custom seeds
#   (HF_TOKEN auto-loaded from .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

SEEDS="${SEEDS:-42 123 456}"
GCS_DATA_PATH="${GCS_DATA_PATH:-}"
PROJECT="tokyo-micron-494016-s9"
TPU_NAME="bohdi-lora-v4"
TPU_TYPE="v4-32"
TPU_RUNTIME="tpu-vm-base"
ZONE="us-central2-b"

echo "Seeds to run: $SEEDS"
echo "Estimated time: ~1.5 - 2 hours for 3 seeds"
echo ""

# Create the VM once
echo "=== Creating TPU VM $TPU_NAME ($TPU_TYPE, on-demand) ==="
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$TPU_TYPE" \
    --version="$TPU_RUNTIME" \
    --project="$PROJECT"

# On exit (normal, error, or Ctrl-C): rescue any unsaved checkpoints first,
# then delete the VM. Completed seeds are already local; this catches
# whatever is on the VM for any in-progress seed that was interrupted.
trap '
    echo ""
    echo "=== Saving any remaining checkpoints before VM deletion ==="
    mkdir -p "./results/_rescue"
    gcloud compute tpus tpu-vm scp \
        --recurse \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/checkpoints" "./results/_rescue/" 2>/dev/null \
        && echo "Rescue copy done — check ./results/_rescue/" \
        || echo "Rescue copy failed or nothing to copy."
    echo "=== Deleting TPU VM ==="
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null
    echo "VM deleted."
' EXIT

# ── One-time setup ────────────────────────────────────────────────────────────
echo "=== One-time setup (deps + repo + data) ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail

git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
cd ~/bohdi-lora
bash tpu/setup_tpu.sh
mkdir -p data/sft eval logs

if [ -n '${GCS_DATA_PATH}' ]; then
    echo '=== Downloading training data ==='
    gsutil -m cp '${GCS_DATA_PATH}/train.jsonl' data/sft/train.jsonl
    gsutil -m cp '${GCS_DATA_PATH}/val.jsonl'   data/sft/val.jsonl
    echo \"Data: \$(wc -l < data/sft/train.jsonl) train, \$(wc -l < data/sft/val.jsonl) val\"
else
    echo 'WARNING: GCS_DATA_PATH not set — data/sft/train.jsonl must already be on this VM'
fi

# Pre-download the model so all seeds share the cache
echo '=== Pre-downloading MedGemma-27B (runs once, ~20 min) ==='
HF_TOKEN='${HF_TOKEN}' python3 -c \"
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/medgemma-27b-text-it', token='${HF_TOKEN}')
from transformers import AutoModelForCausalLM
import torch
AutoModelForCausalLM.from_pretrained(
    'google/medgemma-27b-text-it', token='${HF_TOKEN}',
    torch_dtype=torch.bfloat16)
print('Model cached.')
\"
"

# ── Train each seed sequentially ──────────────────────────────────────────────
for SEED in $SEEDS; do
    echo ""
    echo "=== Training seed $SEED ==="

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
set -euo pipefail
cd ~/bohdi-lora
mkdir -p checkpoints/seed_${SEED}

export HF_TOKEN='${HF_TOKEN}'

accelerate launch \
    --config_file tpu/accelerate_config_v4_32.yaml \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --seed ${SEED} \
    --output-dir checkpoints/seed_${SEED}

echo 'Seed ${SEED} done.'
"

    # Pull this seed's checkpoints back immediately
    echo "Copying seed $SEED checkpoints..."
    mkdir -p "./results/seed_${SEED}"
    gcloud compute tpus tpu-vm scp \
        --recurse \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/checkpoints/seed_${SEED}" \
        "./results/seed_${SEED}/"

    echo "Seed $SEED complete — checkpoints in ./results/seed_${SEED}/"
done

echo ""
echo "All seeds done: $SEEDS"
echo "Results in ./results/"
