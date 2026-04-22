#!/bin/bash
# launch_all_seeds.sh — fan out training across all 6 TRC quota VMs in parallel.
#
# Each VM runs a different seed. The on-demand v4-32 is the reliable result;
# the 5 spot VMs are bonus seeds (may be preempted, results are extra).
#
# Outputs land in ./results/seed_<N>/ locally after each VM finishes.
#
# Usage:
#   bash tpu/launch_all_seeds.sh
#   (HF_TOKEN auto-loaded from .env)
#
# To cancel all jobs: kill $(cat /tmp/bohdi_tpu_pids.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

GCS_DATA_PATH="${GCS_DATA_PATH:-}"
PROJECT="tokyo-micron-494016-s9"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# Each entry: "VM_NAME  ACCELERATOR_TYPE  ZONE  RUNTIME  SPOT_FLAG  ACCEL_CONFIG  SEED"
JOBS=(
    "bohdi-v4-ondemand  v4-32   us-central2-b   tpu-vm-base  ''       accelerate_config_v4_32.yaml    42"
    "bohdi-v4-spot      v4-32   us-central2-b   tpu-vm-base  --spot   accelerate_config_v4_32.yaml    123"
    "bohdi-v5e-usw      v5e-64  us-central1-a   tpu-vm-base  --spot   accelerate_config_v5e64.yaml    456"
    "bohdi-v5e-eu       v5e-64  europe-west4-b  tpu-vm-base  --spot   accelerate_config_v5e64.yaml    789"
    "bohdi-v6e-use      v6e-64  us-east1-d      tpu-vm-base  --spot   accelerate_config_v6e64.yaml    1337"
    "bohdi-v6e-eu       v6e-64  europe-west4-a  tpu-vm-base  --spot   accelerate_config_v6e64.yaml    2024"
)

echo "Launching ${#JOBS[@]} VMs in parallel..."
echo "Seeds: 42 (on-demand, reliable) + 123 456 789 1337 2024 (spot, bonus)"

PID_FILE="/tmp/bohdi_tpu_pids.txt"
> "$PID_FILE"

for JOB in "${JOBS[@]}"; do
    read -r VM_NAME TPU_TYPE ZONE RUNTIME SPOT_FLAG ACCEL_CFG SEED <<< "$JOB"

    echo "Launching $VM_NAME ($TPU_TYPE, zone $ZONE, seed $SEED)..."

    (
        set -euo pipefail
        SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
        mkdir -p "$SEED_DIR"
        LOG="${SEED_DIR}/tpu_launch.log"

        # Create the TPU VM
        # shellcheck disable=SC2086
        gcloud compute tpus tpu-vm create "$VM_NAME" \
            --zone="$ZONE" \
            --accelerator-type="$TPU_TYPE" \
            --version="$RUNTIME" \
            --project="$PROJECT" \
            ${SPOT_FLAG} 2>&1 | tee -a "$LOG" || {
                echo "[$VM_NAME] VM creation failed — skipping" | tee -a "$LOG"
                exit 0
            }

        cleanup() {
            echo "[$VM_NAME] Cleaning up..." | tee -a "$LOG"
            gcloud compute tpus tpu-vm delete "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null || true
        }
        trap cleanup EXIT

        # Run setup + training
        gcloud compute tpus tpu-vm ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --command="
set -euo pipefail

git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
cd ~/bohdi-lora

bash tpu/setup_tpu.sh

mkdir -p data/sft eval checkpoints logs

if [ -n '${GCS_DATA_PATH}' ]; then
    gsutil -m cp '${GCS_DATA_PATH}/train.jsonl' data/sft/train.jsonl
    gsutil -m cp '${GCS_DATA_PATH}/val.jsonl'   data/sft/val.jsonl
fi

export HF_TOKEN='${HF_TOKEN}'

accelerate launch \
    --config_file tpu/${ACCEL_CFG} \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --seed ${SEED} \
    --output-dir checkpoints
" 2>&1 | tee -a "$LOG"

        # Copy checkpoints back
        gcloud compute tpus tpu-vm scp \
            --recurse \
            --zone="$ZONE" \
            --project="$PROJECT" \
            "${VM_NAME}:~/bohdi-lora/checkpoints" "$SEED_DIR/" 2>&1 | tee -a "$LOG"

        echo "[$VM_NAME] Done. Checkpoints in ${SEED_DIR}/checkpoints/" | tee -a "$LOG"

    ) &

    echo "$!" >> "$PID_FILE"
    sleep 2   # stagger launches slightly to avoid gcloud API rate limits
done

echo ""
echo "All VMs launched. PIDs: $(cat "$PID_FILE")"
echo "To cancel everything: kill \$(cat $PID_FILE)"
echo ""
echo "Waiting for all jobs to finish..."
wait

echo ""
echo "All done. Results in $RESULTS_DIR/"
ls "$RESULTS_DIR/"
