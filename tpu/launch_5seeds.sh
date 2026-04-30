#!/bin/bash
# launch_5seeds.sh — fan out 5 v6e-8 spot VMs in parallel, one seed per VM.
#
# Why 5: scripts/aggregate_seeds.py only emits across-seed percentile 95%
# CIs when n_seeds >= 5 (line 101 of that file). 5 is also the default
# in scripts/run_multi_seed.sh ("42 7 13 99 101"). Going to 5 from 3
# unlocks the across-seed CI band the paper needs.
#
# Quota: TRC grants 64 v6e chips in europe-west4-a + 64 in us-east1-d.
# 5 v6e-8 = 40 chips total. We split 3 in eur4a (24 chips) + 2 in use1d
# (16 chips) — both well within the 64-chip-per-zone quota.
#
# Per-VM workflow:
#   1. acquire v6e-8 spot (with retry on capacity errors)
#   2. clone repo, bash tpu/setup_tpu.sh
#   3. either
#        - download pre-graded ${GCS_DATA_PATH}/{train,val}.jsonl (skip Stage 1+2)
#        - or run Stage 1 (gen) + Stage 2 (filter) themselves
#   4. Stage 3: train LoRA (configs/lora_medgemma27b_tpu.yaml, --seed N)
#   5. Stage 4: eval 4 configs (base_no_wrapper / base_bodhi /
#      lora_no_wrapper / lora_bodhi). LoRA configs use the XLA direct-
#      inference backend (PR #93) since vllm-tpu lacks add_lora.
#   6. SCP results back to ./results/seed_<N>/ on the local box
#   7. delete the VM (trap on EXIT)
#
# Each VM writes ~/pipeline.log + ~/{setup,gen,filter,train,eval}.log.
# scripts/dashboard polls these — open http://localhost:8000 while it runs.
#
# Wall-clock estimate (no preempts):
#   Stage 1+2 shared via GCS:  ~5 h once
#   Stage 3 train (parallel):  ~12-24 h per seed (max wins, not sum)
#   Stage 4 eval (parallel):   ~6 h per seed
#   Total wall:                ~25-35 h
#
# Usage:
#   export HF_TOKEN=...                              # required (in .env is fine)
#   GCS_DATA_PATH=gs://bucket/path                   # optional, skips Stage 1+2
#   SEEDS="42 7 13 99 101"                           # optional override (default 42 7 13 99 101)
#   bash tpu/launch_5seeds.sh
#
# Cancel everything (clean up all 5 VMs):
#   kill $(cat /tmp/bohdi_5seeds_pids.txt)
#   for n in 42 7 13; do
#       gcloud compute tpus tpu-vm delete bohdi-seed$n --zone=europe-west4-a --quiet
#   done
#   for n in 99 101; do
#       gcloud compute tpus tpu-vm delete bohdi-seed$n --zone=us-east1-d --quiet
#   done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

PROJECT="tokyo-micron-494016-s9"
RUNTIME="v2-alpha-tpuv6e"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# 5 seeds split 3 in europe-west4-a + 2 in us-east1-d. 24 chips in eur4a
# and 16 chips in use1d both fit the 64-chip quota; if one zone runs hot,
# the other still hosts its share. Default seed list mirrors
# scripts/run_multi_seed.sh so multi-seed comparisons stay consistent.
SEEDS="${SEEDS:-42 7 13 99 101}"
read -r -a SEED_ARR <<< "$SEEDS"
if [ "${#SEED_ARR[@]}" -ne 5 ]; then
    echo "ERROR: SEEDS must list exactly 5 seeds (got: ${#SEED_ARR[@]})" >&2
    exit 1
fi

# Index 0..2 -> eur4a (3 VMs), index 3..4 -> use1d (2 VMs).
ZONES=("europe-west4-a" "europe-west4-a" "europe-west4-a" "us-east1-d" "us-east1-d")
VM_NAMES=(
    "bohdi-seed${SEED_ARR[0]}"
    "bohdi-seed${SEED_ARR[1]}"
    "bohdi-seed${SEED_ARR[2]}"
    "bohdi-seed${SEED_ARR[3]}"
    "bohdi-seed${SEED_ARR[4]}"
)

# Optional shared data location. When set, each VM skips Stage 1+2 and
# downloads ${GCS_DATA_PATH}/train.jsonl + val.jsonl. Recommended — saves
# ~5h of duplicated trace generation per VM (15h total).
GCS_DATA_PATH="${GCS_DATA_PATH:-}"

# How many times to retry spot-create on TRC capacity errors before
# giving up on a particular VM. Each retry waits 60s.
CREATE_RETRIES="${CREATE_RETRIES:-5}"

echo "=== launch_5seeds: 5 v6e-8 spot VMs in parallel (3 eur4a + 2 use1d) ==="
echo "  seeds: ${SEEDS}"
echo "  GCS_DATA_PATH: ${GCS_DATA_PATH:-(not set — each VM will run Stage 1+2)}"
echo "  results -> $RESULTS_DIR/seed_<N>/"
echo

PID_FILE="/tmp/bohdi_5seeds_pids.txt"
> "$PID_FILE"

# Build the remote pipeline command once — same script on every VM, only
# the SEED env var differs per VM. We embed HF_TOKEN inline (visible in
# the gcloud SSH command's argv); if you need stricter token hygiene,
# write it to ~/.bohdi-env on the VM via stdin and source it inside the
# heredoc instead.
build_remote_cmd() {
    local SEED="$1"
    cat <<REMOTE
set -euo pipefail
export PJRT_DEVICE=TPU
export HF_TOKEN='${HF_TOKEN}'

if [ ! -d ~/bohdi-lora ]; then
    git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
fi
cd ~/bohdi-lora

echo "--- 0/4 setup_tpu.sh ---" | tee -a ~/pipeline.log
bash tpu/setup_tpu.sh > ~/setup.log 2>&1 || { echo "setup FAILED" >> ~/pipeline.log; exit 1; }
echo SETUP_OK >> ~/pipeline.log

mkdir -p data/sft eval checkpoints logs

if [ -n "${GCS_DATA_PATH}" ]; then
    echo "--- 1+2/4 download pre-graded data from ${GCS_DATA_PATH} ---" | tee -a ~/pipeline.log
    gsutil -m cp "${GCS_DATA_PATH}/train.jsonl" data/sft/train.jsonl
    gsutil -m cp "${GCS_DATA_PATH}/val.jsonl"   data/sft/val.jsonl
    echo DATA_DOWNLOADED >> ~/pipeline.log
else
    echo "--- 1/4 generate BODHI traces (this VM, no GCS shortcut) ---" | tee -a ~/pipeline.log
    python -u scripts/download_data.py >> ~/pipeline.log 2>&1
    python -u scripts/generate_traces.py \\
        --model google/medgemma-27b-text-it \\
        --datasets healthbench_hard healthbench \\
        --output data/sft/raw_traces.jsonl \\
        --use-bodhi \\
        > ~/gen.log 2>&1
    echo GEN_OK >> ~/pipeline.log

    echo "--- 2/4 filter+grade with seed ${SEED} ---" | tee -a ~/pipeline.log
    python -u scripts/filter_traces.py \\
        --input data/sft/raw_traces.jsonl \\
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \\
        --grader-model Qwen/Qwen2.5-14B-Instruct \\
        --output-dir data/sft \\
        --min-score 0.4 \\
        --val-ratio 0.1 \\
        --seed ${SEED} \\
        > ~/filter.log 2>&1
    echo FILTER_OK >> ~/pipeline.log
fi

echo "--- 3/4 train LoRA seed=${SEED} ---" | tee -a ~/pipeline.log
python -u scripts/train_lora.py \\
    --config configs/lora_medgemma27b_tpu.yaml \\
    --seed ${SEED} \\
    --output-dir "checkpoints/seed_${SEED}" \\
    > ~/train.log 2>&1
echo TRAIN_OK >> ~/pipeline.log

echo "--- 4/4 eval 4 configs (base/lora x wrapper/no-wrapper) ---" | tee -a ~/pipeline.log
mkdir -p "eval/seed_${SEED}"
LORA_DIR="checkpoints/seed_${SEED}/best"

run_eval() {
    local name="\$1" args="\$2"
    local out="eval/seed_${SEED}/\${name}.json"
    if [ -s "\$out" ]; then
        echo "[\$name] already exists, skipping" >> ~/pipeline.log
        return
    fi
    echo "--- 4.\$name ---" | tee -a ~/pipeline.log
    # shellcheck disable=SC2086
    python -u scripts/eval_healthbench.py \$args \\
        --sample-ids data/raw/hard_200_sample_ids.json \\
        --grader-model Qwen/Qwen2.5-14B-Instruct \\
        --output "\$out" \\
        --seed ${SEED} >> ~/eval.log 2>&1 || echo "eval \$name FAILED" >> ~/pipeline.log
}

run_eval "base_no_wrapper"  "--model google/medgemma-27b-text-it"
run_eval "base_bodhi"       "--model google/medgemma-27b-text-it --use-bodhi"
run_eval "lora_no_wrapper"  "--model google/medgemma-27b-text-it --lora-path \$LORA_DIR"
run_eval "lora_bodhi"       "--model google/medgemma-27b-text-it --lora-path \$LORA_DIR --use-bodhi"
echo EVAL_OK >> ~/pipeline.log

echo "=== seed ${SEED} pipeline complete ===" >> ~/pipeline.log
REMOTE
}

for i in 0 1 2 3 4; do
    SEED="${SEED_ARR[$i]}"
    ZONE="${ZONES[$i]}"
    VM_NAME="${VM_NAMES[$i]}"
    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    LOG="${SEED_DIR}/launch.log"

    echo "Launching $VM_NAME (seed $SEED, $ZONE)..."

    (
        set -euo pipefail

        # Acquire the VM with a retry loop. TRC v6e-8 spot has
        # occasional "internal error" capacity wobbles; we retry a few
        # times before giving up on this seed.
        attempt=0
        until gcloud compute tpus tpu-vm create "$VM_NAME" \
            --zone="$ZONE" \
            --accelerator-type="v6e-8" \
            --version="$RUNTIME" \
            --project="$PROJECT" \
            --spot 2>&1 | tee -a "$LOG"; do
            attempt=$((attempt + 1))
            if [ "$attempt" -ge "$CREATE_RETRIES" ]; then
                echo "[$VM_NAME] giving up after $attempt attempts" | tee -a "$LOG"
                exit 0
            fi
            echo "[$VM_NAME] capacity error, retry $attempt/$CREATE_RETRIES in 60s..." | tee -a "$LOG"
            sleep 60
        done

        # Trap: always copy results back and delete the VM, even on
        # SIGINT or pipeline failure.
        cleanup() {
            echo "[$VM_NAME] copying results + deleting VM..." | tee -a "$LOG"
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/checkpoints/seed_${SEED}" "$SEED_DIR/" \
                2>&1 | tee -a "$LOG" || echo "  (no checkpoints to copy)" | tee -a "$LOG"
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/eval/seed_${SEED}" "$SEED_DIR/" \
                2>&1 | tee -a "$LOG" || echo "  (no eval to copy)" | tee -a "$LOG"
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/pipeline.log" "${SEED_DIR}/pipeline.log" \
                2>&1 | tee -a "$LOG" || true
            gcloud compute tpus tpu-vm delete "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null || true
            echo "[$VM_NAME] cleaned up" | tee -a "$LOG"
        }
        trap cleanup EXIT

        # Run the pipeline. SSH stays open until the remote command
        # returns; on preempt/network blip the SSH dies but the VM keeps
        # running — re-running this script will reuse the existing VM
        # and the eval-skip-if-output-exists logic resumes.
        REMOTE_CMD="$(build_remote_cmd "$SEED")"
        gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
            --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
            --command="$REMOTE_CMD" 2>&1 | tee -a "$LOG"

        echo "[$VM_NAME] pipeline finished" | tee -a "$LOG"
    ) &

    echo "$!" >> "$PID_FILE"
    sleep 4   # stagger gcloud creates a bit so we don't hammer the API
done

echo
echo "All 5 jobs spawned. PIDs: $(cat "$PID_FILE")"
echo "Open the dashboard at http://localhost:8000 to watch progress."
echo
echo "Waiting for all VMs to finish..."
wait
echo
echo "All seeds done. Results in $RESULTS_DIR/seed_*/"
ls -la "$RESULTS_DIR" 2>/dev/null || true
