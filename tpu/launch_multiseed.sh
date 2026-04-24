#!/bin/bash
# launch_multiseed.sh — run the full pipeline on one on-demand v4-32 VM.
#
# Runs all three stages on the same VM:
#   Stage 1 — generate BOHDI traces (MedGemma-27B inference)
#   Stage 2 — grade + filter traces (Qwen2.5-14B, non-AWQ, fits on one chip)
#   Stage 3 — LoRA SFT across all 32 chips, one run per seed
#
# Quota: 32 on-demand Cloud TPU v4 chips in zone us-central2-b (never preempted).
# Estimated total time: ~3-5 hours (generation dominates).
#
# Usage:
#   bash tpu/launch_multiseed.sh                        # seeds 42 123 456
#   SEEDS="42 123" bash tpu/launch_multiseed.sh         # custom seeds
#   MAX_EXAMPLES=100 bash tpu/launch_multiseed.sh       # smaller run for testing
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

# Optional training-method overrides (TPU: 4bit/8bit will error; only variant/rank).
#   LORA_VARIANT=dora bash tpu/launch_multiseed.sh
#   LORA_VARIANT=rslora bash tpu/launch_multiseed.sh
#   LORA_R=32 bash tpu/launch_multiseed.sh
QUANT="${QUANT:-}"
LORA_VARIANT="${LORA_VARIANT:-}"
LORA_R="${LORA_R:-}"

# Build the extra-flags string to pass through to the remote SSH command.
TRAIN_EXTRA_FLAGS=""
[ -n "$QUANT" ]        && TRAIN_EXTRA_FLAGS="$TRAIN_EXTRA_FLAGS --quantization $QUANT"
[ -n "$LORA_VARIANT" ] && TRAIN_EXTRA_FLAGS="$TRAIN_EXTRA_FLAGS --lora-variant $LORA_VARIANT"
[ -n "$LORA_R" ]       && TRAIN_EXTRA_FLAGS="$TRAIN_EXTRA_FLAGS --lora-r $LORA_R"

PROJECT="tokyo-micron-494016-s9"
TPU_NAME="bohdi-lora-v4"
TPU_TYPE="v6e-8"
TPU_RUNTIME="v2-alpha-tpuv6e"
ZONE="us-east1-d"

echo "Seeds to run: $SEEDS"
echo "TPU type will be shown once a slot is acquired"
echo ""

# TRC-granted slots — v6e-8 spot in TRC regions only.
# v6e-8 = 8 chips × 32 GB HBM = 256 GB; spot OK for TRC quota.
# Format: "TPU_TYPE ZONE SPOT(yes/no) ACCEL_CONFIG"
TRC_SLOTS=(
    "v6e-8 us-east1-d     yes tpu/accelerate_config_v6e8.yaml"
    "v6e-8 europe-west4-a yes tpu/accelerate_config_v6e8.yaml"
)

# Try each slot in one pass, then sleep and retry the whole list.
# 200 rounds * 5 min = ~16 hours of overnight retrying. Override: MAX_ROUNDS=10 bash tpu/launch_multiseed.sh
MAX_ROUNDS="${MAX_ROUNDS:-200}"
RETRY_DELAY="${RETRY_DELAY:-300}"

echo "=== Acquiring TPU VM from TRC quota ==="
ACCEL_CFG=""
found=false

# If a VM with this name already exists (e.g. from a previous interrupted run),
# detect its zone/type and reuse it rather than failing with ALREADY_EXISTS.
for slot in "${TRC_SLOTS[@]}"; do
    read -r _TYPE _ZONE _SPOT _CFG <<< "$slot"
    EXISTING=$(gcloud compute tpus tpu-vm list --zone="$_ZONE" --project="$PROJECT" \
        --format="value(name,state)" 2>/dev/null | grep "^${TPU_NAME}\b" || true)
    if [ -n "$EXISTING" ]; then
        STATE=$(echo "$EXISTING" | awk '{print $2}')
        echo "Found existing VM $TPU_NAME in $_ZONE (state=$STATE) — waiting for READY..."
        while [ "$STATE" != "READY" ]; do
            sleep 15
            STATE=$(gcloud compute tpus tpu-vm list --zone="$_ZONE" --project="$PROJECT" \
                --format="value(state)" 2>/dev/null | head -1)
            echo "  state: $STATE"
            # VM vanished (preempted or failed during CREATING) — fall through to fresh create
            if [ -z "$STATE" ]; then
                echo "VM $TPU_NAME disappeared from $_ZONE, will try a fresh create."
                break
            fi
        done
        [ "$STATE" != "READY" ] && continue
        TPU_TYPE="$_TYPE"; ZONE="$_ZONE"; ACCEL_CFG="$_CFG"
        echo "Reusing existing VM: $_TYPE in $_ZONE"
        found=true
        break
    fi
done

for round in $(seq 1 $MAX_ROUNDS); do
    $found && break
    for slot in "${TRC_SLOTS[@]}"; do
        read -r _TYPE _ZONE _SPOT _CFG <<< "$slot"
        SPOT_FLAG=""
        [ "$_SPOT" = "yes" ] && SPOT_FLAG="--spot"
        echo "  Trying $_TYPE ($(echo $_TYPE | grep -o '[0-9]*$') chips) in $_ZONE (spot=$_SPOT)..."
        if gcloud compute tpus tpu-vm create "$TPU_NAME" \
            --zone="$_ZONE" \
            --accelerator-type="$_TYPE" \
            --version="$TPU_RUNTIME" \
            --project="$PROJECT" \
            $SPOT_FLAG 2>&1; then
            TPU_TYPE="$_TYPE"
            ZONE="$_ZONE"
            ACCEL_CFG="$_CFG"
            echo "VM created: $_TYPE ($(echo $_TYPE | grep -o '[0-9]*$') chips) in $_ZONE (spot=$_SPOT)"
            found=true
            break
        fi
    done
    $found && break
    echo "No capacity found in any TRC zone (round $round/$MAX_ROUNDS) — waiting ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done

if ! $found; then
    echo "ERROR: could not get capacity in any TRC zone after $MAX_ROUNDS rounds."
    exit 1
fi

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
echo "=== One-time setup (deps + repo) ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail
git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora 2>/dev/null || (cd ~/bohdi-lora && git pull)
cd ~/bohdi-lora
bash tpu/setup_tpu.sh
# Ensure jinja2 meets apply_chat_template requirement (>=3.1.0).
# setup_tpu.sh pins it, but transitive deps can downgrade it; re-pin here.
export PATH=\"\$HOME/.local/bin:\$PATH\"
pip install -q \"jinja2>=3.1.0\"
mkdir -p data/raw data/sft eval logs
"

# ── Stage 1: generate traces (separate SSH so a failure here is identifiable
#    and restartable without re-running setup) ─────────────────────────────────
echo "=== Stage 1: generate BOHDI traces (MedGemma-27B) ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
export HF_TOKEN='${HF_TOKEN}'
# --resume-from means a restart after preemption or SSH drop continues from
# the last completed example rather than starting over.
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --output data/sft/raw_traces.jsonl \
    --resume-from data/sft/raw_traces.jsonl \
    --use-bodhi \
    --max-examples \${MAX_EXAMPLES:-4800}
echo \"Generate done: \$(wc -l < data/sft/raw_traces.jsonl) traces\"
"

# ── Stage 2: grade + filter (separate SSH — runs fresh Qwen2.5-14B grader) ───
echo "=== Stage 2: grade and filter traces (Qwen2.5-14B) ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
export HF_TOKEN='${HF_TOKEN}'
python scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output-dir data/sft \
    --min-score 0.4
echo \"Training data ready: \$(wc -l < data/sft/train.jsonl) train, \$(wc -l < data/sft/val.jsonl) val\"
"

# ── Train each seed sequentially ──────────────────────────────────────────────
for SEED in $SEEDS; do
    echo ""
    echo "=== Stage 3: training seed $SEED ==="

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
mkdir -p checkpoints/seed_${SEED}
export HF_TOKEN='${HF_TOKEN}'

accelerate launch \
    --config_file "${ACCEL_CFG}" \
    scripts/train_lora.py \
    --config configs/lora_medgemma27b_tpu.yaml \
    --seed ${SEED} \
    --output-dir checkpoints/seed_${SEED} \
    ${TRAIN_EXTRA_FLAGS}

echo 'Seed ${SEED} done.'
"

    # ── Stage 4: evaluate all 4 configurations ────────────────────────────────
    echo "=== Stage 4: evaluation (4 configs) for seed $SEED ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
export HF_TOKEN='${HF_TOKEN}'

MODEL='google/medgemma-27b-text-it'
IDS='data/raw/hard_200_sample_ids.json'
LORA='checkpoints/seed_${SEED}/best'
EVAL_DIR='eval/seed_${SEED}'
FIG_DIR='figures/seed_${SEED}'
mkdir -p \"\$EVAL_DIR\" \"\$FIG_DIR\"

echo '-- base, no wrapper --'
python scripts/eval_healthbench.py --model \"\$MODEL\" --sample-ids \"\$IDS\" --output \"\$EVAL_DIR/base_no_wrapper.json\"

echo '-- base + BOHDI wrapper --'
python scripts/eval_healthbench.py --model \"\$MODEL\" --use-bodhi --sample-ids \"\$IDS\" --output \"\$EVAL_DIR/base_bodhi.json\"

echo '-- LoRA, no wrapper --'
python scripts/eval_healthbench.py --model \"\$MODEL\" --lora-path \"\$LORA\" --sample-ids \"\$IDS\" --output \"\$EVAL_DIR/lora_no_wrapper.json\"

echo '-- LoRA + BOHDI wrapper --'
python scripts/eval_healthbench.py --model \"\$MODEL\" --lora-path \"\$LORA\" --use-bodhi --sample-ids \"\$IDS\" --output \"\$EVAL_DIR/lora_bodhi.json\"

echo '-- U-shape analysis --'
python scripts/eval_ushape.py \
    --eval-jsons \"\$EVAL_DIR/base_no_wrapper.json\" \"\$EVAL_DIR/base_bodhi.json\" \
                  \"\$EVAL_DIR/lora_no_wrapper.json\" \"\$EVAL_DIR/lora_bodhi.json\" \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output \"\$EVAL_DIR/ushape.json\"

echo '-- plots --'
python scripts/plot_ushape.py \
    --input \"\$EVAL_DIR/ushape.json\" \
    --eval-jsons \"\$EVAL_DIR/base_no_wrapper.json\" \"\$EVAL_DIR/base_bodhi.json\" \
                  \"\$EVAL_DIR/lora_no_wrapper.json\" \"\$EVAL_DIR/lora_bodhi.json\" \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --n-bins 10 --out-dir \"\$FIG_DIR\"

if [ -f \"\$LORA/trainer_state.json\" ]; then
    python scripts/plot_training.py --trainer-state \"\$LORA/trainer_state.json\" --output \"\$FIG_DIR/training_loss.png\"
fi

echo '-- rubric diff (base_no_wrapper → lora_bodhi) --'
python scripts/rubric_diff.py \
    \"\$EVAL_DIR/base_no_wrapper.json\" \
    \"\$EVAL_DIR/lora_bodhi.json\" \
    --output \"\$EVAL_DIR/rubric_diff.json\"

echo 'Eval done for seed ${SEED}.'
"

    # Pull checkpoints + eval results back
    echo "Copying seed $SEED outputs..."
    mkdir -p "./results/seed_${SEED}"
    gcloud compute tpus tpu-vm scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/checkpoints/seed_${SEED}" "./results/seed_${SEED}/"
    gcloud compute tpus tpu-vm scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/eval/seed_${SEED}"        "./results/seed_${SEED}/"
    gcloud compute tpus tpu-vm scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/figures/seed_${SEED}"     "./results/seed_${SEED}/"

    echo "Seed $SEED complete — results in ./results/seed_${SEED}/"
done

# ── Aggregate across all seeds ────────────────────────────────────────────────
echo ""
echo "=== Aggregate: multi-seed summary ==="
SEED_DIRS_ARG=""
for SEED in $SEEDS; do
    SEED_DIRS_ARG="$SEED_DIRS_ARG eval/seed_${SEED}"
done

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
python scripts/aggregate_seeds.py \
    --seed-dirs $SEED_DIRS_ARG \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/multi_seed_summary.json
echo 'Aggregate done.'
"

mkdir -p ./results
gcloud compute tpus tpu-vm scp --zone="$ZONE" --project="$PROJECT" \
    "${TPU_NAME}:~/bohdi-lora/eval/multi_seed_summary.json" "./results/"

echo ""
echo "All seeds done: $SEEDS"
echo "Results in ./results/"
echo "Multi-seed summary: ./results/multi_seed_summary.json"
