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
echo "=== One-time setup (deps + repo) ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
set -euo pipefail

git clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
cd ~/bohdi-lora
bash tpu/setup_tpu.sh
mkdir -p data/raw data/sft eval logs
export HF_TOKEN='${HF_TOKEN}'

# ── Stage 1: generate traces ──────────────────────────────────────────────────
echo '=== Stage 1: generate BOHDI traces (MedGemma-27B) ==='
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --output data/sft/raw_traces.jsonl \
    --use-bodhi \
    --max-examples \${MAX_EXAMPLES:-4800}

# ── Stage 2: grade + filter ───────────────────────────────────────────────────
echo '=== Stage 2: grade and filter traces (Qwen2.5-14B) ==='
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

    # ── Stage 4: evaluate all 4 configurations ────────────────────────────────
    echo "=== Stage 4: evaluation (4 configs) for seed $SEED ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
set -euo pipefail
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

echo ""
echo "All seeds done: $SEEDS"
echo "Results in ./results/"
