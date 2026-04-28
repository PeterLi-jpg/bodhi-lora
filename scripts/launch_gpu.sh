#!/bin/bash
# launch_gpu.sh — Stage 3 (LoRA SFT) + Stage 4 (HealthBench eval) on a GPU box.
#
# Assumes you already have:
#   ~/bohdi-lora/                         (this repo, git-cloned)
#   ~/bohdi-lora/data/sft/train.jsonl     (838 graded SFT examples, from
#   ~/bohdi-lora/data/sft/val.jsonl       (93)   the rescued TPU run)
#   ~/bohdi-lora/data/raw/hard_200_sample_ids.json  (eval prompt ids)
#
# Optional:
#   HF_TOKEN     in the environment (gated MedGemma needs this)
#   SEED         seed (default 42)
#   SKIP_DEPS=1  skip pip install (fast re-runs)
#
# What it does:
#   1. pip install train + eval deps
#   2. Stage 3: python scripts/train_lora.py ...   (writes checkpoints/seed_{SEED}/)
#   3. Stage 4: 4× eval_healthbench.py (base/lora × wrapper/no-wrapper)
#   4. Post-processing: ushape, plots, rubric_diff
#
# Skips Stages 1+2 entirely — those already ran on the TPU side and the
# graded SFT data has been copied into data/sft/ for us.
#
# This script is GPU-only.  The TPU pipeline lives in tpu/launch_multiseed.sh.

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root

SEED="${SEED:-42}"
MODEL="google/medgemma-27b-text-it"
TRAIN_CONFIG="configs/lora_medgemma27b_tpu.yaml"   # config is hardware-agnostic
LORA_DIR="checkpoints/seed_${SEED}/best"
EVAL_DIR="results/seed_${SEED}/eval"
FIG_DIR="results/seed_${SEED}/figures"
IDS="data/raw/hard_200_sample_ids.json"
HB="data/raw/healthbench_hard.jsonl"
EVAL_MAX_FLAG="${EVAL_MAX:+--max-examples ${EVAL_MAX}}"

# ── 1. install deps (once) ──────────────────────────────────────────────────
if [ "${SKIP_DEPS:-0}" != "1" ]; then
    echo "=== Installing python deps ==="
    pip install --quiet \
        "torch>=2.5,<2.9" \
        "transformers==4.57.6" \
        "peft==0.19.1" \
        "trl==0.11.4" \
        "accelerate==1.13.0" \
        "datasets>=2.18.0,<4.0.0" \
        "bodhi-llm[all]==0.1.4" \
        "pyyaml>=6.0,<7.0" \
        "numpy>=1.24,<3.0" \
        "tqdm>=4.65" \
        "rich>=13.0,<15.0" \
        "matplotlib>=3.7,<4.0"
    # vLLM for Stage 4 inference.  In subprocess mode (used inside a cloud
    # GPU pod where there is no host docker daemon), `vllm serve` is run
    # directly as a Python subprocess by _vllm_engine.py — so we need vllm
    # pip-installed here.  pip auto-selects the CUDA wheel on Nvidia hosts.
    # No `docker pull` step in this path: the pod is already a container.
    pip install --quiet "vllm>=0.6.5,<0.10"
fi

# ── 2. Stage 3: LoRA SFT ───────────────────────────────────────────────────
echo "=== Stage 3: LoRA SFT (seed ${SEED}) ==="
mkdir -p "checkpoints/seed_${SEED}"
python -u scripts/train_lora.py \
    --config "${TRAIN_CONFIG}" \
    --seed "${SEED}" \
    --output-dir "checkpoints/seed_${SEED}"

if [ ! -d "${LORA_DIR}" ]; then
    echo "ERROR: training finished but ${LORA_DIR} not found — Stage 4 will fail" >&2
    exit 1
fi

# ── 3. Stage 4: HealthBench Hard 200 eval, 4 configs ───────────────────────
mkdir -p "${EVAL_DIR}" "${FIG_DIR}"

# Pre-download HealthBench Hard so eval doesn't redownload 4 times.
if [ ! -f "${HB}" ]; then
    echo "=== Pre-downloading HealthBench Hard ==="
    mkdir -p data/raw
    python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl',
    '${HB}')"
fi

# SKIP wrapper: don't redo an eval whose output already exists (rerunning
# the script after a partial Stage 4 should resume, not redo).
run_eval() {
    local name="$1"
    local args="$2"
    local out="${EVAL_DIR}/${name}"
    if [ -s "${out}" ]; then
        echo "  ${name} already exists, skipping"
        return
    fi
    echo "=== Stage 4: ${name} ==="
    # shellcheck disable=SC2086
    python -u scripts/eval_healthbench.py ${args} --output "${out}"
}

run_eval "base_no_wrapper.json" \
    "--model ${MODEL} --sample-ids ${IDS} ${EVAL_MAX_FLAG}"

run_eval "base_bodhi.json" \
    "--model ${MODEL} --use-bodhi --sample-ids ${IDS} ${EVAL_MAX_FLAG}"

run_eval "lora_no_wrapper.json" \
    "--model ${MODEL} --lora-path ${LORA_DIR} --sample-ids ${IDS} ${EVAL_MAX_FLAG}"

run_eval "lora_bodhi.json" \
    "--model ${MODEL} --lora-path ${LORA_DIR} --use-bodhi --sample-ids ${IDS} ${EVAL_MAX_FLAG}"

# ── 4. post-processing ─────────────────────────────────────────────────────
echo "=== Post-processing: ushape + plots + rubric_diff ==="
python -u scripts/eval_ushape.py \
    --eval-jsons \
        "${EVAL_DIR}/base_no_wrapper.json" \
        "${EVAL_DIR}/base_bodhi.json" \
        "${EVAL_DIR}/lora_no_wrapper.json" \
        "${EVAL_DIR}/lora_bodhi.json" \
    --healthbench "${HB}" \
    --output "${EVAL_DIR}/ushape.json"

python -u scripts/plot_ushape.py \
    --input "${EVAL_DIR}/ushape.json" \
    --eval-jsons \
        "${EVAL_DIR}/base_no_wrapper.json" \
        "${EVAL_DIR}/base_bodhi.json" \
        "${EVAL_DIR}/lora_no_wrapper.json" \
        "${EVAL_DIR}/lora_bodhi.json" \
    --healthbench "${HB}" \
    --n-bins 10 \
    --out-dir "${FIG_DIR}"

if [ -f "${LORA_DIR}/trainer_state.json" ]; then
    python -u scripts/plot_training.py \
        --trainer-state "${LORA_DIR}/trainer_state.json" \
        --output "${FIG_DIR}/training_loss.png"
fi

python -u scripts/rubric_diff.py \
    "${EVAL_DIR}/base_no_wrapper.json" \
    "${EVAL_DIR}/lora_bodhi.json" \
    --output "${EVAL_DIR}/rubric_diff.json"

echo
echo "=== Done ==="
echo "checkpoints: ${LORA_DIR}"
echo "evals:       ${EVAL_DIR}/"
echo "figures:     ${FIG_DIR}/"
