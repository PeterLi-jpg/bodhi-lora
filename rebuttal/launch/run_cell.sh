#!/usr/bin/env bash
# run_cell.sh — run ONE (model x benchmark) cell of the rebuttal generality grid.
#
# GPU-only, one clean pipeline (same stages as gpu/launch_multiseed.sh, generalized
# to any base model + benchmark). Seeds are parallelized across CURRENTLY-IDLE GPUs
# only (never touches a GPU another job is using), one training process per GPU.
#
# Usage:
#   MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
#   CONFIG=rebuttal/configs/lora_mistral_small_24b_qlora.yaml \
#   BENCH=healthbench SEEDS="42 7 13 99 101" \
#   bash rebuttal/launch/run_cell.sh
#
# BENCH in {healthbench, medqa, medquad}.
set -euo pipefail
cd "$(dirname "$0")/../.."                    # repo root
REPO="$PWD"
# Two Python envs: training needs transformers 4.57.6, but vLLM serving needs an
# older, incompatible transformers (the 'aimv2' config clash), so the inference
# stages (generate / filter / eval) run in a separate .venv-infer.
TRAIN_PY="${TRAIN_PY:-$REPO/.venv/bin/python}"
INFER_PY="${INFER_PY:-$REPO/.venv-infer/bin/python}"

MODEL="${MODEL:?set MODEL}"
CONFIG="${CONFIG:?set CONFIG}"
BENCH="${BENCH:?set BENCH (healthbench|medqa|medquad)}"
SEEDS="${SEEDS:-42 7 13 99 101}"
EVAL_HOLDOUT_N="${EVAL_HOLDOUT_N:-200}"       # eval prompts held out of training
GEN_MAX="${GEN_MAX:-4200}"                    # prompts to generate traces over
# Eval grader = Llama-3.1-8B (paper). meta-llama/* is gated; default to the
# non-gated NousResearch mirror (identical weights) so eval works without Meta access.
GRADER_MODEL="${GRADER_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
TAG="$(echo "${MODEL}" | tr '/:.' '___')"
WORK="results_rebuttal/${BENCH}__${TAG}"
SFT="${WORK}/sft"
mkdir -p "$WORK" "$SFT" data/raw logs

# HF token: reuse the box's cached token if not exported (never printed).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
: "${HF_TOKEN:?HF_TOKEN not set and no ~/.cache/huggingface/token found}"

# vLLM: run the venv's GPU vLLM as a subprocess. The auto-detected docker path
# pulls a vllm/vllm-tpu image (TPU) that hangs on this GPU box. Pin tensor-parallel
# to 1 because each stage runs on a single CUDA_VISIBLE_DEVICES GPU; raise VLLM_TP
# to tensor-parallel generation/eval across several idle GPUs when they are free.
export BODHI_VLLM_MODE="${BODHI_VLLM_MODE:-subprocess}"
export BODHI_VLLM_ACCEL="${BODHI_VLLM_ACCEL:-gpu}"
export BODHI_VLLM_TP="${VLLM_TP:-1}"

# Idle GPUs = 0% util AND <2GB used. Snapshotted per wave so we never grab a card
# another job started using between waves.
idle_gpus() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<2000 {print $1}'
}

echo "=== cell: $BENCH x $MODEL  (seeds: $SEEDS) ==="

# ---- Stage 0: benchmark data + eval holdout -------------------------------------
if [ "$BENCH" = "healthbench" ]; then
    BENCH_JSONL=""                                        # eval downloads HB-Hard
    SAMPLE_IDS="data/raw/hard_200_sample_ids.json"
    GEN_SRC=(--datasets healthbench_hard healthbench
             --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json)
    FILTER_DATA=(--healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl)
    EVAL_BENCH=()
else
    case "$BENCH" in
        medqa)   BENCH_JSONL="data/raw/medqa_open.jsonl" ;;
        medquad) BENCH_JSONL="data/raw/medquad.jsonl" ;;
        *) echo "unknown BENCH: $BENCH"; exit 1 ;;
    esac
    if [ ! -s "$BENCH_JSONL" ]; then
        "$INFER_PY" scripts/build_benchmark_jsonl.py --benchmark "$BENCH" --out "$BENCH_JSONL" --max "$((GEN_MAX + EVAL_HOLDOUT_N + 500))"
    fi
    SAMPLE_IDS="${WORK}/eval_ids.json"
    if [ ! -s "$SAMPLE_IDS" ]; then       # deterministic holdout = first N prompt_ids
        "$INFER_PY" - "$BENCH_JSONL" "$SAMPLE_IDS" "$EVAL_HOLDOUT_N" <<'PY'
import json, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = [json.loads(l)["prompt_id"] for l in open(src) if l.strip()][:n]
json.dump({"prompt_ids": ids}, open(out, "w"))
print(f"held out {len(ids)} eval ids -> {out}")
PY
    fi
    GEN_SRC=(--dataset-files "$BENCH_JSONL" --exclude-ids "$SAMPLE_IDS")
    FILTER_DATA=(--healthbench-data "$BENCH_JSONL")
    EVAL_BENCH=(--benchmark-jsonl "$BENCH_JSONL")
fi

# ---- Stage 1: generate BODHI traces (ONCE per cell) -----------------------------
RAW="${SFT}/raw_traces.jsonl"
if [ ! -s "$RAW" ]; then
    GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for generation}"
    echo "[gen] GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" BODHI_VLLM_PORT="$((8000 + GPU))" "$INFER_PY" scripts/generate_traces.py \
        --model "$MODEL" "${GEN_SRC[@]}" --use-bodhi \
        --output "$RAW" --max-examples "$GEN_MAX" 2>&1 | tee "logs/gen_${BENCH}_${TAG}.log"
fi

# ---- Stage 2: grade + filter (ONCE per cell) ------------------------------------
if [ ! -s "${SFT}/train.jsonl" ] || [ ! -s "${SFT}/val.jsonl" ]; then
    GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for filtering}"
    echo "[filter] GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" BODHI_VLLM_PORT="$((8000 + GPU))" "$INFER_PY" scripts/filter_traces.py \
        --input "$RAW" "${FILTER_DATA[@]}" \
        --output-dir "$SFT" --min-score 0.4 2>&1 | tee "logs/filter_${BENCH}_${TAG}.log"
fi
echo "train: $(wc -l < "${SFT}/train.jsonl")  val: $(wc -l < "${SFT}/val.jsonl")"

# ---- Stages 3-4: per-seed train + 2x2 eval, parallelized across idle GPUs --------
run_seed() {   # $1=seed  $2=gpu
    local SEED="$1" GPU="$2"
    local SD="${WORK}/seed_${SEED}" CK="${WORK}/seed_${SEED}/ckpt" EV="${WORK}/seed_${SEED}/eval"
    local LORA="${CK}/best"
    mkdir -p "$CK" "$EV"
    echo "[seed $SEED] train on GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "$TRAIN_PY" scripts/train_lora.py \
        --config "$CONFIG" --seed "$SEED" --output-dir "$CK" \
        --train-file "${SFT}/train.jsonl" --val-file "${SFT}/val.jsonl" \
        > "logs/train_${BENCH}_${TAG}_s${SEED}.log" 2>&1
    for spec in "base_no_wrapper::" "base_bodhi:--use-bodhi:" \
                "lora_no_wrapper::--lora-path ${LORA}" "lora_bodhi:--use-bodhi:--lora-path ${LORA}"; do
        local name="${spec%%:*}"
        local rest="${spec#*:}"
        local wrap="${rest%%:*}"
        local lora="${rest#*:}"
        CUDA_VISIBLE_DEVICES="$GPU" BODHI_VLLM_PORT="$((8000 + GPU))" "$INFER_PY" scripts/eval_healthbench.py \
            --model "$MODEL" ${wrap} ${lora} "${EVAL_BENCH[@]}" \
            --grader-model "$GRADER_MODEL" --sample-ids "$SAMPLE_IDS" --seed "$SEED" \
            --output "${EV}/${name}.json" >> "logs/eval_${BENCH}_${TAG}_s${SEED}.log" 2>&1
    done
    # epistemic 7-dim grade (benchmark-agnostic)
    CUDA_VISIBLE_DEVICES="$GPU" BODHI_VLLM_PORT="$((8000 + GPU))" "$INFER_PY" scripts/eval_epistemic.py \
        --response-files "${EV}/base_no_wrapper.json" "${EV}/base_bodhi.json" \
                         "${EV}/lora_no_wrapper.json" "${EV}/lora_bodhi.json" \
        --grader-model "$GRADER_MODEL" \
        --output "${EV}/epistemic_scores.json" >> "logs/eval_${BENCH}_${TAG}_s${SEED}.log" 2>&1
    echo "[seed $SEED] done"
}

seeds=($SEEDS); i=0
while [ $i -lt ${#seeds[@]} ]; do
    mapfile -t IDLE < <(idle_gpus)                        # re-check idle set each wave
    if [ ${#IDLE[@]} -eq 0 ]; then echo "no idle GPU; waiting 60s"; sleep 60; continue; fi
    pids=()
    for g in "${IDLE[@]}"; do
        [ $i -lt ${#seeds[@]} ] || break
        run_seed "${seeds[$i]}" "$g" & pids+=($!); i=$((i+1))
    done
    wait "${pids[@]}"                                     # barrier before next wave
done

# ---- Stage 5: aggregate across seeds --------------------------------------------
SEED_EVAL_DIRS=(); for s in $SEEDS; do SEED_EVAL_DIRS+=("${WORK}/seed_${s}/eval"); done
"$INFER_PY" scripts/aggregate_seeds.py --seed-dirs "${SEED_EVAL_DIRS[@]}" \
    ${BENCH_JSONL:+--healthbench "$BENCH_JSONL"} \
    --output "${WORK}/multi_seed_summary.json" || \
    echo "NOTE: aggregate_seeds may need a benchmark flag for non-HealthBench; see logs."

echo "=== cell done: ${WORK} ==="
