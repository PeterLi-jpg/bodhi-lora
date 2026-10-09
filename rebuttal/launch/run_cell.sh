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
#   bash rebuttal/launch/run_cell.sh [--allow-same-family] [--dry-run]
#
# BENCH in {healthbench, medqa, medquad, medicationqa, medmcqa, chatdoctor}.
#
# Settings live in _common.sh. The main ones:
#   PROFILE=submitted (default) | rebuttal-posted. submitted uses the eval
#     settings of the submitted MedGemma cell (max_model_len 4096, 32 scheduler
#     slots, meta-llama/Llama-3.1-8B-Instruct grader) and a separate random 200
#     eval prompts per seed from a held-out pool. rebuttal-posted reproduces the
#     settings the posted rebuttal cells ran with; legacy/README.md lists them.
#   PROTOCOL=bodhi (default) | ask_first. ask_first is the second CoT protocol
#     (bodhi_lora/protocols/ask_first.py); its cell goes to <cell>__ask_first.
#   TAU_RULE=fixed (default, tau = MIN_SCORE = 0.4) | percentile (tau =
#     TAU_PERCENTILE of the graded scores) | legacy-auto (the posted rule:
#     fixed, then the 22nd percentile when a non-HealthBench cell keeps < 40%).
#   EVAL_BASE_QUANT=bf16 (default) | match (serve the base in 4-bit when the
#     training config is QLoRA).
#
# --allow-same-family runs a cell whose base shares a family with the filter
# or the evaluator (configs/families.yaml); the manifest records the waiver.
# --dry-run (or DRY_RUN=1) needs no GPU and no HF token: it prints every GPU
# command, runs trace generation and eval with stub engines on the CPU, and
# writes the manifests.
#
# Every cell writes <cell>/manifest.json (settings, family check, filter counts
# and tau, overlap check, per-seed manifests), and each stage writes its own
# manifest next to its output (bodhi_lora/manifest.py).
set -euo pipefail
: "${CONFIG:?set CONFIG}"
source "$(dirname "$0")/_common.sh"

echo "=== cell: $BENCH x $MODEL  (seeds: $SEEDS, profile: $PROFILE, protocol: $PROTOCOL)$([ "$DRY_RUN" = 1 ] && echo '  [DRY RUN]') ==="

family_guard
build_eval_flags
prepare_data
build_settings
write_cell_manifest() {   # $1 = status
    local inc=(--include "family=${WORK}/family_check.json"
               --include "filter=${SFT}/manifest.json"
               --include "overlap=${WORK}/overlap_check.json")
    local s
    for s in $SEEDS; do inc+=(--include "seed_${s}=${WORK}/seed_${s}/manifest.json"); done
    "$INFER_PY" -m bodhi_lora.manifest write --stage cell --out "${WORK}/manifest.json" \
        --config "$CONFIG" --model "base=${MODEL}${MODEL_REVISION:+@$MODEL_REVISION}" \
        --model "filter=${FILTER_MODEL}${FILTER_REVISION:+@$FILTER_REVISION}" \
        --model "evaluator=${GRADER_MODEL}${GRADER_REVISION:+@$GRADER_REVISION}" \
        --holdout-ids "$HOLDOUT_IDS" --input "$BENCH_JSONL" \
        "${SETTINGS_ARGS[@]}" --set "STATUS=$1" \
        --set "TAU_LEGACY_TRIGGERED=${TAU_LEGACY_TRIGGERED:-0}" \
        --set "FIRST_PASS_KEPT=${FIRST_PASS_KEPT:-}" --set "FIRST_PASS_RAW=${FIRST_PASS_RAW:-}" \
        "${inc[@]}"
}
write_cell_manifest started

# ---- Stage 0b: contamination probe (HealthBench cells) -------------------------
# Does the base model complete HealthBench Hard prompts verbatim more often than
# PubMedQA text? One probe per base model; the JSON stays with the cell.
if [ "$BENCH" = "healthbench" ] && [ "$CONTAMINATION_PROBE" = 1 ] \
        && [ ! -s "${WORK}/contamination_probe.json" ]; then
    GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for the contamination probe}"
    run_gpu "$GPU" "$INFER_PY" -m bodhi_lora.pipeline.probe --model "$MODEL" \
        ${MODEL_REVISION:+--model-revision "$MODEL_REVISION"} \
        --healthbench-data "$BENCH_JSONL" --output "${WORK}/contamination_probe.json" \
        2>&1 | tee -a "${LOG_DIR}/probe_${BENCH}_${TAG}.log"
fi

# ---- Stage 1: generate teacher traces (ONCE per cell) ----------------------------
# Idempotent + resumable: if a partial raw_traces.jsonl exists (interrupted run),
# --resume-from continues from the prompt_ids already done instead of discarding
# them. generate_traces short-circuits without booting vLLM when it is already
# complete, so re-running this stage on a finished cell is free.
RAW="${SFT}/raw_traces.jsonl"
GEN_RESUME=()
[ -s "$RAW" ] && GEN_RESUME=(--resume-from "$RAW")
GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for generation}"
echo "[gen] GPU $GPU$([ -s "$RAW" ] && echo " (resuming $(wc -l < "$RAW") traces)")"
run_gpu_py "$GPU" -m bodhi_lora.pipeline.traces \
    --model "$MODEL" ${MODEL_REVISION:+--model-revision "$MODEL_REVISION"} \
    "${GEN_SRC[@]}" --protocol "$PROTOCOL" ${GEN_RESUME[@]+"${GEN_RESUME[@]}"} \
    --output "$RAW" --max-examples "$GEN_MAX" 2>&1 | tee -a "${LOG_DIR}/gen_${BENCH}_${TAG}.log"

# ---- Stage 2: grade + filter (grading ONCE per cell) -----------------------------
# tau flags for filter_traces.py. The value used is printed ("TAU: ...") and
# stored in <sft>/manifest.json.
tau_flags() {
    case "$TAU_RULE" in
        fixed|legacy-auto) echo "--min-score $MIN_SCORE" ;;
        percentile) echo "--tau-percentile $TAU_PERCENTILE" ;;
        *) echo "unknown TAU_RULE: $TAU_RULE" >&2; exit 2 ;;
    esac
}
FILTER_COMMON=(--input "$RAW" "${FILTER_DATA[@]}" --exclude-ids "${EXCLUDE[@]}"
               --grader-model "$FILTER_MODEL")
[ -n "$FILTER_REVISION" ] && FILTER_COMMON+=(--grader-revision "$FILTER_REVISION")
if [ ! -s "${SFT}/train.jsonl" ] || [ ! -s "${SFT}/val.jsonl" ]; then
    GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for filtering}"
    echo "[filter] GPU $GPU  tau rule: $TAU_RULE"
    # Reuse cached grades when re-filtering (rubric unchanged): grading 4k traces costs
    # ~50 min, and --resume-from carries existing grades forward so only the threshold
    # is re-applied. Delete graded.jsonl if the RUBRIC changed and you need a re-grade.
    FILTER_RESUME=()
    [ -s "${SFT}/graded.jsonl" ] && FILTER_RESUME=(--resume-from "${SFT}/graded.jsonl")
    # shellcheck disable=SC2046  # tau_flags is two words on purpose
    run_gpu "$GPU" "$INFER_PY" -m bodhi_lora.pipeline.filter "${FILTER_COMMON[@]}" \
        ${FILTER_RESUME[@]+"${FILTER_RESUME[@]}"} $(tau_flags) \
        --output-dir "$SFT" --graded-output "${SFT}/graded.jsonl" \
        2>&1 | tee "${LOG_DIR}/filter_${BENCH}_${TAG}.log"

    # legacy-auto (the posted cells): if a non-HealthBench cell kept < 40% of the
    # raw traces in train.jsonl at the fixed tau, re-filter at the 22nd-percentile
    # tau from the ALREADY-GRADED file. filter_traces.py does not start the grader
    # when nothing is left to grade, and it refuses a percentile of < 100 grades.
    if [ "$TAU_RULE" = legacy-auto ] && [ "$BENCH" != "healthbench" ] && [ "$DRY_RUN" != 1 ]; then
        FIRST_PASS_KEPT=$(wc -l < "${SFT}/train.jsonl" 2>/dev/null || echo 0)
        FIRST_PASS_RAW=$(wc -l < "$RAW" 2>/dev/null || echo 1)
        if [ "$FIRST_PASS_RAW" -gt 0 ] && [ "$((FIRST_PASS_KEPT * 100 / FIRST_PASS_RAW))" -lt 40 ]; then
            TAU_LEGACY_TRIGGERED=1
            echo "[filter] train $FIRST_PASS_KEPT/$FIRST_PASS_RAW < 40% at tau=$MIN_SCORE;" \
                 "re-filtering at the ${TAU_PERCENTILE}th-percentile tau (legacy-auto)"
            cp "${SFT}/manifest.json" "${SFT}/manifest_first_pass.json"
            "$INFER_PY" -m bodhi_lora.pipeline.filter "${FILTER_COMMON[@]}" \
                --resume-from "${SFT}/graded.jsonl" --tau-percentile "$TAU_PERCENTILE" \
                --output-dir "$SFT" --graded-output "${SFT}/graded.jsonl" \
                2>&1 | tail -12 | tee -a "${LOG_DIR}/filter_${BENCH}_${TAG}.log"
        fi
    fi
fi
[ -s "${SFT}/train.jsonl" ] && echo "train: $(wc -l < "${SFT}/train.jsonl")  val: $(wc -l < "${SFT}/val.jsonl")"

# Leakage gate (EVAL-12): no graded trace may be a held-out prompt, by id or by
# text. graded.jsonl holds every trace any seed can train on.
if [ -s "${SFT}/graded.jsonl" ]; then
    if [ "$BENCH" = "healthbench" ]; then
        "$INFER_PY" -m bodhi_lora.data.leakage \
            --healthbench "${DATA_RAW}/healthbench.jsonl" --healthbench-hard "$BENCH_JSONL" \
            --eval-ids "${DATA_RAW}/hard_200_sample_ids.json" \
            --train-jsonl "${SFT}/graded.jsonl" --json-out "${WORK}/overlap_check.json"
    else
        SEED_ID_FILES=()
        for s in $SEEDS; do SEED_ID_FILES+=("$(eval_ids_for_seed "$s")"); done
        "$INFER_PY" -m bodhi_lora.data.leakage --benchmark-jsonl "$BENCH_JSONL" \
            --holdout-ids "$HOLDOUT_IDS" --train-jsonl "${SFT}/graded.jsonl" \
            --eval-ids-files "${SEED_ID_FILES[@]}" --json-out "${WORK}/overlap_check.json"
    fi
fi

# Per-seed train/val split (SPLIT_POLICY=per-seed, as in the submitted cell,
# where the filter ran once per seed with --seed). Same graded traces and tau;
# only the 90/10 shuffle changes. CPU only: nothing is re-graded.
seed_sft_dir() {   # $1 = seed
    if [ "$SPLIT_POLICY" = per-seed ]; then echo "${WORK}/seed_$1/sft"; else echo "$SFT"; fi
}
if [ "$SPLIT_POLICY" = per-seed ]; then
    TAU_NOW="$(tau_flags)"
    if [ "${TAU_LEGACY_TRIGGERED:-0}" = 1 ]; then TAU_NOW="--tau-percentile $TAU_PERCENTILE"; fi
    for s in $SEEDS; do
        d="$(seed_sft_dir "$s")"
        if [ "$DRY_RUN" = 1 ] || [ ! -s "${SFT}/graded.jsonl" ]; then
            printf 'DRY-RUN [cpu]: split %s for seed %s\n' "${SFT}/graded.jsonl" "$s"
            continue
        fi
        [ -s "${d}/train.jsonl" ] && continue
        # shellcheck disable=SC2086  # TAU_NOW is two words on purpose
        "$INFER_PY" -m bodhi_lora.pipeline.filter "${FILTER_COMMON[@]}" \
            --resume-from "${SFT}/graded.jsonl" $TAU_NOW --seed "$s" \
            --output-dir "$d" > "${LOG_DIR}/split_${BENCH}_${TAG}_s${s}.log" 2>&1
    done
fi

# ---- Stages 3-4: per-seed train + 2x2 eval, parallelized across idle GPUs --------
if [ "$PROTOCOL" = bodhi ]; then WRAP_TAG=bodhi; else WRAP_TAG="$PROTOCOL"; fi
run_seed() {   # $1=seed  $2=gpu
    local SEED="$1" GPU="$2"
    local SD="${WORK}/seed_${SEED}" CK="${WORK}/seed_${SEED}/ckpt" EV="${WORK}/seed_${SEED}/eval"
    local LORA="${CK}/best" DATA_DIR IDS
    DATA_DIR="$(seed_sft_dir "$SEED")"
    IDS="$(eval_ids_for_seed "$SEED")"
    mkdir -p "$CK" "$EV"
    echo "[seed $SEED] train on GPU $GPU"
    run_gpu "$GPU" "$TRAIN_PY" -m bodhi_lora.pipeline.train \
        --config "$CONFIG" --seed "$SEED" --output-dir "$CK" \
        ${MODEL_REVISION:+--model-revision "$MODEL_REVISION"} \
        --train-file "${DATA_DIR}/train.jsonl" --val-file "${DATA_DIR}/val.jsonl" \
        > "${LOG_DIR}/train_${BENCH}_${TAG}_s${SEED}.log" 2>&1
    local name wrap lora outs=()
    for spec in "base_no_wrapper::" "base_${WRAP_TAG}:--wrapper ${PROTOCOL}:" \
                "lora_no_wrapper::--lora-path ${LORA}" "lora_${WRAP_TAG}:--wrapper ${PROTOCOL}:--lora-path ${LORA}"; do
        name="${spec%%:*}"
        local rest="${spec#*:}"
        wrap="${rest%%:*}"
        lora="${rest#*:}"
        # shellcheck disable=SC2086  # wrap and lora are flag pairs on purpose
        run_gpu_py "$GPU" -m bodhi_lora.pipeline.eval \
            --model "$MODEL" ${wrap} ${lora} "${EVAL_BENCH[@]}" "${EVAL_FLAGS[@]}" \
            --sample-ids "$IDS" --seed "$SEED" \
            --output "${EV}/${name}.json" >> "${LOG_DIR}/eval_${BENCH}_${TAG}_s${SEED}.log" 2>&1
        if [ "$DRY_RUN" = 1 ]; then outs+=("${EV}/${name}.dryrun.json"); else outs+=("${EV}/${name}.json"); fi
    done
    # epistemic 7-dim grade (benchmark-agnostic)
    run_gpu_py "$GPU" -m bodhi_lora.grading.epistemic \
        --response-files "${outs[@]}" "${EPI_DATA[@]}" "${EPI_FLAGS[@]}" \
        --output "${EV}/epistemic_scores.json" >> "${LOG_DIR}/eval_${BENCH}_${TAG}_s${SEED}.log" 2>&1
    "$INFER_PY" -m bodhi_lora.manifest write --stage seed --out "${SD}/manifest.json" \
        --config "$CONFIG" --model "base=${MODEL}${MODEL_REVISION:+@$MODEL_REVISION}" \
        --adapter "$LORA" --eval-ids "$IDS" --input "${DATA_DIR}/train.jsonl" \
        --input "${DATA_DIR}/val.jsonl" "${SETTINGS_ARGS[@]}" --set "SEED=$SEED" \
        --include "train=${CK}/manifest.json" --include "split=${DATA_DIR}/manifest.json" \
        > /dev/null
    echo "[seed $SEED] done"
}

seeds=($SEEDS); i=0
while [ $i -lt ${#seeds[@]} ]; do
    IDLE=($(idle_gpus))                                   # re-check idle set each wave
    if [ ${#IDLE[@]} -eq 0 ]; then echo "no idle GPU; waiting 60s"; sleep 60; continue; fi
    pids=()
    for g in "${IDLE[@]}"; do
        [ $i -lt ${#seeds[@]} ] || break
        run_seed "${seeds[$i]}" "$g" & pids+=($!); i=$((i+1))
        # A dry run fakes one GPU; run its seeds one after another.
        if [ "$DRY_RUN" = 1 ]; then wait "${pids[@]}"; pids=(); fi
    done
    wait ${pids[@]+"${pids[@]}"}                          # barrier before next wave
done

# ---- Stage 5: aggregate across seeds --------------------------------------------
SEED_EVAL_DIRS=(); for s in $SEEDS; do SEED_EVAL_DIRS+=("${WORK}/seed_${s}/eval"); done
if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY-RUN [cpu]: aggregate_seeds over %s\n' "${SEED_EVAL_DIRS[*]}"
elif [ "$PROTOCOL" = bodhi ]; then
    "$INFER_PY" scripts/aggregate_seeds.py --seed-dirs "${SEED_EVAL_DIRS[@]}" \
        "${AGG_DATA[@]}" \
        --output "${WORK}/multi_seed_summary.json" || \
        echo "NOTE: aggregate_seeds failed; per-seed eval json are intact (peek_results.py works)."
else
    # aggregate_seeds.py reads the four bodhi config names only.
    echo "NOTE: PROTOCOL=$PROTOCOL; aggregate the ${WRAP_TAG} configs separately."
fi

write_cell_manifest finished
echo "=== cell done: ${WORK} ==="
