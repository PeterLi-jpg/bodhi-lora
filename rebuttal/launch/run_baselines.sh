#!/usr/bin/env bash
# run_baselines.sh: inference-time prompt baselines for one cell (EVAL-8).
#
# Runs each baseline of scripts/baselines/prompts.py on the unchanged base
# model: base_clarify (a clarifying-question system prompt) and
# base_verbalized_confidence (an uncertainty-expression prompt in the style of
# Lin et al. 2022 and Tian et al. 2023). It uses the cell's eval prompts, eval
# settings and graders (the same _common.sh as run_cell.sh), and grades the
# responses on the epistemic dimensions too. Outputs go to the cell's
# seed_<N>/eval/ next to the 2x2, so the baseline rows pair with the base and
# LoRA rows prompt by prompt.
#
# Usage:
#   MODEL=mistralai/Mistral-Small-24B-Instruct-2501 BENCH=healthbench \
#   SEEDS="42 7 13 99 101" bash rebuttal/launch/run_baselines.sh [--allow-same-family] [--dry-run]
#
# BASELINES picks a subset (default "clarify verbalized_confidence"). PROFILE,
# SEEDS and the eval settings mean the same as in run_cell.sh; set them the
# same way as for the cell. CONFIG is needed only with EVAL_BASE_QUANT=match.
# --dry-run runs everything with stub engines on the CPU.
set -euo pipefail
CONFIG="${CONFIG:-}"
BASELINES="${BASELINES:-clarify verbalized_confidence}"
source "$(dirname "$0")/_common.sh"

echo "=== baselines: $BENCH x $MODEL  ($BASELINES; seeds: $SEEDS, profile: $PROFILE)$([ "$DRY_RUN" = 1 ] && echo '  [DRY RUN]') ==="
if [ "$EVAL_BASE_QUANT" = match ] && [ -z "$CONFIG" ]; then
    echo "EVAL_BASE_QUANT=match needs CONFIG (the cell's training config)"; exit 2
fi

family_guard
build_eval_flags
prepare_data
build_settings

GPU="$(idle_gpus | head -1)"; : "${GPU:?no idle GPU for the baselines}"
for SEED in $SEEDS; do
    EV="${WORK}/seed_${SEED}/eval"
    IDS="$(eval_ids_for_seed "$SEED")"
    mkdir -p "$EV"
    outs=()
    for b in $BASELINES; do
        echo "[seed $SEED] baseline $b on GPU $GPU"
        run_gpu_py "$GPU" scripts/eval_healthbench.py \
            --model "$MODEL" --baseline "$b" "${EVAL_BENCH[@]}" "${EVAL_FLAGS[@]}" \
            --sample-ids "$IDS" --seed "$SEED" --output "${EV}/base_${b}.json" \
            >> "${LOG_DIR}/baselines_${BENCH}_${TAG}_s${SEED}.log" 2>&1
        if [ "$DRY_RUN" = 1 ]; then outs+=("${EV}/base_${b}.dryrun.json"); else outs+=("${EV}/base_${b}.json"); fi
    done
    run_gpu_py "$GPU" scripts/eval_epistemic.py \
        --response-files "${outs[@]}" "${EPI_DATA[@]}" "${EPI_FLAGS[@]}" \
        --output "${EV}/epistemic_scores_baselines.json" \
        >> "${LOG_DIR}/baselines_${BENCH}_${TAG}_s${SEED}.log" 2>&1
done

"$INFER_PY" scripts/run_manifest.py write --stage baselines \
    --out "${WORK}/baselines_manifest.json" \
    --model "base=${MODEL}${MODEL_REVISION:+@$MODEL_REVISION}" \
    --model "evaluator=${GRADER_MODEL}${GRADER_REVISION:+@$GRADER_REVISION}" \
    --holdout-ids "$HOLDOUT_IDS" "${SETTINGS_ARGS[@]}" --set "BASELINES=$BASELINES" \
    --include "family=${WORK}/family_check.json"
echo "=== baselines done: ${WORK}/seed_*/eval/base_{$(echo "$BASELINES" | tr ' ' ',')}.json ==="
