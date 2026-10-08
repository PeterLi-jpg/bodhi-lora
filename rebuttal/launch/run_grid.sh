#!/usr/bin/env bash
# run_grid.sh — run the rebuttal generality grid, one cell after another.
#
# Each cell is run by run_cell.sh, which parallelizes its seeds across whatever
# GPUs are idle (and re-checks the idle set every wave, never touching a GPU
# another job is using). Cells run sequentially here because on a shared box the
# idle set is small; when more GPUs free up, within-cell seed parallelism uses
# them automatically. A failing cell is logged and skipped so it can't block the rest.
#
# Usage:  SEEDS="42 7 13 99 101" bash rebuttal/launch/run_grid.sh [--allow-same-family] [--dry-run]
#         CELLS="mistral_hb biomistral_hb" bash rebuttal/launch/run_grid.sh   # subset
#
# Arguments go to every run_cell.sh. The Med42 cells have a Llama-3 base, the
# evaluator's family, so the grader-family guard refuses them unless
# --allow-same-family is passed (each cell manifest then records the waiver).
# PROFILE, PROTOCOL and the other run_cell.sh settings pass through the env.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs

export SEEDS="${SEEDS:-42 7 13 99 101}"
export GEN_MAX="${GEN_MAX:-4200}"

MED=configs/lora_medgemma27b_qlora.yaml
MIS=rebuttal/configs/lora_mistral_small_24b_qlora.yaml
BIO=rebuttal/configs/lora_biomistral7b.yaml
M42=rebuttal/configs/lora_med42_8b.yaml
MISTRAL=mistralai/Mistral-Small-24B-Instruct-2501

# cell key -> "MODEL|CONFIG|BENCH". A case statement, not an associative array,
# so the script also runs under bash 3.2.
cell_spec() {
  case "$1" in
    mistral_hb)          echo "$MISTRAL|$MIS|healthbench" ;;
    med42_hb)            echo "m42-health/Llama3-Med42-8B|$M42|healthbench" ;;
    biomistral_hb)       echo "BioMistral/BioMistral-7B|$BIO|healthbench" ;;
    mistral_chatdoctor)  echo "$MISTRAL|$MIS|chatdoctor" ;;
    med42_chatdoctor)    echo "m42-health/Llama3-Med42-8B|$M42|chatdoctor" ;;
    mistral_medquad)     echo "$MISTRAL|$MIS|medquad" ;;
    mistral_medqa)       echo "$MISTRAL|$MIS|medqa" ;;
    medgemma_medqa)      echo "google/medgemma-27b-text-it|$MED|medqa" ;;
    medgemma_medquad)    echo "google/medgemma-27b-text-it|$MED|medquad" ;;
    biomistral_medqa)    echo "BioMistral/BioMistral-7B|$BIO|medqa" ;;
    biomistral_medquad)  echo "BioMistral/BioMistral-7B|$BIO|medquad" ;;
    *) return 1 ;;
  esac
}
# Default order: the six posted cells, then the set-aside MedQA cell, then
# cells that were planned but never posted.
ORDER="mistral_hb med42_hb biomistral_hb mistral_chatdoctor med42_chatdoctor mistral_medquad \
mistral_medqa medgemma_medqa medgemma_medquad biomistral_medqa biomistral_medquad"

CELLS="${CELLS:-$ORDER}"

for key in $CELLS; do
  spec="$(cell_spec "$key")" || { echo "unknown cell: $key"; continue; }
  IFS='|' read -r MODEL CONFIG BENCH <<< "$spec"
  echo "======================================================================"
  echo "  CELL $key : $BENCH x $MODEL"
  echo "======================================================================"
  MODEL="$MODEL" CONFIG="$CONFIG" BENCH="$BENCH" SEEDS="$SEEDS" GEN_MAX="$GEN_MAX" \
    bash rebuttal/launch/run_cell.sh "$@" 2>&1 | tee "logs/grid_${key}.log" \
    || echo "CELL $key FAILED (exit $?) — see logs/grid_${key}.log; continuing"
done
echo "=== grid done ==="
