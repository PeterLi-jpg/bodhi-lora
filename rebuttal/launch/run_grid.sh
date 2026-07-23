#!/usr/bin/env bash
# run_grid.sh — run the rebuttal generality grid, one cell after another.
#
# Each cell is run by run_cell.sh, which parallelizes its seeds across whatever
# GPUs are idle (and re-checks the idle set every wave, never touching a GPU
# another job is using). Cells run sequentially here because on a shared box the
# idle set is small; when more GPUs free up, within-cell seed parallelism uses
# them automatically. A failing cell is logged and skipped so it can't block the rest.
#
# Usage:  SEEDS="42 7 13 99 101" bash rebuttal/launch/run_grid.sh
#         CELLS="mistral_hb biomistral_hb" bash rebuttal/launch/run_grid.sh   # subset
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

export SEEDS="${SEEDS:-42 7 13 99 101}"
export GEN_MAX="${GEN_MAX:-4200}"

MED=configs/lora_medgemma27b_qlora.yaml
MIS=rebuttal/configs/lora_mistral_small_24b_qlora.yaml
BIO=rebuttal/configs/lora_biomistral7b.yaml

# cell key -> "MODEL|CONFIG|BENCH", in priority order (most rebuttal value first).
declare -A CELL=(
  [mistral_hb]="mistralai/Mistral-Small-24B-Instruct-2501|$MIS|healthbench"
  [biomistral_hb]="BioMistral/BioMistral-7B|$BIO|healthbench"
  [medgemma_medqa]="google/medgemma-27b-text-it|$MED|medqa"
  [medgemma_medquad]="google/medgemma-27b-text-it|$MED|medquad"
  [mistral_medqa]="mistralai/Mistral-Small-24B-Instruct-2501|$MIS|medqa"
  [mistral_medquad]="mistralai/Mistral-Small-24B-Instruct-2501|$MIS|medquad"
  [biomistral_medqa]="BioMistral/BioMistral-7B|$BIO|medqa"
  [biomistral_medquad]="BioMistral/BioMistral-7B|$BIO|medquad"
)
ORDER=(mistral_hb biomistral_hb medgemma_medqa medgemma_medquad \
       mistral_medqa mistral_medquad biomistral_medqa biomistral_medquad)

CELLS="${CELLS:-${ORDER[*]}}"

for key in $CELLS; do
  spec="${CELL[$key]:-}"
  if [ -z "$spec" ]; then echo "unknown cell: $key"; continue; fi
  IFS='|' read -r MODEL CONFIG BENCH <<< "$spec"
  echo "======================================================================"
  echo "  CELL $key : $BENCH x $MODEL"
  echo "======================================================================"
  MODEL="$MODEL" CONFIG="$CONFIG" BENCH="$BENCH" SEEDS="$SEEDS" GEN_MAX="$GEN_MAX" \
    bash rebuttal/launch/run_cell.sh 2>&1 | tee "logs/grid_${key}.log" \
    || echo "CELL $key FAILED (exit $?) — see logs/grid_${key}.log; continuing"
done
echo "=== grid done ==="
