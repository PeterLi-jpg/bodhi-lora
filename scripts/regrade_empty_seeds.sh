#!/usr/bin/env bash
# Re-grade epistemic scores for seeds whose score file is missing or EMPTY.
#
# Seeds evaluated before the eval_epistemic prompt-lookup fix wrote score files
# with zero graded examples (all prompt_ids failed to resolve). The 4-config
# response files are intact, so this re-grades from them: no retraining, no
# re-inference of responses.
#
# Usage: regrade_empty_seeds.sh <cell_dir> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
W="${1:?cell dir}"; GPU="${2:?gpu}"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
# Must force subprocess: docker auto-detect pulls a TPU image that hangs here.
export BODHI_VLLM_MODE=subprocess BODHI_VLLM_ACCEL=gpu BODHI_VLLM_TP=1
GRADER="${GRADER_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
for s in 42 7 13 99 101; do
  EV="$W/seed_$s/eval"
  [ -f "$EV/base_no_wrapper.json" ] || continue
  [ -f "$EV/lora_bodhi.json" ] || continue          # skip seeds still mid-eval
  n=$(.venv-infer/bin/python -c "
import json,sys
try:
    d=json.load(open('$EV/epistemic_scores.json'))
    print(sum(len(c.get('examples') or []) for c in d['configs']))
except Exception: print(0)" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then echo "seed $s already good ($n)"; continue; fi
  echo "seed $s: regrading"
  rm -f "$EV/epistemic_scores.json"
  CUDA_VISIBLE_DEVICES="$GPU" BODHI_VLLM_PORT="$((8000 + GPU))" \
    .venv-infer/bin/python scripts/eval_epistemic.py \
      --response-files "$EV/base_no_wrapper.json" "$EV/base_bodhi.json" \
                       "$EV/lora_no_wrapper.json" "$EV/lora_bodhi.json" \
      --grader-model "$GRADER" --output "$EV/epistemic_scores.json" \
      >> "logs/regrade_$(basename $W).log" 2>&1 \
    && echo "seed $s OK" || echo "seed $s FAILED"
done
echo REGRADE_DONE
