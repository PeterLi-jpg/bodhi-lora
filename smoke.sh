#!/bin/bash
# Smoke test: run the full pipeline end-to-end on a tiny subset with a
# small local model. Meant to catch bugs (template/tokenizer/dtype/path)
# locally before burning slurm time on the 27B run.
#
# Prereqs:
#   - Python env with requirements installed (bash setup.sh)
#   - HF_TOKEN only if using a gated Gemma / MedGemma smoke model
#
# Runs in ~5-10 min on a single consumer GPU. CPU-only works but is slow.

set -euo pipefail

cd "$(dirname "$0")"

DEFAULT_GATED_MODEL="google/gemma-3n-E4B-it"

if [ -n "${SMOKE_MODEL:-}" ]; then
    MODEL="$SMOKE_MODEL"
else
    MODEL="$DEFAULT_GATED_MODEL"
fi

# small non-gated grader so smoke doesn't need a second gated access
GRADER="${SMOKE_GRADER:-Qwen/Qwen2.5-0.5B-Instruct}"
# override with: N_EXAMPLES=10 bash smoke.sh
N_EXAMPLES="${N_EXAMPLES:-3}"
RUNTIME_CONFIG="data/sft/smoke/runtime_train_config.yaml"

case "$MODEL" in
    google/gemma-*|google/medgemma-*)
        NEED_HF_TOKEN=1
        ;;
    *)
        NEED_HF_TOKEN=0
        ;;
esac

if [ "$NEED_HF_TOKEN" -eq 1 ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required for gated model $MODEL"
    echo "  1. Accept terms at https://huggingface.co/$MODEL"
    echo "  2. export HF_TOKEN=hf_..."
    echo "  3. rerun bash smoke.sh"
    echo
    echo "If you only want an ungated local wiring check, set for example:"
    echo "  SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash smoke.sh"
    exit 1
fi

echo "=== smoke test | $(date) ==="
echo "model:   $MODEL"
echo "grader:  $GRADER"
echo "samples: $N_EXAMPLES"
echo

mkdir -p data/sft/smoke eval/smoke logs

python - <<PY
import platform
from pathlib import Path

import yaml

cfg_path = Path("configs/lora_gemma_smoke.yaml")
out_path = Path("$RUNTIME_CONFIG")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["model"]["name"] = "$MODEL"

# Local Mac smoke is an engineering check, not the final training recipe.
# Float16 avoids MPS bf16 rough edges while keeping Linux slurm behavior intact.
if platform.system() == "Darwin":
    cfg["model"]["torch_dtype"] = "float16"
    cfg["training"]["bf16"] = False

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote runtime config -> {out_path}")
PY

echo "--- 0/4: preflight ---"
# Verify the env before we burn minutes; use smoke's own models for the check.
if [ "$NEED_HF_TOKEN" -eq 1 ]; then
    python scripts/preflight.py --models "$MODEL" "$GRADER"
else
    python scripts/preflight.py --models "$MODEL" "$GRADER" --skip-hf-access
fi

echo "--- 1/4: download data ---"
python scripts/download_data.py

echo "--- 2/4: generate $N_EXAMPLES BOHDI traces ---"
python scripts/generate_traces.py \
    --model "$MODEL" \
    --datasets healthbench_hard \
    --output data/sft/smoke/raw_traces.jsonl \
    --use-bodhi \
    --max-examples "$N_EXAMPLES"

echo "--- 3/4: grade and filter (threshold lowered so nothing is dropped) ---"
python scripts/filter_traces.py \
    --input data/sft/smoke/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl \
    --grader-model "$GRADER" \
    --output-dir data/sft/smoke \
    --min-score -999 \
    --val-ratio 0.34

echo "--- 4a/4: train 1 epoch on the smoke set ---"
# Smoke runs locally (Mac / single GPU / CPU).  We do NOT use `accelerate
# launch` here even on TPU: accelerate's tpu_launcher always forks
# addressable_device_count() processes via xmp.spawn, which is incompatible
# with our single-process SPMD design (see launch_multiseed.sh comments).
# Plain `python` works on Mac/CPU/GPU.  The TPU production path is its own
# launcher (tpu/launch_multiseed.sh).
python scripts/train_lora.py --config "$RUNTIME_CONFIG"

echo "--- 4b/4: eval on $N_EXAMPLES examples ---"
python scripts/eval_healthbench.py \
    --model "$MODEL" \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --grader-model "$GRADER" \
    --output eval/smoke/lora.json \
    --max-examples "$N_EXAMPLES"

if [ "$N_EXAMPLES" -ge 3 ]; then
    echo "--- 4c/4: U-shape stratification (tertiles on holdout only since n=$N_EXAMPLES) ---"
    python scripts/eval_ushape.py \
        --eval-jsons eval/smoke/lora.json \
        --healthbench data/raw/healthbench_hard.jsonl \
        --tertile-on-holdout-only \
        --output eval/smoke/ushape.json

    # Plots are thin with only 1 config + 3 samples, but we still run
    # plot_ushape as a wiring check so cluster-scale plotting isn't first-run
    # on the cluster.
    echo "--- 4d/4: render U-shape figures ---"
    python scripts/plot_ushape.py \
        --input eval/smoke/ushape.json \
        --eval-jsons eval/smoke/lora.json \
        --healthbench data/raw/healthbench_hard.jsonl \
        --n-bins 3 \
        --out-dir eval/smoke/figures
else
    echo "--- 4c/4: skipping U-shape + plots because N_EXAMPLES=$N_EXAMPLES < 3 ---"
fi

echo
echo "=== smoke test PASSED | $(date) ==="
echo "artifacts:"
echo "  data/sft/smoke/{train,val}.jsonl"
echo "  checkpoints/best/"
echo "  eval/smoke/lora.json"
if [ "$N_EXAMPLES" -ge 3 ]; then
    echo "  eval/smoke/ushape.json"
    echo "  eval/smoke/figures/{u_curve,u_fail,theme_fail}.png"
fi
