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

# Optional Stage 4 eval cap.  Default empty = run on the full 200-prompt
# HealthBench Hard holdout.  Set EVAL_MAX=50 (or similar) for smoke runs to
# cut Stage 4 wall clock 4x.  Each config (4 per seed) honors the cap.
EVAL_MAX="${EVAL_MAX:-}"
_EVAL_MAX_FLAG=""
[ -n "$EVAL_MAX" ] && _EVAL_MAX_FLAG="--max-examples ${EVAL_MAX}"

# Model + train config — defaults to MedGemma-27B production setup.
# Swap for a smoke run with smaller models:
#   MODEL_NAME=google/gemma-3-4b-it TRAIN_CONFIG=configs/lora_gemma3_4b_tpu.yaml \
#     EVAL_MAX=50 SEEDS=42 MAX_EXAMPLES=200 bash tpu/launch_multiseed.sh
# Stage 1 (gen) and Stage 4 (eval) use MODEL_NAME; Stage 3 (train) uses
# TRAIN_CONFIG.  These must be consistent — TRAIN_CONFIG's model.name should
# equal MODEL_NAME or LoRA training won't be against the same base as gen/eval.
MODEL_NAME="${MODEL_NAME:-google/medgemma-27b-text-it}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lora_medgemma27b_tpu.yaml}"

# Stage 2 grader threshold and training floor.
# Lower both for smoke runs where the model is small/general (gemma-3-4b-it):
#   MIN_SCORE=0.0 TRAIN_FLOOR=1 bash tpu/launch_multiseed.sh
# Production default: 0.4 / 10.
MIN_SCORE="${MIN_SCORE:-0.4}"
TRAIN_FLOOR="${TRAIN_FLOOR:-10}"

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
TPU_RUNTIME="v2-alpha-tpuv6e"
ZONE="us-east1-d"  # overwritten once a slot is acquired

echo "Seeds to run: $SEEDS"
echo "TPU type will be shown once a slot is acquired"
echo ""

# TRC-granted slots — v6e-8 spot in TRC regions only.
# v6e-8 = 8 chips × 32 GB HBM = 256 GB; spot OK for TRC quota.
# Format: "TPU_TYPE ZONE SPOT(yes/no) _UNUSED_CFG"
# (_UNUSED_CFG was the accelerate config path; kept for readability only.)
TRC_SLOTS=(
    "v6e-8 us-east1-d     yes tpu/accelerate_config_v6e8.yaml"
    "v6e-8 europe-west4-a yes tpu/accelerate_config_v6e8.yaml"
)

# Try each slot in one pass, then sleep and retry the whole list.
# 200 rounds * 5 min = ~16 hours of overnight retrying. Override: MAX_ROUNDS=10 bash tpu/launch_multiseed.sh
MAX_ROUNDS="${MAX_ROUNDS:-200}"
RETRY_DELAY="${RETRY_DELAY:-300}"

echo "=== Acquiring TPU VM from TRC quota ==="
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
            # Filter by VM name so a neighbour VM (e.g. neural-operator-tpu) in
            # a different state doesn't cause us to spin forever after deletion.
            STATE=$(gcloud compute tpus tpu-vm list --zone="$_ZONE" --project="$PROJECT" \
                --format="value(name,state)" 2>/dev/null \
                | grep "^${TPU_NAME}\b" | awk '{print $2}')
            echo "  state: ${STATE:-(gone)}"
            # VM is unusable — fall through to fresh create.  PREEMPTED is the
            # spot-preemption end state; the VM still exists (gcloud lists it)
            # but cannot run our pipeline.  Without this branch, the loop hangs
            # forever waiting for READY.  HIDDEN/UNHIDING are GCP admin states
            # that can also stick — treat the same way.
            if [ -z "$STATE" ] || [ "$STATE" = "DELETING" ] \
               || [ "$STATE" = "PREEMPTED" ] \
               || [ "$STATE" = "HIDDEN" ] || [ "$STATE" = "UNHIDING" ]; then
                echo "VM $TPU_NAME unusable in $_ZONE (state=$STATE), will delete + try fresh create."
                # Delete the unusable VM ourselves so the create attempt
                # below doesn't hit ALREADY_EXISTS.
                gcloud compute tpus tpu-vm delete "$TPU_NAME" \
                    --zone="$_ZONE" --project="$PROJECT" --quiet 2>/dev/null || true
                break
            fi
        done
        [ "$STATE" != "READY" ] && continue
        ZONE="$_ZONE"
        echo "Reusing existing VM: $_TYPE in $_ZONE"
        found=true
        break
    fi
done

for round in $(seq 1 "$MAX_ROUNDS"); do
    $found && break
    for slot in "${TRC_SLOTS[@]}"; do
        read -r _TYPE _ZONE _SPOT _CFG <<< "$slot"
        SPOT_FLAG=""
        [ "$_SPOT" = "yes" ] && SPOT_FLAG="--spot"
        echo "  Trying $_TYPE ($(echo "$_TYPE" | grep -o '[0-9]*$') chips) in $_ZONE (spot=$_SPOT)..."
        if gcloud compute tpus tpu-vm create "$TPU_NAME" \
            --zone="$_ZONE" \
            --accelerator-type="$_TYPE" \
            --version="$TPU_RUNTIME" \
            --project="$PROJECT" \
            $SPOT_FLAG 2>&1; then
            ZONE="$_ZONE"
            echo "VM created: $_TYPE ($(echo "$_TYPE" | grep -o '[0-9]*$') chips) in $_ZONE (spot=$_SPOT)"
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

# Generic remote long-task runner — same nohup-launch + poll-from-launcher
# pattern that Stage 1 uses, but reusable for Stages 2/3/4.  Without this,
# every long stage hits the same SSH-teardown failure mode: the TPU host CPUs
# are pinned during XLA compile (10-40 min for a 27B SPMD model), the SSH
# connection drops with exit 255, and `set -e` aborts the launcher mid-stage.
#
# Args:
#   $1 — stage name (used in log filenames and progress messages)
#   $2 — pgrep pattern that matches the remote process (alive check)
#   $3 — remote shell command that completes the work and writes output.
#        Must produce a file at $4 on success.  Will be run via `nohup ... &`.
#   $4 — remote sentinel path (file/dir) that exists when the task is done.
#        Polling keeps going until this exists OR the process exits.
#   $5 — optional max poll iterations (default 576 = 48h at 5-min intervals).
#
# On done: returns 0.  On 6 consecutive SSH timeouts or a remote crash with
# missing sentinel: prints the tail of the remote log and returns 1.
run_long_remote() {
    local _name="$1" _pgrep_pat="$2" _remote_cmd="$3" _sentinel="$4"
    local _max_iters="${5:-576}"
    local _log="/tmp/${_name}.log" _pid_file="/tmp/${_name}.pid"

    echo "=== ${_name}: launching on TPU ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
export HF_TOKEN='${HF_TOKEN}'
rm -f ${_log}
nohup bash -c '${_remote_cmd}' > ${_log} 2>&1 &
echo \$! > ${_pid_file}
echo '${_name} running in background (PID '\$(cat ${_pid_file})')'
" || true  # SSH teardown often returns 255 right after launch (CPUs pinned).

    echo "Polling ${_name}..."
    local _ssh_misses=0
    for i in $(seq 1 "$_max_iters"); do
        sleep 300
        local _done _alive
        # || true: grep exits 1 when the TPU is gone (no output to match);
        # without this, set -euo pipefail silently kills the launcher mid-$()
        # with no error message, skipping the SSH-timeout counter entirely.
        _done=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" \
            --command="[ -e ${_sentinel} ] && echo done || echo pending" 2>/dev/null \
            | grep -E '^(done|pending)$' | tail -1) || true
        if [ "$_done" = "done" ]; then
            echo "${_name}: complete (sentinel ${_sentinel} present)."
            return 0
        fi
        _alive=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" \
            --command="pgrep -f '${_pgrep_pat}' > /dev/null 2>&1 && echo alive || echo dead" 2>/dev/null \
            | grep -E '^(alive|dead)$' | tail -1) || true
        if [ -z "$_alive" ] && [ -z "$_done" ]; then
            _ssh_misses=$(( _ssh_misses + 1 ))
            echo "  ${_name} poll $i: SSH timeout (miss $_ssh_misses/6)"
            if [ "$_ssh_misses" -ge 6 ]; then
                echo "ERROR: ${_name}: 6 consecutive SSH timeouts."
                echo "--- last 40 lines of ${_log} ---"
                gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                    --zone="$ZONE" --project="$PROJECT" \
                    --command="tail -40 ${_log} 2>/dev/null || echo '(log not found)'" \
                    2>/dev/null || true
                echo "--- end log ---"
                return 1
            fi
            continue
        fi
        _ssh_misses=0
        echo "  ${_name} poll $i: ${_alive:-unknown}"
        if [ "$_alive" = "dead" ]; then
            # Process exited — recheck sentinel one more time before declaring failure
            # (race: process finished between our two SSH calls).
            _done=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --command="[ -e ${_sentinel} ] && echo done || echo pending" 2>/dev/null \
                | grep -E '^(done|pending)$' | tail -1) || true
            if [ "$_done" = "done" ]; then
                echo "${_name}: complete (sentinel found after process exit)."
                return 0
            fi
            echo "ERROR: ${_name} exited but sentinel ${_sentinel} is missing."
            echo "--- last 40 lines of ${_log} ---"
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --command="tail -40 ${_log} 2>/dev/null || echo '(log not found)'" \
                2>/dev/null || true
            echo "--- end log ---"
            return 1
        fi
    done
    echo "ERROR: ${_name}: exceeded max poll iterations."
    return 1
}

# On exit (normal, error, or Ctrl-C): rescue any unsaved checkpoints first,
# then delete the VM. Completed seeds are already local; this catches
# whatever is on the VM for any in-progress seed that was interrupted.
trap '
    echo ""
    echo "=== Saving any remaining data before VM deletion ==="
    mkdir -p "./results/_rescue/sft"
    # Raw traces are expensive to regenerate (~4-8 hours); save them so
    # the next run can resume via --resume-from rather than starting over.
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/raw_traces.jsonl" "./results/_rescue/" 2>/dev/null \
        && echo "Rescued raw_traces.jsonl" \
        || echo "(no raw_traces.jsonl to rescue)"
    # SFT data (graded + filtered) — grading takes 15-30 min so worth saving.
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/train.jsonl" "./results/_rescue/sft/" 2>/dev/null \
        && echo "Rescued sft/train.jsonl" \
        || echo "(no sft/train.jsonl to rescue)"
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/val.jsonl" "./results/_rescue/sft/" 2>/dev/null \
        && echo "Rescued sft/val.jsonl" \
        || echo "(no sft/val.jsonl to rescue)"
    gcloud compute tpus tpu-vm scp \
        --recurse \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/checkpoints" "./results/_rescue/" 2>/dev/null \
        && echo "Rescue checkpoints done" \
        || echo "(no checkpoints to rescue)"
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
# Pre-download HealthBench datasets so filter_traces.py and eval_healthbench.py
# can look up rubrics on the first run without hitting a FileNotFoundError.
python3 -c \"
import urllib.request, pathlib
files = {
    'data/raw/healthbench_hard.jsonl': 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl',
    'data/raw/healthbench.jsonl':      'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl',
}
for path, url in files.items():
    p = pathlib.Path(path)
    if not p.exists():
        print(f'Downloading {path}...')
        urllib.request.urlretrieve(url, p)
    print(f'  {path}: {p.stat().st_size} bytes')
\"
"

# ── Stage 1: generate traces (separate SSH so a failure here is identifiable
#    and restartable without re-running setup) ─────────────────────────────────
echo "=== Stage 1: generate BOHDI traces (MedGemma-27B) ==="

# If a previous VM was interrupted mid-generation, restore the partial file
# so --resume-from can skip already-done examples.
if [ -n "$GCS_DATA_PATH" ]; then
    echo "Checking GCS for partial traces from previous run..."
    mkdir -p "./results/_rescue"
    gsutil cp "${GCS_DATA_PATH}/raw_traces.jsonl" "./results/_rescue/raw_traces.jsonl" 2>/dev/null || true
fi

if [ -f "./results/_rescue/raw_traces.jsonl" ]; then
    echo "Restoring partial traces from previous run..."
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "./results/_rescue/raw_traces.jsonl" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/raw_traces.jsonl" \
        && echo "  Restored $(wc -l < ./results/_rescue/raw_traces.jsonl) traces" \
        || echo "  Restore failed — starting from scratch"
fi

# Stage 1 full run: nohup + polling so SSH drops during the 4-8 hour
# generation don't kill the process.  Poll every 5 min.
#
# Root cause of previous "hang at 0/1": XLA compiles the 27B SPMD graph on the
# first forward pass, pinning all host CPUs for 20-40 min and dropping SSH.
# The process was never dead — it just looked that way from a blocking SSH.
# Fix: nohup so the process survives SSH drops.
#
# NOTE: do NOT set XLA_FLAGS here.  The TPU runtime already sets
# --xla_gpu_force_compilation_parallelism=8 in the env; appending any
# unrecognised flag (like --xla_persistent_cache_dir) causes a FATAL
# "Unknown flags in XLA_FLAGS" crash before the first forward pass.
# Hard-only training set. HealthBench Hard has 1000 prompts; we hold out 200
# for evaluation (data/raw/hard_200_sample_ids.json), leaving 800 unique
# prompts for trace generation. --exclude-ids prevents the 200 eval prompts
# from leaking into training. Default cap of 800 = "all available trainable".
_MAX=${MAX_EXAMPLES:-800}
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd ~/bohdi-lora
export HF_TOKEN='${HF_TOKEN}'
rm -f /tmp/gen_stage1.log
nohup python scripts/generate_traces.py \
    --model ${MODEL_NAME} \
    --datasets healthbench_hard \
    --exclude-ids data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl \
    --resume-from data/sft/raw_traces.jsonl \
    --use-bodhi \
    --max-examples ${_MAX} \
    > /tmp/gen_stage1.log 2>&1 &
echo \$! > /tmp/gen_stage1.pid
echo 'Stage 1 running in background (PID '\$(cat /tmp/gen_stage1.pid)')'
" || true  # XLA model-load pins all TPU CPUs; SSH teardown may timeout (exit 255)
           # — the nohup process is running; the polling loop handles failure detection.

TARGET="${_MAX}"
echo "Polling Stage 1 progress (target: ${TARGET} traces)..."
_ssh_misses=0   # consecutive polls where SSH timed out (returned nothing)
for i in $(seq 1 576); do   # 576 × 5 min = 48 hours max
    sleep 300
    N=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="wc -l < ~/bohdi-lora/data/sft/raw_traces.jsonl 2>/dev/null || echo 0" 2>/dev/null \
        | grep -E '^[0-9]+$' | tail -1) || true
    echo "  Stage 1 poll $i: ${N:-0}/$TARGET traces"
    if [ "${N:-0}" -gt 0 ]; then
        # Back up to local so we don't lose data on preemption
        mkdir -p "./results/_rescue"
        gcloud compute tpus tpu-vm scp --zone="$ZONE" --project="$PROJECT" "${TPU_NAME}:~/bohdi-lora/data/sft/raw_traces.jsonl" "./results/_rescue/raw_traces.jsonl" 2>/dev/null || true
        # Back up to GCS if configured
        if [ -n "$GCS_DATA_PATH" ]; then
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --command="gsutil cp ~/bohdi-lora/data/sft/raw_traces.jsonl ${GCS_DATA_PATH}/raw_traces.jsonl" 2>/dev/null || true
        fi
    fi
    if [ "${N:-0}" -ge "$TARGET" ]; then
        echo "Stage 1 complete: $TARGET traces written."
        _ssh_misses=0
        break
    fi
    ALIVE=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="pgrep -f '[g]enerate_traces.py' > /dev/null 2>&1 && echo alive || echo dead" 2>/dev/null \
        | grep -E '^(alive|dead)$' | tail -1) || true
    if [ -z "$ALIVE" ]; then
        # SSH timed out — TPU CPUs may be pinned during XLA compilation.
        # Allow up to 6 consecutive misses (~30 min) before giving up.
        _ssh_misses=$(( _ssh_misses + 1 ))
        echo "  (SSH timeout on alive-check, miss $_ssh_misses/6)"
        if [ "$_ssh_misses" -ge 6 ]; then
            echo "ERROR: 6 consecutive SSH timeouts — process presumed dead."
            echo "--- last 40 lines of gen_stage1.log ---"
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --command="tail -40 /tmp/gen_stage1.log 2>/dev/null || echo '(log not found)'" \
                2>/dev/null || true
            echo "--- end log ---"
            exit 1
        fi
        continue
    fi
    _ssh_misses=0
    if [ "$ALIVE" = "dead" ]; then
        echo "Stage 1 process exited with ${N:-0}/$TARGET traces."
        # If we got 0 traces, something crashed early — print the log so we
        # can diagnose the error without SSH-ing in manually.
        if [ "${N:-0}" -eq 0 ]; then
            echo "--- last 40 lines of gen_stage1.log ---"
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --command="tail -40 /tmp/gen_stage1.log 2>/dev/null || echo '(log not found)'" \
                2>/dev/null || true
            echo "--- end log ---"
            echo "ERROR: Stage 1 produced 0 traces — aborting pipeline."
            exit 1
        fi
        break
    fi
done
echo "Generate done: $(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="wc -l < ~/bohdi-lora/data/sft/raw_traces.jsonl 2>/dev/null || echo 0" 2>/dev/null \
    | grep -E '^[0-9]+$' | tail -1) traces"

# Save traces locally so they survive if the VM is preempted before training.
echo "Saving raw_traces.jsonl to local results/..."
mkdir -p "./results"
gcloud compute tpus tpu-vm scp \
    --zone="$ZONE" --project="$PROJECT" \
    "${TPU_NAME}:~/bohdi-lora/data/sft/raw_traces.jsonl" "./results/" \
    && echo "  Saved $(wc -l < ./results/raw_traces.jsonl) traces" \
    || echo "  Save failed (continuing)"
# Also update the rescue copy so the restore path is consistent.
cp -f "./results/raw_traces.jsonl" "./results/_rescue/raw_traces.jsonl" 2>/dev/null || true

# ── Stage 2: grade + filter (Qwen2.5-14B grader, long-running on TPU) ────────
# Stage 2 runs a 14B grader over all (trace, rubric_item) pairs.
# Each grade is a generate() call — same XLA compile / SSH-teardown failure
# mode as Stage 1 if we run it in a foreground SSH.  Use the nohup-polling
# helper instead.  Sentinel: data/sft/train.jsonl is written at the very end.
#
# Preemption resilience: if train.jsonl was saved from a previous run, restore
# it directly and skip grading entirely.
if [ -n "$GCS_DATA_PATH" ]; then
    echo "Checking GCS for graded SFT data from previous run..."
    mkdir -p "./results/_rescue/sft"
    gsutil cp "${GCS_DATA_PATH}/sft/train.jsonl" "./results/_rescue/sft/train.jsonl" 2>/dev/null || true
    gsutil cp "${GCS_DATA_PATH}/sft/val.jsonl" "./results/_rescue/sft/val.jsonl" 2>/dev/null || true
fi

if [ -f "./results/_rescue/sft/train.jsonl" ]; then
    echo "=== Stage 2: restoring graded SFT data from previous run (skip re-grading) ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="mkdir -p ~/bohdi-lora/data/sft" 2>/dev/null || true
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "./results/_rescue/sft/train.jsonl" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/train.jsonl" \
        && echo "  Restored train.jsonl" || echo "  Restore failed — will re-grade"
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "./results/_rescue/sft/val.jsonl" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/val.jsonl" 2>/dev/null || true
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="touch /tmp/stage2_done" 2>/dev/null || true
else
    run_long_remote \
        "stage2_grade" \
        "[f]ilter_traces.py" \
        "python scripts/filter_traces.py --input data/sft/raw_traces.jsonl --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl --output-dir data/sft --min-score ${MIN_SCORE} && touch /tmp/stage2_done" \
        "/tmp/stage2_done"
    # Save SFT data to the runner immediately — survives TPU preemption.
    echo "Saving Stage 2 SFT data to local results/_rescue/sft/..."
    mkdir -p "./results/_rescue/sft"
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/train.jsonl" "./results/_rescue/sft/" \
        && echo "  Saved train.jsonl" || echo "  Save failed (continuing)"
    gcloud compute tpus tpu-vm scp \
        --zone="$ZONE" --project="$PROJECT" \
        "${TPU_NAME}:~/bohdi-lora/data/sft/val.jsonl" "./results/_rescue/sft/" 2>/dev/null || true

    if [ -n "$GCS_DATA_PATH" ]; then
        echo "Saving Stage 2 SFT data to GCS..."
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --command="gsutil cp ~/bohdi-lora/data/sft/train.jsonl ${GCS_DATA_PATH}/sft/train.jsonl && gsutil cp ~/bohdi-lora/data/sft/val.jsonl ${GCS_DATA_PATH}/sft/val.jsonl" 2>/dev/null || true
    fi
fi
_TRAIN_LINES=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="wc -l < ~/bohdi-lora/data/sft/train.jsonl 2>/dev/null || echo 0" 2>/dev/null \
    | grep -E '^[0-9]+$' | tail -1)
echo "Training data ready: ${_TRAIN_LINES:-0} train"
# Stage 3 needs at least a handful of training examples — and load_best_model
# requires a non-empty eval set.  10 is a defensible floor; below that the
# whole experiment is meaningless and we'd just waste another v6e-8-hour OOMing
# or NaN-ing.  Surface the failure HERE rather than from a confusing stack
# trace inside SFTTrainer or DataCollatorForCompletionOnlyLM.
if [ "${_TRAIN_LINES:-0}" -lt "${TRAIN_FLOOR}" ]; then
    echo "ERROR: Stage 2 produced only ${_TRAIN_LINES:-0} training examples (< ${TRAIN_FLOOR} floor)."
    echo "Likely causes: grader threshold too high, grader produced 0% positive scores,"
    echo "or grader silently failed.  Inspect /tmp/stage2_grade.log on the VM."
    echo "For smoke runs with a small model, pass MIN_SCORE=0.0 TRAIN_FLOOR=1."
    exit 1
fi

# ── Train each seed sequentially ──────────────────────────────────────────────
for SEED in $SEEDS; do
    echo ""
    echo "=== Stage 3: training seed $SEED ==="
    # NOTE: do NOT use 'accelerate launch'.  accelerate's tpu_launcher
    # ALWAYS calls xmp.spawn() with no nprocs arg, which forks
    # addressable_device_count() processes (= 8 on v6e-8).  Each forked
    # process loads a full 54 GB MedGemma-27B copy on its chip → 8x OOM.
    # That is incompatible with our single-process SPMD design (where ONE
    # python process drives all 8 chips and we shard the model with
    # mark_sharding).  Run the script directly with PJRT_DEVICE=TPU so
    # HF Trainer's native XLA path picks up SPMD without xmp.spawn.
    # Sentinel: <output-dir>/best/adapter_model.safetensors is the final
    # artifact written by trainer.save_model() at the end of training.
    run_long_remote \
        "stage3_train_seed${SEED}" \
        "[t]rain_lora.py" \
        "mkdir -p checkpoints/seed_${SEED} && PJRT_DEVICE=TPU python -u scripts/train_lora.py --config ${TRAIN_CONFIG} --seed ${SEED} --output-dir checkpoints/seed_${SEED} ${TRAIN_EXTRA_FLAGS}" \
        "checkpoints/seed_${SEED}/best/adapter_model.safetensors"

    # ── Stage 4: evaluate all 4 configurations ────────────────────────────────
    # Same long-running pattern as the other stages.  Sentinel: rubric_diff.json
    # is the very last artifact produced; if it exists, the whole eval finished.
    # NOTE: the heredoc-style remote command is condensed onto one big line because
    # run_long_remote wraps it in `bash -c '<cmd>'` — embedded newlines in a
    # single-quoted bash -c break the parser.
    EVAL_DIR="eval/seed_${SEED}"
    FIG_DIR="figures/seed_${SEED}"
    LORA="checkpoints/seed_${SEED}/best"
    MODEL="${MODEL_NAME}"
    IDS="data/raw/hard_200_sample_ids.json"
    HB="data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl"
    EVAL_CMD="set -e; mkdir -p ${EVAL_DIR} ${FIG_DIR}"
    EVAL_CMD="${EVAL_CMD} && python scripts/eval_healthbench.py --model ${MODEL} --sample-ids ${IDS} ${_EVAL_MAX_FLAG} --output ${EVAL_DIR}/base_no_wrapper.json"
    EVAL_CMD="${EVAL_CMD} && python scripts/eval_healthbench.py --model ${MODEL} --use-bodhi --sample-ids ${IDS} ${_EVAL_MAX_FLAG} --output ${EVAL_DIR}/base_bodhi.json"
    EVAL_CMD="${EVAL_CMD} && python scripts/eval_healthbench.py --model ${MODEL} --lora-path ${LORA} --sample-ids ${IDS} ${_EVAL_MAX_FLAG} --output ${EVAL_DIR}/lora_no_wrapper.json"
    EVAL_CMD="${EVAL_CMD} && python scripts/eval_healthbench.py --model ${MODEL} --lora-path ${LORA} --use-bodhi --sample-ids ${IDS} ${_EVAL_MAX_FLAG} --output ${EVAL_DIR}/lora_bodhi.json"
    EVAL_CMD="${EVAL_CMD} && python scripts/eval_ushape.py --eval-jsons ${EVAL_DIR}/base_no_wrapper.json ${EVAL_DIR}/base_bodhi.json ${EVAL_DIR}/lora_no_wrapper.json ${EVAL_DIR}/lora_bodhi.json --healthbench ${HB} --output ${EVAL_DIR}/ushape.json"
    EVAL_CMD="${EVAL_CMD} && python scripts/plot_ushape.py --input ${EVAL_DIR}/ushape.json --eval-jsons ${EVAL_DIR}/base_no_wrapper.json ${EVAL_DIR}/base_bodhi.json ${EVAL_DIR}/lora_no_wrapper.json ${EVAL_DIR}/lora_bodhi.json --healthbench ${HB} --n-bins 10 --out-dir ${FIG_DIR}"
    EVAL_CMD="${EVAL_CMD} && (if [ -f ${LORA}/trainer_state.json ]; then python scripts/plot_training.py --trainer-state ${LORA}/trainer_state.json --output ${FIG_DIR}/training_loss.png; fi)"
    EVAL_CMD="${EVAL_CMD} && python scripts/rubric_diff.py ${EVAL_DIR}/base_no_wrapper.json ${EVAL_DIR}/lora_bodhi.json --output ${EVAL_DIR}/rubric_diff.json"

    run_long_remote \
        "stage4_eval_seed${SEED}" \
        "[e]val_healthbench.py|[e]val_ushape.py|[p]lot_ushape.py|[r]ubric_diff.py|[p]lot_training.py" \
        "${EVAL_CMD}" \
        "${EVAL_DIR}/rubric_diff.json"

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

# ── Aggregate across all seeds (requires N >= 2) ──────────────────────────────
# aggregate_seeds.py computes cross-seed means + 95% CIs and needs at least 2
# seed dirs.  Single-seed smoke runs skip this step rather than crashing.
_N_SEEDS=$(echo "$SEEDS" | wc -w | tr -d ' ')
echo ""
if [ "$_N_SEEDS" -ge 2 ]; then
    echo "=== Aggregate: multi-seed summary ($SEEDS) ==="
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
    echo "Multi-seed summary: ./results/multi_seed_summary.json"
else
    echo "Single-seed run — skipping multi-seed aggregation (requires >= 2 seeds)."
fi

echo ""
echo "All seeds done: $SEEDS"
echo "Results in ./results/"
