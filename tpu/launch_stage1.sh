#!/bin/bash
# launch_stage1.sh - spin up ONE v6e-8 spot VM, run Stage 1 only
# (BODHI trace generation), upload raw_traces.jsonl to GCS, delete the
# VM. Cheap pre-step before launch_5seeds.sh: once raw_traces.jsonl
# lives in GCS, the 5 follower VMs in launch_5seeds.sh skip Stage 1
# (~5h x 4 = 20 chip-hours saved per multi-seed launch).
#
# Why a separate script instead of "just run launch_5seeds with one
# seed": launch_5seeds requires exactly 5 seeds (it splits across
# 2 zones for the across-seed CI band the paper needs). For Stage 1
# alone we want a focused, self-deleting launcher with simpler retry
# logic.
#
# Usage:
#   export HF_TOKEN=...                                   # in .env is fine
#   GCS_OUTPUT_PATH=gs://bohdi-runs-tokyo-micron/ bash tpu/launch_stage1.sh
#
# What lands in GCS on success:
#   ${GCS_OUTPUT_PATH%/}/raw_traces.jsonl
#
# To verify after this script returns:
#   gsutil ls -l ${GCS_OUTPUT_PATH%/}/raw_traces.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE=""
_d="$SCRIPT_DIR"
for _ in 1 2 3 4 5 6; do
    _d="$(dirname "$_d")"
    if [ -f "${_d}/.env" ]; then ENV_FILE="${_d}/.env"; break; fi
done
if [ -n "$ENV_FILE" ] && [ -z "${HF_TOKEN:-}" ]; then
    # shellcheck source=/dev/null
    set -a; source "$ENV_FILE"; set +a
fi
: "${HF_TOKEN:?HF_TOKEN not set: add HF_TOKEN=hf_... to .env or export it}"
: "${GCS_OUTPUT_PATH:?GCS_OUTPUT_PATH not set: required so raw_traces.jsonl survives VM deletion}"

GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
: "${GH_TOKEN:?GH_TOKEN not set and gh auth token returned empty: auth gh or export GH_TOKEN}"

PROJECT="tokyo-micron-494016-s9"
ZONE="europe-west4-a"
VM_NAME="bohdi-stage1"
RUNTIME="v2-alpha-tpuv6e"
CREATE_RETRIES="${CREATE_RETRIES:-200}"
MAX_PREEMPT_RETRIES="${MAX_PREEMPT_RETRIES:-10}"
LOG="${SCRIPT_DIR}/../results/stage1_launch.log"
mkdir -p "$(dirname "$LOG")"

GCS_BASE="${GCS_OUTPUT_PATH%/}"

echo "=== launch_stage1: 1x v6e-8 spot ($ZONE), Stage 1 only ==="
echo "  GCS_OUTPUT_PATH: $GCS_OUTPUT_PATH"
echo "  raw_traces.jsonl will land at: ${GCS_BASE}/raw_traces.jsonl"
echo "  log: $LOG"
echo

log() { printf '[%s %s] %s\n' "$(date -u +%H:%M:%S)" "$VM_NAME" "$*" | tee -a "$LOG"; }

try_create() {
    local attempt=0
    until gcloud compute tpus tpu-vm create "$VM_NAME" \
        --zone="$ZONE" \
        --accelerator-type="v6e-8" \
        --version="$RUNTIME" \
        --project="$PROJECT" \
        --spot >>"$LOG" 2>&1; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$CREATE_RETRIES" ]; then
            log "create gave up after $attempt attempts"
            return 1
        fi
        log "create error, retry $attempt/$CREATE_RETRIES in 60s..."
        sleep 60
    done
    log "VM created"
    return 0
}

push_tokens() {
    printf '%s\n%s\n' "$GH_TOKEN" "$HF_TOKEN" \
        | gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
            --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
            --command='read -r G; read -r H; umask 077; { echo "GH_TOKEN=$G"; echo "HF_TOKEN=$H"; } > ~/.bohdi-env; chmod 600 ~/.bohdi-env' \
            >>"$LOG" 2>&1
}

build_remote_cmd() {
    cat <<REMOTE
set -euo pipefail
if [ -f ~/.bohdi-env ]; then
    set -a; source ~/.bohdi-env; set +a
fi
: "\${HF_TOKEN:?HF_TOKEN missing from ~/.bohdi-env}"
: "\${GH_TOKEN:?GH_TOKEN missing from ~/.bohdi-env}"
export PJRT_DEVICE=TPU
# Pin to the py3.11 venv that setup_tpu.sh installs, so PATH ordering
# can't drop us back onto system python 3.10.
PY=~/.venv-py311/bin/python

if [ ! -d ~/bohdi-lora ]; then
    git -c "url.https://x-access-token:\${GH_TOKEN}@github.com/.insteadOf=https://github.com/" \\
        clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
fi
cd ~/bohdi-lora
git -c "url.https://x-access-token:\${GH_TOKEN}@github.com/.insteadOf=https://github.com/" \\
    fetch origin main 2>&1 | tail -2 || true
git reset --hard origin/main 2>&1 | tail -1 || true

echo "--- 0/1 setup_tpu.sh ---" | tee -a ~/pipeline.log
bash tpu/setup_tpu.sh > ~/setup.log 2>&1 || { echo "setup FAILED" >> ~/pipeline.log; exit 1; }
echo SETUP_OK >> ~/pipeline.log

mkdir -p data/sft data/raw

# Resume base: if raw_traces.jsonl already exists in GCS (e.g. a partial
# run from a prior preempted attempt), pull it down so generate_traces.py
# can resume against it. The generate script self-skips when nothing is
# left to do, so this path doubles as the "Stage 1 already complete"
# fast-exit. --exclude-ids drops *all 1000* HealthBench Hard prompts from
# the trace pool (not just the 200-sample eval holdout) so the per-seed
# bootstrap eval (issue #60) is honestly held-out.  HealthBench Hard is a
# strict subset of HealthBench Full, so passing the .jsonl directly
# excludes every Hard prompt; we also pass the 200-sample file
# explicitly as belt-and-suspenders / for documentation.
if gsutil -q stat "${GCS_BASE}/raw_traces.jsonl" 2>/dev/null; then
    echo "--- pulling resume base from ${GCS_BASE}/raw_traces.jsonl ---" | tee -a ~/pipeline.log
    gsutil -q cp "${GCS_BASE}/raw_traces.jsonl" data/sft/raw_traces.jsonl
    echo "  resume rows: \$(wc -l < data/sft/raw_traces.jsonl 2>/dev/null || echo 0)" >> ~/pipeline.log
fi

echo "--- 1/1 generate BODHI traces (with resume + eval-id exclusion) ---" | tee -a ~/pipeline.log
\${PY} -u scripts/download_data.py >> ~/pipeline.log 2>&1
# --resume-from points at the same path as --output; if generate_traces
# finds it, it skips done prompt_ids and appends.  --exclude-ids drops
# all 1000 HealthBench Hard prompts so the SFT corpus has zero overlap
# with the per-seed bootstrap eval (issue #60).
# --force-resume tolerates rescue files that pre-date the
# ablate_component metadata field (issue #69 added it; older rows
# omit it, which would otherwise trigger the resume-config-mismatch
# guard).
\${PY} -u scripts/generate_traces.py \\
    --model google/medgemma-27b-text-it \\
    --datasets healthbench_hard healthbench \\
    --output data/sft/raw_traces.jsonl \\
    --resume-from data/sft/raw_traces.jsonl \\
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \\
    --force-resume \\
    --use-bodhi \\
    > ~/gen.log 2>&1
echo GEN_OK >> ~/pipeline.log

# Upload to the shared, non-seed-prefixed path that launch_5seeds.sh
# followers poll for.
echo "--- uploading raw_traces.jsonl to ${GCS_BASE}/raw_traces.jsonl ---" | tee -a ~/pipeline.log
gsutil -q -m cp data/sft/raw_traces.jsonl "${GCS_BASE}/raw_traces.jsonl"
echo "  final rows: \$(wc -l < data/sft/raw_traces.jsonl)" >> ~/pipeline.log
echo STAGE1_OK >> ~/pipeline.log
echo "=== Stage 1 complete; raw_traces.jsonl in GCS ==="
REMOTE
}

run_pipeline() {
    local remote_cmd
    remote_cmd="$(build_remote_cmd)"
    gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
        --command="$remote_cmd" >>"$LOG" 2>&1
}

stage1_marker_present() {
    gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
        --command='grep -q ^STAGE1_OK ~/pipeline.log' >/dev/null 2>&1
}

vm_state() {
    gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(state)" 2>/dev/null \
        || echo "MISSING"
}

delete_vm() {
    gcloud compute tpus tpu-vm delete "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet \
        >>"$LOG" 2>&1 || true
}

cleanup() {
    log "cleanup: delete VM"
    delete_vm
    log "cleaned up"
}
trap cleanup EXIT

# Verify GCS bucket is reachable before burning chip time.
if ! gsutil ls "$GCS_BASE/" >/dev/null 2>&1; then
    echo "ERROR: cannot list $GCS_BASE/ - is the bucket created and writable?" >&2
    exit 1
fi

# NOTE: we no longer fast-exit when raw_traces.jsonl exists in GCS,
# because the file may be PARTIAL (e.g. the rescue file from a prior
# preempted run). The VM-side resume path (in build_remote_cmd) pulls
# the existing file as a resume base and lets generate_traces.py skip
# already-done prompt_ids. If the file is already complete after
# resume + exclude filtering, generate_traces.py prints "Nothing to
# generate" and exits 0 without spinning up vLLM, so the chip-time
# cost of a no-op call is just one setup_tpu.sh run (~10 min).

preempt_attempt=0
while :; do
    if ! try_create; then
        log "exhausted create retries - giving up"
        exit 1
    fi
    push_tokens
    run_pipeline || true
    if stage1_marker_present; then
        log "Stage 1 complete (STAGE1_OK)"
        break
    fi

    state=$(vm_state)
    log "SSH ended without STAGE1_OK; vm state=$state"
    if [ "$state" = "PREEMPTED" ] || [ "$state" = "MISSING" ]; then
        preempt_attempt=$((preempt_attempt + 1))
        if [ "$preempt_attempt" -ge "$MAX_PREEMPT_RETRIES" ]; then
            log "hit MAX_PREEMPT_RETRIES=$MAX_PREEMPT_RETRIES, giving up"
            break
        fi
        log "preempted - reacquiring (attempt $preempt_attempt/$MAX_PREEMPT_RETRIES)"
        delete_vm
        sleep 30
        continue
    fi

    log "non-preempt failure - not retrying"
    break
done
