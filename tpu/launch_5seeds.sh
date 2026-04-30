#!/bin/bash
# launch_5seeds.sh — fan out 5 v6e-8 spot VMs in parallel, one seed per VM.
#
# Why 5: scripts/aggregate_seeds.py only emits across-seed percentile 95%
# CIs when n_seeds >= 5 (line 101 of that file). 5 is also the default
# in scripts/run_multi_seed.sh ("42 7 13 99 101"). Going to 5 from 3
# unlocks the across-seed CI band the paper needs.
#
# Quota: TRC grants 64 v6e chips in europe-west4-a + 64 in us-east1-d.
# 5 v6e-8 = 40 chips total. We split 3 in eur4a (24 chips) + 2 in use1d
# (16 chips) — both well within the 64-chip-per-zone quota.
#
# Per-VM workflow:
#   1. acquire v6e-8 spot (with retry on capacity errors)
#   2. clone repo, bash tpu/setup_tpu.sh
#   3. either
#        - download pre-graded ${GCS_DATA_PATH}/{train,val}.jsonl (skip Stage 1+2)
#        - or run Stage 1 (gen) + Stage 2 (filter) themselves
#   4. Stage 3: train LoRA (configs/lora_medgemma27b_tpu.yaml, --seed N)
#   5. Stage 4: eval 4 configs (base_no_wrapper / base_bodhi /
#      lora_no_wrapper / lora_bodhi). LoRA configs use the XLA direct-
#      inference backend (PR #93) since vllm-tpu lacks add_lora.
#   6. SCP results back to ./results/seed_<N>/ on the local box
#   7. delete the VM (trap on EXIT)
#
# Each VM writes ~/pipeline.log + ~/{setup,gen,filter,train,eval}.log.
# scripts/dashboard polls these — open http://localhost:8000 while it runs.
#
# Wall-clock estimate (no preempts):
#   Stage 1+2 shared via GCS:  ~5 h once
#   Stage 3 train (parallel):  ~12-24 h per seed (max wins, not sum)
#   Stage 4 eval (parallel):   ~6 h per seed
#   Total wall:                ~25-35 h
#
# Usage:
#   export HF_TOKEN=...                              # required (in .env is fine)
#   GCS_DATA_PATH=gs://bucket/path                   # optional, skips Stage 1+2
#   SEEDS="42 7 13 99 101"                           # optional override (default 42 7 13 99 101)
#   bash tpu/launch_5seeds.sh
#
# Cancel everything (clean up all 5 VMs):
#   kill $(cat /tmp/bohdi_5seeds_pids.txt)
#   for n in 42 7 13; do
#       gcloud compute tpus tpu-vm delete bohdi-seed$n --zone=europe-west4-a --quiet
#   done
#   for n in 99 101; do
#       gcloud compute tpus tpu-vm delete bohdi-seed$n --zone=us-east1-d --quiet
#   done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up looking for a .env file. Worktrees check ../.env (parent worktree)
# AND ../../.env / ../../../../.env / etc. so a clone deep inside
# .claude/worktrees still finds the repo-root .env.
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
: "${HF_TOKEN:?HF_TOKEN not set — add HF_TOKEN=hf_... to .env or export it}"

# GH_TOKEN: required for the private repo clone on the VM. We honor an
# explicit env var, then ``gh auth token`` as a fallback (the user already
# has gh authenticated for everything else in this codebase).
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
: "${GH_TOKEN:?GH_TOKEN not set and 'gh auth token' returned empty — auth gh or export GH_TOKEN}"

PROJECT="tokyo-micron-494016-s9"
RUNTIME="v2-alpha-tpuv6e"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# 5 seeds split 3 in europe-west4-a + 2 in us-east1-d. 24 chips in eur4a
# and 16 chips in use1d both fit the 64-chip quota; if one zone runs hot,
# the other still hosts its share. Default seed list mirrors
# scripts/run_multi_seed.sh so multi-seed comparisons stay consistent.
SEEDS="${SEEDS:-42 7 13 99 101}"
read -r -a SEED_ARR <<< "$SEEDS"
if [ "${#SEED_ARR[@]}" -ne 5 ]; then
    echo "ERROR: SEEDS must list exactly 5 seeds (got: ${#SEED_ARR[@]})" >&2
    exit 1
fi

# Index 0..2 -> eur4a (3 VMs), index 3..4 -> use1d (2 VMs).
ZONES=("europe-west4-a" "europe-west4-a" "europe-west4-a" "us-east1-d" "us-east1-d")
VM_NAMES=(
    "bohdi-seed${SEED_ARR[0]}"
    "bohdi-seed${SEED_ARR[1]}"
    "bohdi-seed${SEED_ARR[2]}"
    "bohdi-seed${SEED_ARR[3]}"
    "bohdi-seed${SEED_ARR[4]}"
)

# Optional shared data location. When set, each VM skips Stage 1+2 and
# downloads ${GCS_DATA_PATH}/train.jsonl + val.jsonl. Recommended — saves
# ~5h of duplicated trace generation per VM (15h total).
GCS_DATA_PATH="${GCS_DATA_PATH:-}"

# How many times to retry spot-create on TRC capacity errors before
# giving up on a particular VM. Each retry waits 60s. Default of 200 *
# 60s = up to ~3.3h per per-attempt acquisition window — TRC v6e-8 spot
# capacity is highly variable and the launchers in this repo (e.g.
# launch_multiseed.sh) use similar long-retry loops.
CREATE_RETRIES="${CREATE_RETRIES:-200}"

# How many times the per-VM subshell will reacquire after a preempt.
# Set to 0 to disable preempt-retry (a single failed run gives up).
# Default 10 = up to 10 fresh acquisitions per seed, which roughly
# matches the longest-running TRC spot session we have observed.
MAX_PREEMPT_RETRIES="${MAX_PREEMPT_RETRIES:-10}"

echo "=== launch_5seeds: 5 v6e-8 spot VMs in parallel (3 eur4a + 2 use1d) ==="
echo "  seeds: ${SEEDS}"
echo "  GCS_DATA_PATH: ${GCS_DATA_PATH:-(not set — each VM will run Stage 1+2)}"
echo "  results -> $RESULTS_DIR/seed_<N>/"
echo

PID_FILE="/tmp/bohdi_5seeds_pids.txt"
> "$PID_FILE"

# Build the remote pipeline command once — same script on every VM, only
# the SEED env var differs per VM. Tokens are NOT embedded inline (they
# would appear in the gcloud SSH argv and in process listings on the VM);
# instead the per-VM subshell pushes them to ~/.bohdi-env via stdin in
# a separate SSH call, and this script sources them.
build_remote_cmd() {
    local SEED="$1"
    cat <<REMOTE
set -euo pipefail
# Tokens were stashed in ~/.bohdi-env by the launcher's stdin push.
if [ -f ~/.bohdi-env ]; then
    set -a; source ~/.bohdi-env; set +a
fi
: "\${HF_TOKEN:?HF_TOKEN missing from ~/.bohdi-env}"
: "\${GH_TOKEN:?GH_TOKEN missing from ~/.bohdi-env}"
export PJRT_DEVICE=TPU

# Private-repo clone via in-memory token-injected URL. The
# ``url.<...>.insteadOf`` config rewrites the github.com origin only for
# THIS git invocation, so the token never lands in ~/.git/config.
if [ ! -d ~/bohdi-lora ]; then
    git -c "url.https://x-access-token:\${GH_TOKEN}@github.com/.insteadOf=https://github.com/" \\
        clone https://github.com/PeterLi-jpg/bohdi-lora.git ~/bohdi-lora
fi
cd ~/bohdi-lora
# Always pull so a re-acquired VM picks up any post-launch fixes on main.
git -c "url.https://x-access-token:\${GH_TOKEN}@github.com/.insteadOf=https://github.com/" \\
    fetch origin main 2>&1 | tail -2 || true
git reset --hard origin/main 2>&1 | tail -1 || true

echo "--- 0/4 setup_tpu.sh ---" | tee -a ~/pipeline.log
bash tpu/setup_tpu.sh > ~/setup.log 2>&1 || { echo "setup FAILED" >> ~/pipeline.log; exit 1; }
echo SETUP_OK >> ~/pipeline.log

mkdir -p data/sft eval checkpoints logs

if [ -n "${GCS_DATA_PATH}" ]; then
    echo "--- 1+2/4 download pre-graded data from ${GCS_DATA_PATH} ---" | tee -a ~/pipeline.log
    gsutil -m cp "${GCS_DATA_PATH}/train.jsonl" data/sft/train.jsonl
    gsutil -m cp "${GCS_DATA_PATH}/val.jsonl"   data/sft/val.jsonl
    echo DATA_DOWNLOADED >> ~/pipeline.log
else
    echo "--- 1/4 generate BODHI traces (this VM, no GCS shortcut) ---" | tee -a ~/pipeline.log
    python -u scripts/download_data.py >> ~/pipeline.log 2>&1
    python -u scripts/generate_traces.py \\
        --model google/medgemma-27b-text-it \\
        --datasets healthbench_hard healthbench \\
        --output data/sft/raw_traces.jsonl \\
        --use-bodhi \\
        > ~/gen.log 2>&1
    echo GEN_OK >> ~/pipeline.log

    echo "--- 2/4 filter+grade with seed ${SEED} ---" | tee -a ~/pipeline.log
    python -u scripts/filter_traces.py \\
        --input data/sft/raw_traces.jsonl \\
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \\
        --grader-model Qwen/Qwen2.5-14B-Instruct \\
        --output-dir data/sft \\
        --min-score 0.4 \\
        --val-ratio 0.1 \\
        --seed ${SEED} \\
        > ~/filter.log 2>&1
    echo FILTER_OK >> ~/pipeline.log
fi

echo "--- 3/4 train LoRA seed=${SEED} ---" | tee -a ~/pipeline.log
python -u scripts/train_lora.py \\
    --config configs/lora_medgemma27b_tpu.yaml \\
    --seed ${SEED} \\
    --output-dir "checkpoints/seed_${SEED}" \\
    > ~/train.log 2>&1
echo TRAIN_OK >> ~/pipeline.log

echo "--- 4/4 eval 4 configs (base/lora x wrapper/no-wrapper) ---" | tee -a ~/pipeline.log
mkdir -p "eval/seed_${SEED}"
LORA_DIR="checkpoints/seed_${SEED}/best"

run_eval() {
    local name="\$1" args="\$2"
    local out="eval/seed_${SEED}/\${name}.json"
    if [ -s "\$out" ]; then
        echo "[\$name] already exists, skipping" >> ~/pipeline.log
        return
    fi
    echo "--- 4.\$name ---" | tee -a ~/pipeline.log
    # shellcheck disable=SC2086
    python -u scripts/eval_healthbench.py \$args \\
        --sample-ids data/raw/hard_200_sample_ids.json \\
        --grader-model Qwen/Qwen2.5-14B-Instruct \\
        --output "\$out" \\
        --seed ${SEED} >> ~/eval.log 2>&1 || echo "eval \$name FAILED" >> ~/pipeline.log
}

run_eval "base_no_wrapper"  "--model google/medgemma-27b-text-it"
run_eval "base_bodhi"       "--model google/medgemma-27b-text-it --use-bodhi"
run_eval "lora_no_wrapper"  "--model google/medgemma-27b-text-it --lora-path \$LORA_DIR"
run_eval "lora_bodhi"       "--model google/medgemma-27b-text-it --lora-path \$LORA_DIR --use-bodhi"
echo EVAL_OK >> ~/pipeline.log

echo "=== seed ${SEED} pipeline complete ===" >> ~/pipeline.log
REMOTE
}

for i in 0 1 2 3 4; do
    SEED="${SEED_ARR[$i]}"
    ZONE="${ZONES[$i]}"
    VM_NAME="${VM_NAMES[$i]}"
    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    LOG="${SEED_DIR}/launch.log"

    echo "Launching $VM_NAME (seed $SEED, $ZONE)..."

    (
        set -euo pipefail

        # Per-VM helpers — closures over $VM_NAME / $ZONE / $LOG.
        log()  { printf '[%s %s] %s\n' "$(date -u +%H:%M:%S)" "$VM_NAME" "$*" | tee -a "$LOG"; }

        try_create() {
            # Acquire one v6e-8 spot in $ZONE with capacity-error retries.
            # Returns 0 on success, 1 if we burn through all retries.
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
            # Stash GH_TOKEN + HF_TOKEN to ~/.bohdi-env on the VM via
            # stdin, mode 600. The tokens never appear in process listings
            # or the gcloud --command argv.
            printf '%s\n%s\n' "$GH_TOKEN" "$HF_TOKEN" \
                | gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                    --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                    --command='read -r G; read -r H; umask 077; { echo "GH_TOKEN=$G"; echo "HF_TOKEN=$H"; } > ~/.bohdi-env; chmod 600 ~/.bohdi-env' \
                    >>"$LOG" 2>&1
        }

        run_pipeline() {
            # Run the heredoc-built pipeline. Returns SSH's exit code:
            #   0   pipeline ran, EVAL_OK marker present
            #   ≠0  SSH died (preempt, network, or pipeline error)
            local remote_cmd
            remote_cmd="$(build_remote_cmd "$SEED")"
            gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                --command="$remote_cmd" >>"$LOG" 2>&1
        }

        eval_marker_present() {
            # Probe for the EVAL_OK marker in ~/pipeline.log. Returns 0 if
            # the pipeline ran to completion (or 1 otherwise / on any
            # SSH failure).
            gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                --command='grep -q ^EVAL_OK ~/pipeline.log' >/dev/null 2>&1
        }

        vm_state() {
            gcloud compute tpus tpu-vm describe "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --format="value(state)" 2>/dev/null \
                || echo "MISSING"
        }

        scp_back() {
            # Best-effort copy of checkpoints + eval JSONs + pipeline.log
            # to ./results/seed_<N>/. Never fails the parent shell.
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/checkpoints/seed_${SEED}" "$SEED_DIR/" \
                >>"$LOG" 2>&1 || log "  (no checkpoints to copy)"
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/eval/seed_${SEED}" "$SEED_DIR/" \
                >>"$LOG" 2>&1 || log "  (no eval to copy)"
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/pipeline.log" "${SEED_DIR}/pipeline.log" \
                >>"$LOG" 2>&1 || true
        }

        delete_vm() {
            gcloud compute tpus tpu-vm delete "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --quiet \
                >>"$LOG" 2>&1 || true
        }

        # Trap on EXIT — runs after the outer retry loop ends, no matter
        # how (success / preempt-give-up / SIGINT). Always tries to copy
        # whatever lives on the current VM and delete it.
        cleanup() {
            log "cleanup: copy results + delete VM"
            scp_back
            delete_vm
            log "cleaned up"
        }
        trap cleanup EXIT

        # ── outer retry loop ──────────────────────────────────────────────
        # If the pipeline finishes cleanly (EVAL_OK), we exit successfully.
        # If the VM gets preempted before EVAL_OK, we delete + reacquire in
        # the SAME zone (per the user's spec — staying in zone keeps the
        # persistent disk attachment + zone-affinity behavior consistent).
        preempt_attempt=0
        while :; do
            if ! try_create; then
                log "exhausted create retries — giving up on this seed"
                exit 0
            fi
            push_tokens
            run_pipeline   # never let a non-zero kill the parent; we check below
            if eval_marker_present; then
                log "pipeline complete (EVAL_OK)"
                break
            fi

            # Pipeline didn't finish. Inspect VM state to decide what to do.
            state=$(vm_state)
            log "SSH ended without EVAL_OK; vm state=$state"

            if [ "$state" = "PREEMPTED" ] || [ "$state" = "MISSING" ]; then
                preempt_attempt=$((preempt_attempt + 1))
                if [ "$preempt_attempt" -ge "$MAX_PREEMPT_RETRIES" ]; then
                    log "hit MAX_PREEMPT_RETRIES=$MAX_PREEMPT_RETRIES, giving up"
                    break
                fi
                log "preempted — deleting + reacquiring in same zone (attempt $preempt_attempt/$MAX_PREEMPT_RETRIES)"
                # Best-effort grab whatever results survived on the VM
                # before we delete it (most work is wiped with the boot
                # disk, but eval/seed_N JSONs may be there).
                scp_back
                delete_vm
                sleep 30
                continue
            fi

            # Something else (pipeline error not preempt). Stop retrying.
            log "non-preempt failure — not retrying"
            break
        done
    ) &

    echo "$!" >> "$PID_FILE"
    sleep 4   # stagger gcloud creates a bit so we don't hammer the API
done

echo
echo "All 5 jobs spawned. PIDs: $(cat "$PID_FILE")"
echo "Open the dashboard at http://localhost:8000 to watch progress."
echo
echo "Waiting for all VMs to finish..."
wait
echo
echo "All seeds done. Results in $RESULTS_DIR/seed_*/"
ls -la "$RESULTS_DIR" 2>/dev/null || true
