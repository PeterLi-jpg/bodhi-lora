#!/bin/bash
# launch_5seeds_tunix.sh: fan out 5 v6e-8 spot VMs in parallel, one seed
# per VM, training Stage 3 via the tunix + qwix path.
#
# Mirrors tpu/launch_5seeds_maxtext.sh structure exactly. Differences:
#
#   1. Stage 3 (LoRA fine-tune) runs scripts/train_lora_tunix.py with
#      configs/lora_medgemma27b_tunix_smoke.yaml (default smoke config;
#      production runs override TRAIN_CONFIG).
#   2. No HF -> Orbax conversion. tunix loads the safetensors weights
#      directly from the HF cache (snapshot_download into /dev/shm/hf,
#      already set up by setup_tpu.sh).
#   3. After training, scripts/export_tunix_lora_to_peft.py converts the
#      tunix Orbax LoRA checkpoint to a PEFT-format adapter under
#      checkpoints/seed_<N>/best/ so the existing vllm-tpu LoRA eval
#      path keeps working unchanged.
#
# Stage 3b (data conversion to MaxText input format) is retained for
# parity with the maxtext launcher; the tunix trainer does not consume
# the converted data, but the conversion is idempotent and harmless.
#
# Why 5 seeds, why this zone split, quota math, wall-clock estimate, and
# every other rationale: see the header in launch_5seeds.sh. Same logic
# here, just with the tunix invocation.
#
# Usage:
#   export HF_TOKEN=...                              # required (in .env is fine)
#   GCS_DATA_PATH=gs://bucket/path                   # optional, skips Stage 1+2
#   GCS_OUTPUT_PATH=gs://bucket/runs                 # strongly recommended
#   SEEDS="42 7 13 99 101"                           # optional override (default 42 7 13 99 101)
#   SECOND_GRADER_MODEL=Qwen/Qwen2.5-14B-Instruct  # optional cross-grader pass
#   bash tpu/launch_5seeds_tunix.sh
#
# Cross-grader pass: when SECOND_GRADER_MODEL is set, each VM re-grades
# the same 4 eval configs with that model and reports Spearman rho
# between the two graders. Off by default (adds ~12h H100 of grader
# compute per seed). Recommended secondary: a different family from
# the primary Llama grader, e.g. Qwen/Qwen2.5-14B-Instruct.
#
# Cancel everything (clean up all 5 VMs):
#   kill $(cat /tmp/bohdi_5seeds_tunix_pids.txt)
#   for n in 42 7 13; do
#       gcloud compute tpus tpu-vm delete bohdi-tx-seed$n --zone=europe-west4-a --quiet
#   done
#   for n in 99 101; do
#       gcloud compute tpus tpu-vm delete bohdi-tx-seed$n --zone=us-east1-d --quiet
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
: "${HF_TOKEN:?HF_TOKEN not set, add HF_TOKEN=hf_... to .env or export it}"

# GH_TOKEN: required for the private repo clone on the VM. We honor an
# explicit env var, then ``gh auth token`` as a fallback (the user already
# has gh authenticated for everything else in this codebase).
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
: "${GH_TOKEN:?GH_TOKEN not set and 'gh auth token' returned empty, auth gh or export GH_TOKEN}"

PROJECT="tokyo-micron-494016-s9"
RUNTIME="v2-alpha-tpuv6e"
RESULTS_DIR="./results_tunix"
mkdir -p "$RESULTS_DIR"

# 5 seeds split 3 in europe-west4-a + 2 in us-east1-d. 24 chips in eur4a
# and 16 chips in use1d both fit the 64-chip quota; if one zone runs hot,
# the other still hosts its share. Default seed list mirrors
# scripts/run_multi_seed.sh so multi-seed comparisons stay consistent.
SEEDS="${SEEDS:-42 7 13 99 101}"
read -r -a SEED_ARR <<< "$SEEDS"
N_SEEDS="${#SEED_ARR[@]}"
if [ "$N_SEEDS" -lt 1 ] || [ "$N_SEEDS" -gt 5 ]; then
    echo "ERROR: SEEDS must list 1-5 seeds (got: $N_SEEDS)" >&2
    exit 1
fi

# Slots 0..2 -> eur4a, slots 3..4 -> use1d. We slice this to N_SEEDS so
# a smoke run with SEEDS="42" gets one eur4a VM, SEEDS="42 7" gets two
# eur4a, SEEDS="42 7 13 99" gets three eur4a + one use1d, and the full
# default gets the canonical 3 + 2 split. Across-seed CIs from
# scripts/aggregate_seeds.py still need n_seeds>=5 in the final paper run.
if [ -n "${ZONES_OVERRIDE:-}" ]; then
    # Explicit per-seed zone list, e.g. when launching followers around an
    # already-running VM in another zone. Must list one zone per seed,
    # space-separated. Example: ZONES_OVERRIDE="europe-west4-a europe-west4-a us-east1-d us-east1-d"
    read -r -a ZONES <<< "$ZONES_OVERRIDE"
    if [ "${#ZONES[@]}" -ne "$N_SEEDS" ]; then
        echo "ERROR: ZONES_OVERRIDE has ${#ZONES[@]} entries, need $N_SEEDS" >&2
        exit 1
    fi
else
    ZONE_SLOTS=("europe-west4-a" "europe-west4-a" "europe-west4-a" "us-east1-d" "us-east1-d")
    ZONES=("${ZONE_SLOTS[@]:0:$N_SEEDS}")
fi
# VM names get a "tx-" infix so tunix launches don't collide with any
# in-flight launch_5seeds.sh PyTorch VMs (bohdi-seed<N>) or
# launch_5seeds_maxtext.sh VMs (bohdi-mt-seed<N>) holding the same name.
VM_NAMES=()
for s in "${SEED_ARR[@]}"; do
    VM_NAMES+=("bohdi-tx-seed${s}")
done

# Optional shared data location. When set, each VM skips Stage 1+2 and
# downloads ${GCS_DATA_PATH}/train.jsonl + val.jsonl. Recommended: saves
# ~5h of duplicated trace generation per VM (15h total).
GCS_DATA_PATH="${GCS_DATA_PATH:-}"

# Cross-grader bias-control. Each VM re-grades the same 4 generated
# response sets with this secondary grader IN PROCESS via
# eval_healthbench.py's --secondary-grader-model flag (see run_eval
# below). Set to empty string explicitly to skip
# ("SECOND_GRADER_MODEL='' bash ..."). The secondary pass adds ~30-40%
# to Stage 4 wall on TPU (it's just the grader pass over
# already-generated responses, not a full regeneration).
#
# Default: Mistral-7B-Instruct-v0.3 — picked specifically because it is
# a THIRD family from both:
#   - the filter (Stage 2) grader, which is Qwen/Qwen2.5-14B-Instruct
#   - the primary eval (Stage 4) grader, which is meta-llama/Llama-3.1-8B-Instruct
# Using a same-family cross-grader (e.g. Qwen for both filter + secondary)
# would let a Qwen-flavored bias propagate from training-data selection
# into the bias-control check itself; Mistral cleanly separates them.
SECOND_GRADER_MODEL="${SECOND_GRADER_MODEL-mistralai/Mistral-7B-Instruct-v0.3}"

# Smoke knobs (mirrors tpu/launch_multiseed.sh).
# MAX_EXAMPLES caps Stage 1 trace generation; EVAL_MAX caps Stage 4 eval
# prompts. TRAIN_CONFIG overrides the train YAML; default is the tunix
# smoke config for fast end-to-end validation. Production runs override
# via env var (e.g. configs/lora_medgemma27b_tunix.yaml when that lands).
# All three are empty by default = full production run for the eval knobs.
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
EVAL_MAX="${EVAL_MAX:-}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lora_medgemma27b_tunix_smoke.yaml}"
# Model id used by Stage 1 trace gen, Stage 3b tokenizer, Stage 3 PEFT
# export, and the four Stage 4 eval invocations. Defaults to MedGemma-27B
# for production. The smoke configs override to a small Gemma3 variant
# (e.g. google/gemma-3-270m-it) so the JIT compile fits in v6e-8 HBM
# without 22+GB activations; v33 hit train_step OOM at 27B with
# max_seq=256, freeing 18.61G but needing 22.86G. Keeping the two
# (this env var + the YAML's model.name) in sync is required: the
# trainer reads model.name from the YAML, the rest of the pipeline
# reads MODEL_NAME from env. The launch wrapper for the smoke
# (e.g. tpu/launch_5seeds_tunix.sh invocations under SMOKE_*) sets
# both to the same value.
MODEL_NAME="${MODEL_NAME:-google/medgemma-27b-text-it}"
_GEN_MAX_FLAG=""
[ -n "$MAX_EXAMPLES" ] && _GEN_MAX_FLAG="--max-examples ${MAX_EXAMPLES}"
_EVAL_MAX_FLAG=""
[ -n "$EVAL_MAX" ] && _EVAL_MAX_FLAG="--max-examples ${EVAL_MAX}"

# How many times to retry spot-create on TRC capacity errors before
# giving up on a particular VM. Each retry waits 60s. Default of 200 *
# 60s = up to ~3.3h per per-attempt acquisition window: TRC v6e-8 spot
# capacity is highly variable and the launchers in this repo (e.g.
# launch_multiseed.sh) use similar long-retry loops.
CREATE_RETRIES="${CREATE_RETRIES:-200}"

# How many times the per-VM subshell will reacquire after a preempt.
# Set to 0 to disable preempt-retry (a single failed run gives up).
# Default 10 = up to 10 fresh acquisitions per seed, which roughly
# matches the longest-running TRC spot session we have observed.
MAX_PREEMPT_RETRIES="${MAX_PREEMPT_RETRIES:-10}"

echo "=== launch_5seeds_tunix: 5 v6e-8 spot VMs in parallel (3 eur4a + 2 use1d) ==="
echo "  seeds: ${SEEDS}"
echo "  GCS_DATA_PATH:        ${GCS_DATA_PATH:-(not set, each VM will run Stage 1+2)}"
echo "  GCS_OUTPUT_PATH:      ${GCS_OUTPUT_PATH:-(not set, preempts will lose progress)}"
echo "  SECOND_GRADER_MODEL:  ${SECOND_GRADER_MODEL:-(not set, cross-grader pass disabled)}"
echo "  TRAIN_CONFIG:         ${TRAIN_CONFIG}"
echo "  MODEL_NAME:           ${MODEL_NAME}"
echo "  results -> $RESULTS_DIR/seed_<N>/"
echo

if [ -z "${GCS_OUTPUT_PATH:-}" ]; then
    cat <<'WARN' >&2

WARNING: GCS_OUTPUT_PATH is not set. Each preempt will lose all
progress for that seed (boot disk wiped). To survive preempts, set
GCS_OUTPUT_PATH to a bucket the TPUs can write to:

  GCS_OUTPUT_PATH=gs://your-bucket/bohdi-runs bash tpu/launch_5seeds_tunix.sh

The launcher uploads stage outputs (raw_traces, train/val, training
checkpoints, eval JSONs) to ${GCS_OUTPUT_PATH}/seed_<N>/ as it goes,
and a re-acquired VM after a preempt downloads whatever survived
before continuing, so a preempt costs at most one stage's increment,
not the whole run.

WARN
fi


PID_FILE="/tmp/bohdi_5seeds_tunix_pids.txt"
> "$PID_FILE"

# Build the remote pipeline command once: same script on every VM, only
# the SEED env var (and IS_LEADER flag) differs per VM. Tokens are NOT
# embedded inline (they would appear in the gcloud SSH argv and in
# process listings on the VM); instead the per-VM subshell pushes them
# to ~/.bohdi-env via stdin in a separate SSH call, and this script
# sources them.
#
# Stage-1 leader-elect: ``IS_LEADER=1`` means this VM does the trace
# generation (Stage 1) itself and uploads to GCS. ``IS_LEADER=0`` (the
# other 4 seeds) waits up to STAGE1_WAIT_S for the leader's
# raw_traces.jsonl to land in GCS, then downloads + skips Stage 1. The
# leader is whichever seed appears first in $SEEDS: by default seed 42.
# Followers fall through to running Stage 1 themselves if the wait
# times out, so a preempted leader doesn't strand the followers.
build_remote_cmd() {
    local SEED="$1"
    local IS_LEADER="$2"
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
# Do NOT swallow failures here: if fetch/reset fails the daemon would silently
# run stale code, and the launcher's wait loop would later see "DONE" against
# whatever ancient checkout was on disk. Abort instead so wait_for_completion
# classifies it as a real failure.
if ! git -c "url.https://x-access-token:\${GH_TOKEN}@github.com/.insteadOf=https://github.com/" \\
        fetch origin main 2>&1 | tail -2; then
    echo "git fetch origin main FAILED" >> ~/pipeline.log
    exit 1
fi
if ! git reset --hard origin/main 2>&1 | tail -1; then
    echo "git reset --hard origin/main FAILED" >> ~/pipeline.log
    exit 1
fi

echo "--- 0/4 setup_tpu.sh ---" | tee -a ~/pipeline.log
bash tpu/setup_tpu.sh > ~/setup.log 2>&1 || { echo "setup FAILED" >> ~/pipeline.log; exit 1; }
echo SETUP_OK >> ~/pipeline.log
# Use the venv interpreter created by setup_tpu.sh (system python3 is 3.10
# on v6e VMs; we need 3.11). Backslash-\$ keeps the expansion remote-side.
PY=~/.venv-py311/bin/python
# /etc/profile.d/bohdi-hf-cache.sh is sourced only by login shells; this
# daemon is non-login (nohup setsid bash). Source it explicitly so HF_HOME
# + TRANSFORMERS_CACHE actually point at /dev/shm (or /mnt/cache). Without
# this the huggingface library ignores the redirect and downloads to the
# default ~/.cache/huggingface on the boot disk, ENOSPC at Stage 2.
[ -f /etc/profile.d/bohdi-hf-cache.sh ] && source /etc/profile.d/bohdi-hf-cache.sh
echo "  HF_HOME=\${HF_HOME:-(unset)}" >> ~/pipeline.log

mkdir -p data/sft eval checkpoints logs "checkpoints/seed_${SEED}"

# Resume from prior progress in GCS, if any.
# Each stage's output is uploaded to GCS as it lands (see "upload"
# blocks below). On a preempt + reacquire the new VM downloads
# whatever survived before continuing, so the worst case is losing
# one stage's increment instead of the whole run.
#
# Layout:
#   \${GCS_OUTPUT_PATH}/raw_traces.jsonl          shared across all 5 seeds
#                                                  (BODHI is deterministic
#                                                  with greedy decoding,
#                                                  so Stage 1 produces the
#                                                  same traces regardless
#                                                  of seed; first seed to
#                                                  finish wins, others skip)
#   \${GCS_OUTPUT_PATH}/seed_<N>/train.jsonl      per-seed (filter shuffle)
#   \${GCS_OUTPUT_PATH}/seed_<N>/val.jsonl
#   \${GCS_OUTPUT_PATH}/seed_<N>/checkpoints/...  per-seed (LoRA init)
#   \${GCS_OUTPUT_PATH}/seed_<N>/eval/...
# GCS_OUTPUT_PATH lives only on the LOCAL launcher (heredoc-build time).
# Bake the resolved value in here without backslashes so the remote VM
# sees a literal path. The previous \${GCS_OUTPUT_PATH...} form expanded
# on the remote, where the env var is unset -> GCS_BASE was empty,
# resume + sidecar upload paths silently no-op'd, and every VM ran
# Stage 1 from scratch (and lost training state on every preempt).
GCS_BASE="${GCS_OUTPUT_PATH:+${GCS_OUTPUT_PATH%/}}"
GCS_SEED_DIR="\${GCS_BASE:+\${GCS_BASE}/seed_${SEED}}"
if [ -n "\${GCS_SEED_DIR:-}" ]; then
    echo "--- 0b checking \${GCS_SEED_DIR} for prior progress ---" | tee -a ~/pipeline.log
    # Shared across-seeds Stage 1 output first.
    gsutil -q cp "\${GCS_BASE}/raw_traces.jsonl" data/sft/raw_traces.jsonl 2>/dev/null \\
        && echo "  resumed: raw_traces.jsonl from GCS (shared across seeds)" >> ~/pipeline.log || true
    # Per-seed inputs/outputs.
    gsutil -q cp "\${GCS_SEED_DIR}/train.jsonl" data/sft/train.jsonl 2>/dev/null \\
        && echo "  resumed: train.jsonl from GCS" >> ~/pipeline.log || true
    gsutil -q cp "\${GCS_SEED_DIR}/val.jsonl" data/sft/val.jsonl 2>/dev/null \\
        && echo "  resumed: val.jsonl from GCS" >> ~/pipeline.log || true
    gsutil -q -m rsync -r "\${GCS_SEED_DIR}/checkpoints/" "checkpoints/seed_${SEED}/" 2>/dev/null \\
        && echo "  resumed: checkpoints from GCS" >> ~/pipeline.log || true
    gsutil -q -m rsync -r "\${GCS_SEED_DIR}/eval/" "eval/seed_${SEED}/" 2>/dev/null \\
        && echo "  resumed: eval JSONs from GCS" >> ~/pipeline.log || true
    if gsutil -q ls "\${GCS_SEED_DIR}/maxtext/dataset/" >/dev/null 2>&1; then
        mkdir -p data/sft/maxtext
        gsutil -q -m rsync -r "\${GCS_SEED_DIR}/maxtext/dataset/" data/sft/maxtext/ \\
            && echo "--- 0c resumed: maxtext dataset from GCS ---" >> ~/pipeline.log || true
    fi
fi

# Helper: upload a path to GCS if GCS_OUTPUT_PATH is set, never fail loudly.
gcs_upload() {
    local local_path="\$1"
    local remote_subpath="\$2"
    [ -z "\${GCS_SEED_DIR:-}" ] && return 0
    [ -e "\$local_path" ] || return 0
    gsutil -q -m cp -r "\$local_path" "\${GCS_SEED_DIR}/\${remote_subpath}" 2>&1 \\
        | tail -3 >> ~/pipeline.log || true
}
gcs_rsync() {
    local local_path="\$1"
    local remote_subpath="\$2"
    [ -z "\${GCS_SEED_DIR:-}" ] && return 0
    [ -d "\$local_path" ] || return 0
    gsutil -q -m rsync -r "\$local_path" "\${GCS_SEED_DIR}/\${remote_subpath}" 2>&1 \\
        | tail -3 >> ~/pipeline.log || true
}

if [ -n "${GCS_DATA_PATH}" ] && [ ! -s data/sft/train.jsonl ]; then
    echo "--- 1+2/4 download pre-graded data from ${GCS_DATA_PATH} ---" | tee -a ~/pipeline.log
    gsutil -m cp "${GCS_DATA_PATH}/train.jsonl" data/sft/train.jsonl
    gsutil -m cp "${GCS_DATA_PATH}/val.jsonl"   data/sft/val.jsonl
    echo DATA_DOWNLOADED >> ~/pipeline.log
elif [ -s data/sft/train.jsonl ] && [ -s data/sft/val.jsonl ]; then
    echo "--- 1+2/4 train/val.jsonl already present (resumed), skipping Stage 1+2 ---" | tee -a ~/pipeline.log
else
    if [ ! -s data/sft/raw_traces.jsonl ] && [ "${IS_LEADER}" = "0" ] && [ -n "\${GCS_BASE:-}" ]; then
        # Follower: poll GCS for the leader's raw_traces.jsonl. Falls
        # through to local Stage 1 if the leader doesn't deliver in
        # STAGE1_WAIT_S seconds (default 6 hours: Stage 1 takes ~3-5h
        # so this gives the leader generous slack including a preempt).
        STAGE1_WAIT_S="\${STAGE1_WAIT_S:-21600}"
        echo "--- 1/4 follower waiting for leader's raw_traces.jsonl (up to \${STAGE1_WAIT_S}s) ---" \\
            | tee -a ~/pipeline.log
        deadline=\$((SECONDS + STAGE1_WAIT_S))
        while [ \$SECONDS -lt \$deadline ]; do
            if gsutil -q stat "\${GCS_BASE}/raw_traces.jsonl" 2>/dev/null; then
                gsutil -q cp "\${GCS_BASE}/raw_traces.jsonl" data/sft/raw_traces.jsonl
                echo "  follower received shared raw_traces.jsonl from GCS" >> ~/pipeline.log
                break
            fi
            sleep 60
        done
        if [ ! -s data/sft/raw_traces.jsonl ]; then
            echo "  follower wait timed out, falling through to local Stage 1" >> ~/pipeline.log
        fi
    fi

    # filter_traces.py loads HealthBench rubrics directly from the raw
    # JSONL files via --healthbench-data, so they need to be present even
    # when we skip Stage 1 (resume from a pre-generated raw_traces.jsonl).
    # download_data.py is idempotent (checks before downloading), so it's
    # safe to run unconditionally.
    \${PY} -u scripts/download_data.py >> ~/pipeline.log 2>&1

    if [ ! -s data/sft/raw_traces.jsonl ]; then
        echo "--- 1/4 generate BODHI traces (leader=${IS_LEADER}) ---" | tee -a ~/pipeline.log
        # Exclude all 1000 HealthBench Hard prompts so per-seed bootstrap
        # eval is honestly held-out (issue #60).
        \${PY} -u scripts/generate_traces.py \\
            --model ${MODEL_NAME} \\
            --datasets healthbench_hard healthbench \\
            --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \\
            --output data/sft/raw_traces.jsonl \\
            --use-bodhi \\
            ${_GEN_MAX_FLAG} \\
            > ~/gen.log 2>&1
        echo GEN_OK >> ~/pipeline.log
        # raw_traces.jsonl is identical across seeds (BODHI = greedy decode);
        # publish to the shared GCS path so the other 4 seeds skip Stage 1.
        # Note: the leader uploads first; a follower that fell through
        # the wait timeout also uploads (idempotent: same content).
        if [ -n "\${GCS_BASE:-}" ]; then
            gsutil -q -m cp data/sft/raw_traces.jsonl "\${GCS_BASE}/raw_traces.jsonl" 2>&1 \\
                | tail -3 >> ~/pipeline.log || true
            echo "  uploaded raw_traces.jsonl to shared GCS path" >> ~/pipeline.log
        fi
    else
        echo "--- 1/4 raw_traces.jsonl already present (resumed/follower), skipping Stage 1 ---" | tee -a ~/pipeline.log
    fi

    echo "--- 2/4 filter+grade with seed ${SEED} ---" | tee -a ~/pipeline.log
    # Defensive --exclude-ids drops any HealthBench Hard rows that may have
    # survived in a legacy raw_traces.jsonl (issue #60).
    \${PY} -u scripts/filter_traces.py \\
        --input data/sft/raw_traces.jsonl \\
        --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \\
        --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \\
        --grader-model Qwen/Qwen2.5-14B-Instruct \\
        --output-dir data/sft \\
        --min-score 0.4 \\
        --val-ratio 0.1 \\
        --seed ${SEED} \\
        > ~/filter.log 2>&1
    echo FILTER_OK >> ~/pipeline.log
    gcs_upload data/sft/train.jsonl train.jsonl
    gcs_upload data/sft/val.jsonl val.jsonl
fi

# Preflight leakage gate (issue #60). Aborts the run before training if any
# HealthBench Hard prompt ended up in train.jsonl. Cheap to run; invaluable
# when something upstream regresses (e.g., a stale GCS resume base).
echo "--- 2.5/4 preflight leakage gate ---" | tee -a ~/pipeline.log
# download_data.py is idempotent and the preflight gate needs the raw
# HealthBench JSONL files. Stage 1 normally fetches them, but on a
# resumed run (train/val.jsonl pulled from GCS) Stage 1 was skipped, so
# call here unconditionally.
\${PY} -u scripts/download_data.py >> ~/pipeline.log 2>&1 || true
# SKIP_OVERLAP_CHECK=1 escape hatch for audit/replay runs (per #124).
if [ "\${SKIP_OVERLAP_CHECK:-0}" = "1" ]; then
    echo "WARNING: SKIP_OVERLAP_CHECK=1, skipping leakage gate" | tee -a ~/pipeline.log
else
    \${PY} -u scripts/check_dataset_overlap.py \\
        --train-jsonl data/sft/train.jsonl \\
        --tag-overlap >> ~/pipeline.log 2>&1
fi
echo PREFLIGHT_OK >> ~/pipeline.log

# The Stage-2 grader (and any Stage-1 generation if it ran) used a
# vllm-tpu Docker container with --privileged, started via 'sudo docker
# run', so the bind-mounted HF caches fill up with root-owned files.
# train_lora_tunix.py runs as the regular user and would hit
# "PermissionError: [Errno 13] Permission denied" on the first
# AutoTokenizer.from_pretrained() download attempt. Chown the caches
# back to the user before Stage 3 so HF Hub downloads can proceed.
#
# Three locations to chown:
#   * ~/.cache/huggingface  — default HF cache when HF_HOME is unset
#   * ~/.xla_cache          — XLA persistent compile cache, also bind-mounted
#   * /dev/shm/hf           — HF_HOME on this VM (set by
#       /etc/profile.d/bohdi-hf-cache.sh; the vllm-tpu container's
#       --privileged mount makes this root-owned in /dev/shm too).
#       v38 hit this exact failure: train_lora_tunix's snapshot_download
#       got "PermissionError: '/dev/shm/hf/hub/models--google--gemma-3-4b-it'"
#       on the first run after Stage 2 finished.
# Each path may or may not exist on a fresh VM; silence "no such file"
# warnings with 2>/dev/null and || true.
sudo chown -R "$USER:$USER" ~/.cache/huggingface ~/.xla_cache /dev/shm/hf 2>/dev/null || true

# Convert train/val JSONL -> MaxText input format (Unit 5).
# Retained for parity with the maxtext launcher; tunix does not consume
# the converted data, but the conversion is idempotent and the skip-if-
# present guard means it is a no-op on resumed runs. Drop this block in
# a follow-up cleanup PR once the maxtext path is fully removed.
if [ ! -d data/sft/maxtext ] || [ -z "\$(ls -A data/sft/maxtext 2>/dev/null)" ]; then
    echo "--- 3b/4 convert train/val.jsonl -> MaxText format ---" | tee -a ~/pipeline.log
    mkdir -p data/sft/maxtext
    \${PY} -u scripts/convert_traces_to_maxtext.py \\
        --train data/sft/train.jsonl \\
        --val data/sft/val.jsonl \\
        --tokenizer ${MODEL_NAME} \\
        --output-dir data/sft/maxtext \\
        > ~/convert_data.log 2>&1
    echo CONVERT_DATA_OK >> ~/pipeline.log
    gcs_rsync data/sft/maxtext/ maxtext/dataset/
else
    echo "--- 3b/4 MaxText-format dataset already present, skipping conversion ---" | tee -a ~/pipeline.log
fi

echo "--- 3/4 train LoRA (tunix) seed=${SEED} ---" | tee -a ~/pipeline.log
# Sidecar: rsync checkpoints/seed_<SEED>/ to GCS every 5 min while
# training runs. Trap kills it after train_lora_tunix.py exits
# regardless of how: preempt, success, or python exception.
SIDECAR_PID=""
if [ -n "\${GCS_SEED_DIR:-}" ]; then
    (
        while true; do
            sleep 300
            gsutil -q -m rsync -r "checkpoints/seed_${SEED}/" "\${GCS_SEED_DIR}/checkpoints/" 2>&1 \\
                | tail -3 >> ~/pipeline.log || true
        done
    ) &
    SIDECAR_PID=\$!
    echo "  GCS rsync sidecar pid=\${SIDECAR_PID} (every 300s)" >> ~/pipeline.log
fi
trap '[ -n "'"\${SIDECAR_PID}"'" ] && kill '"\${SIDECAR_PID}"' 2>/dev/null || true' EXIT
\${PY} -u scripts/train_lora_tunix.py \\
    --config "${TRAIN_CONFIG}" \\
    --seed "${SEED}" \\
    --output-dir checkpoints/seed_${SEED} \\
    > ~/train.log 2>&1
echo TRAIN_OK >> ~/pipeline.log
[ -n "\${SIDECAR_PID}" ] && kill \${SIDECAR_PID} 2>/dev/null || true
gcs_rsync "checkpoints/seed_${SEED}/" "checkpoints/"

echo "--- 3/4 export LoRA -> PEFT (tunix) ---" | tee -a ~/pipeline.log
\${PY} -u scripts/export_tunix_lora_to_peft.py \\
    --orbax-dir "checkpoints/seed_${SEED}/orbax" \\
    --output-dir "checkpoints/seed_${SEED}/best" \\
    --base-model-name ${MODEL_NAME} \\
    --r 8 --alpha 16 --dropout 0.0 \\
    > ~/export_lora.log 2>&1
echo EXPORT_LORA_OK >> ~/pipeline.log
gcs_rsync "checkpoints/seed_${SEED}/" "checkpoints/"

# Stage 4/5 cleanup: each eval_healthbench.py call spins up a vLLM-TPU
# Docker container (inference + grader) and may write merged base+LoRA
# scratch checkpoints under ~/bodhi_merged_*. Without cleanup, leftover
# containers hold the TPU and the merged dirs can fill the boot disk.
# Called explicitly after Stages 4 and 5, and on EXIT so partial
# failures (preempt, OOM, killed shell) still clean up.
cleanup_eval() {
    sudo docker ps --filter ancestor=vllm/vllm-tpu -q \\
        | xargs -r sudo docker stop 2>/dev/null || true
    rm -rf ~/bodhi_merged_* 2>/dev/null || true
    # Belt-and-suspenders: the Stage-3 sidecar trap also kills SIDECAR_PID,
    # but if we replace that trap (below) we still want this guarantee.
    [ -n "\${SIDECAR_PID:-}" ] && kill \${SIDECAR_PID} 2>/dev/null || true
}
trap cleanup_eval EXIT

echo "--- 4/4 eval 4 configs (base/lora x wrapper/no-wrapper) ---" | tee -a ~/pipeline.log
mkdir -p "eval/seed_${SEED}"
# scripts/export_tunix_lora_to_peft.py writes a PEFT-format adapter to
# checkpoints/seed_<N>/best/, so the existing vllm-tpu LoRA eval path is
# unchanged from launch_5seeds.sh.
LORA_DIR="checkpoints/seed_${SEED}/best"

# Per-seed bootstrap eval draw (issue #60): each seed gets its own random
# 200-prompt subset of the 1000 HealthBench Hard prompts.
SEED_IDS="data/raw/hard_seed_${SEED}.json"
\${PY} -u scripts/make_bootstrap_eval_ids.py \\
    --healthbench-jsonl data/raw/healthbench_hard.jsonl \\
    --seed ${SEED} \\
    --output "\$SEED_IDS" >> ~/pipeline.log 2>&1

# run_eval: write \$1.json under \$2 graded by \$3, with model+wrapper args from \$4.
# Used by the primary pass (out_dir=eval/seed_<N>, grader=Llama). When
# SECOND_GRADER_MODEL is set on the local launcher (literally baked into
# this heredoc), passes --secondary-grader-model so a second grader
# (e.g. Qwen2.5-14B) re-grades the SAME generated responses in-process
# and appends the result to secondary_grader_runs[] in the output JSON.
# This replaces the older Stage-4b "re-run run_eval with a different
# --grader-model" pattern that doubled inference cost by regenerating
# every response under the secondary grader. With
# --secondary-grader-model the generation runs once, the primary grader
# scores, and the secondary grader re-grades the existing in-memory
# responses (Pass 2b in eval_healthbench.py:469-517). Saves ~50% of
# Stage 4 wall when the cross-grader pass is enabled.
# Skips if the output already exists so preempt-resume picks up where
# it left off.
run_eval() {
    local name="\$1" out_dir="\$2" grader="\$3" args="\$4"
    local out="\${out_dir}/\${name}.json"
    if [ -s "\$out" ]; then
        echo "[\$name @ \$grader] already exists, skipping" >> ~/pipeline.log
        return 0
    fi
    # Build the optional --secondary-grader-model flag once. SECOND_GRADER_MODEL
    # is local-side \${VAR} expansion: empty -> sec_grader_flag stays empty,
    # non-empty -> baked literally into the run_eval invocation below.
    local sec_grader_flag=""
    if [ -n "${SECOND_GRADER_MODEL}" ]; then
        sec_grader_flag="--secondary-grader-model ${SECOND_GRADER_MODEL}"
    fi
    echo "--- eval \$name @ \$grader\${sec_grader_flag:+ (+ ${SECOND_GRADER_MODEL})} ---" | tee -a ~/pipeline.log
    # shellcheck disable=SC2086
    if \${PY} -u scripts/eval_healthbench.py \$args \\
            --sample-ids "\$SEED_IDS" \\
            --grader-model "\$grader" \\
            \$sec_grader_flag \\
            --output "\$out" \\
            ${_EVAL_MAX_FLAG} \\
            --seed ${SEED} >> ~/eval.log 2>&1; then
        return 0
    fi
    echo "eval \$name FAILED" >> ~/pipeline.log
    return 1
}

PRIMARY_DIR="eval/seed_${SEED}"
PRIMARY_GRADER="meta-llama/Llama-3.1-8B-Instruct"
# Track per-config failures so EVAL_OK is only written when all 4 graded
# successfully. Capture each call's exit status without letting set -e
# abort the rest of the pass; partial eval results are still worth saving.
#
# cleanup_eval between configs is load-bearing for disk: each LoRA config
# materialises ~bodhi_merged_<seed>/ (8GB+ for 4B base+LoRA). Without
# cleanup, after 4 configs the merged dirs accumulate to >32GB on a
# ~100GB v6e boot disk that already has Docker images, deps, and HF
# weights staged. Stop the vllm container + delete merged scratch
# between every config so disk free stays roughly constant.
eval_fail_count=0
run_eval "base_no_wrapper"  "\$PRIMARY_DIR" "\$PRIMARY_GRADER" "--model ${MODEL_NAME}" || eval_fail_count=\$((eval_fail_count + 1))
gcs_rsync "eval/seed_${SEED}/" "eval/"
cleanup_eval
run_eval "base_bodhi"       "\$PRIMARY_DIR" "\$PRIMARY_GRADER" "--model ${MODEL_NAME} --use-bodhi" || eval_fail_count=\$((eval_fail_count + 1))
gcs_rsync "eval/seed_${SEED}/" "eval/"
cleanup_eval
run_eval "lora_no_wrapper"  "\$PRIMARY_DIR" "\$PRIMARY_GRADER" "--model ${MODEL_NAME} --lora-path \$LORA_DIR" || eval_fail_count=\$((eval_fail_count + 1))
gcs_rsync "eval/seed_${SEED}/" "eval/"
cleanup_eval
run_eval "lora_bodhi"       "\$PRIMARY_DIR" "\$PRIMARY_GRADER" "--model ${MODEL_NAME} --lora-path \$LORA_DIR --use-bodhi" || eval_fail_count=\$((eval_fail_count + 1))
gcs_rsync "eval/seed_${SEED}/" "eval/"
cleanup_eval
if [ "\$eval_fail_count" -eq 0 ]; then
    echo EVAL_OK >> ~/pipeline.log
else
    echo "EVAL_FAILED (\$eval_fail_count/4 configs failed)" >> ~/pipeline.log
fi

# Stage 4b: cross-grader marker (data is already in-place).
# The secondary grader pass is in-process via run_eval's
# --secondary-grader-model flag (above). Each per-config JSON in
# PRIMARY_DIR now has a secondary_grader_runs[0] block holding the
# secondary grader's per-prompt scores; the per-config primary vs
# secondary correlation is computed post-hoc by
# scripts/analysis/cross_grader_post_hoc.py (workshop appendix path —
# does not need TPU compute, just reads the JSONs). The launcher just
# emits the marker so the dashboard / parser can confirm the in-place
# secondary pass actually wrote the runs.
if [ -n "${SECOND_GRADER_MODEL}" ]; then
    echo "--- 4b/5 cross-grader scores captured in secondary_grader_runs[] ---" | tee -a ~/pipeline.log
    # Sanity check: at least one of the 4 configs must have a non-empty
    # secondary_grader_runs[]. Fail loud if all four are empty (means the
    # --secondary-grader-model flag silently no-op'd, e.g. missing token).
    sec_ok=0
    for cfg in base_no_wrapper base_bodhi lora_no_wrapper lora_bodhi; do
        if [ -s "\${PRIMARY_DIR}/\${cfg}.json" ] && \\
           \${PY} -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('secondary_grader_runs') else 1)" \\
                "\${PRIMARY_DIR}/\${cfg}.json" 2>/dev/null; then
            sec_ok=\$((sec_ok + 1))
        fi
    done
    if [ "\$sec_ok" -ge 1 ]; then
        echo "  secondary_grader_runs present on \$sec_ok/4 configs (grader=${SECOND_GRADER_MODEL})" >> ~/pipeline.log
        echo XGRADER_OK >> ~/pipeline.log
    else
        echo "  WARN: 0/4 configs have secondary_grader_runs — second grader may have failed silently" >> ~/pipeline.log
        echo "XGRADER_FAILED" >> ~/pipeline.log
    fi
    gcs_rsync "eval/seed_${SEED}/" "eval/"
fi

# Stage 5: epistemic-virtue grading.
# eval_epistemic.py grades the same 4 response files on BODHI epistemic
# virtues (uncertainty acknowledgment, active inquiry, abstention, etc.)
# independent of HealthBench rubric correctness, answering the
# "do humility-trained outputs actually exhibit humility?" question that
# rubric scores can't. Same Llama-3.1-8B-Instruct grader as Stage 4 to
# keep methodology consistent. Skipped if any of the 4 input JSONs is
# missing (i.e. a Stage 4 config failed earlier: the eval_epistemic
# CLI requires real response files, not empty ones).
#
# Cleanup before Stage 5: free the TPU/Docker state + merged-LoRA scratch
# from Stage 4 (and Stage 4b if it ran) so eval_epistemic spins up a
# clean vLLM container instead of contending with the previous one.
cleanup_eval

echo "--- 5/5 epistemic virtue eval ---" | tee -a ~/pipeline.log
# Require ALL 4 Stage-4 outputs (base/lora x wrapper/no-wrapper) before
# Stage 5 runs. Earlier code was "run eval_epistemic on whatever subset
# exists, emit EPISTEMIC_OK on success" — which let a partial Stage-4
# failure produce a terminal success marker for the seed. The 4
# conditions are not redundant: the analysis compares them pairwise to
# isolate the LoRA adapter's vs the BODHI wrapper's contribution. A 1-,
# 2-, or 3-condition seed silently corrupts that comparison even though
# the dashboard sees EPISTEMIC_OK.
EPISTEMIC_INPUTS=()
EPISTEMIC_MISSING=()
for cfg in base_no_wrapper base_bodhi lora_no_wrapper lora_bodhi; do
    if [ -s "eval/seed_${SEED}/\${cfg}.json" ]; then
        EPISTEMIC_INPUTS+=("eval/seed_${SEED}/\${cfg}.json")
    else
        EPISTEMIC_MISSING+=("\${cfg}")
    fi
done
if [ \${#EPISTEMIC_MISSING[@]} -gt 0 ]; then
    # Some Stage-4 condition didn't land. Don't write EPISTEMIC_OK; the
    # dashboard parser treats a missing marker as "in progress / failed",
    # which is the honest state here. List the missing configs so the
    # post-mortem can find which Stage-4 invocation needs fixing.
    echo "Stage 4 incomplete: missing \${EPISTEMIC_MISSING[*]}; refusing to run eval_epistemic.py (would produce a partial result reported as success)" >> ~/pipeline.log
elif [ -s "eval/seed_${SEED}/epistemic_scores.json" ]; then
    # Resume path: a prior incarnation of this seed already produced the
    # output (pulled back from GCS) AND we just verified all 4 Stage-4
    # outputs are present. The work succeeded; mark done.
    echo "epistemic_scores.json already exists (all 4 Stage-4 inputs present), skipping" >> ~/pipeline.log
    echo EPISTEMIC_OK >> ~/pipeline.log
else
    # The only path that genuinely runs eval_epistemic.py. Write the marker
    # only on a clean exit; on failure the explicit FAILED line goes to
    # pipeline.log without EPISTEMIC_OK so the dashboard sees "not done".
    if \${PY} -u scripts/eval_epistemic.py \\
            --response-files "\${EPISTEMIC_INPUTS[@]}" \\
            --grader-model meta-llama/Llama-3.1-8B-Instruct \\
            --output "eval/seed_${SEED}/epistemic_scores.json" \\
            --seed ${SEED} >> ~/eval.log 2>&1; then
        echo EPISTEMIC_OK >> ~/pipeline.log
    else
        echo "eval_epistemic FAILED" >> ~/pipeline.log
    fi
fi
gcs_rsync "eval/seed_${SEED}/" "eval/"

# End-of-Stage-5 cleanup: stop any lingering vLLM-TPU container and
# wipe merged-base+LoRA scratch dirs. The EXIT trap also fires
# cleanup_eval, but calling it explicitly here means cleanup happens
# before the "pipeline complete" line prints (success-path ordering).
# cleanup_eval is idempotent so the second EXIT-trap call is a no-op.
cleanup_eval

echo "=== seed ${SEED} pipeline complete ===" >> ~/pipeline.log
REMOTE
}

for ((i=0; i<N_SEEDS; i++)); do
    SEED="${SEED_ARR[$i]}"
    ZONE="${ZONES[$i]}"
    VM_NAME="${VM_NAMES[$i]}"
    # First seed in $SEEDS is the Stage-1 leader. The other 4 are
    # followers that wait for the leader's raw_traces.jsonl to land in
    # GCS (saves ~5h x 4 = 20h of duplicated trace generation).
    if [ "$i" -eq 0 ]; then IS_LEADER=1; else IS_LEADER=0; fi
    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    LOG="${SEED_DIR}/launch.log"

    echo "Launching $VM_NAME (seed $SEED, $ZONE, leader=$IS_LEADER)..."

    (
        set -euo pipefail

        # Per-VM helpers: closures over $VM_NAME / $ZONE / $LOG.
        log()  { printf '[%s %s] %s\n' "$(date -u +%H:%M:%S)" "$VM_NAME" "$*" | tee -a "$LOG"; }

        try_create() {
            # Acquire one v6e-8 spot in $ZONE with capacity-error retries.
            # Returns 0 on success, 1 if we burn through all retries.
            # No data disk: setup_tpu.sh redirects HF cache to /dev/shm
            # (tmpfs, ~700 GB on v6e-8 hosts) so the 100 GB boot disk doesn't
            # ENOSPC when medgemma+qwen weights all sit in cache simultaneously.
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
            #
            # `|| log "..."` is load-bearing under `set -e`: a transient
            # IAP failure here would tear down the parent subshell before
            # the daemon-launch + retry path can run. The daemon won't
            # start if the tokens didn't push (the heredoc fails with
            # "HF_TOKEN missing from ~/.bohdi-env"), but wait_for_completion
            # will see that as DIED and route to the correct preempt /
            # non-preempt classification.
            printf '%s\n%s\n' "$GH_TOKEN" "$HF_TOKEN" \
                | gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                    --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                    --command='read -r G; read -r H; umask 077; { echo "GH_TOKEN=$G"; echo "HF_TOKEN=$H"; } > ~/.bohdi-env; chmod 600 ~/.bohdi-env' \
                    >>"$LOG" 2>&1 || log "push_tokens ssh failed (transient IAP, daemon may fail to start)"
        }

        launch_pipeline_detached() {
            # Stage the heredoc as a local temp file, scp it to the VM,
            # then ssh to start it as a fully-detached background daemon.
            # nohup + setsid + < /dev/null + & + disown together make the
            # process immune to the SIGHUP that fires when the IAP tunnel
            # between this launcher and the VM drops (which happens
            # routinely on multi-hour TPU jobs - long SSH sessions over
            # IAP are not a supported pattern). Stdout + stderr land in
            # ~/run_pipeline.log on the VM. Returns when the launch SSH
            # returns (typically <10s); the pipeline keeps running on
            # the VM independently from there on.
            #
            # We use scp + ssh rather than piping the heredoc body into
            # ssh's stdin because gcloud-ssh through IAP does not
            # reliably forward stdin to the remote --command (the
            # short-lived push_tokens path uses 'read' which appears to
            # work, but a longer 'cat > file' path observed empty input
            # in the live run, leaving run_pipeline.sh as a 0-byte file).
            local remote_cmd
            remote_cmd="$(build_remote_cmd "$SEED" "$IS_LEADER")"
            local local_script="${SEED_DIR}/run_pipeline.sh"
            printf '%s' "$remote_cmd" > "$local_script"
            chmod +x "$local_script"
            # The `|| log "..."` guards are load-bearing under `set -e`:
            # a transient IAP failure (4003 'failed to connect to backend')
            # in either gcloud call would otherwise tear down the parent
            # subshell BEFORE wait_for_completion could see whether the
            # daemon actually started. We log the failure and fall through
            # to wait_for_completion, which will probe via short SSH and
            # detect DIED/UNREACHABLE/PREEMPTED on its own.
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "$local_script" "${VM_NAME}:~/run_pipeline.sh" \
                >>"$LOG" 2>&1 || log "scp run_pipeline.sh failed (will retry via wait/probe)"
            gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                --command='chmod +x ~/run_pipeline.sh && nohup setsid bash ~/run_pipeline.sh > ~/run_pipeline.log 2>&1 < /dev/null & disown; echo "daemon launched, pid=$!"' \
                >>"$LOG" 2>&1 || log "daemon launch ssh failed (will retry via wait/probe)"
        }

        probe_status() {
            # Short SSH probe (~5-10s). Echoes one of:
            #   DONE         - EPISTEMIC_OK in pipeline.log (Stage 5 finished)
            #   RUNNING      - daemon process still alive
            #   DIED         - daemon gone but no terminal marker (real failure)
            #   UNREACHABLE  - SSH itself failed (return code via stdout)
            # Looks for EPISTEMIC_OK rather than EVAL_OK because Stage 5
            # is the last stage; an EVAL_OK without EPISTEMIC_OK means
            # we are partway through but not done.
            local out
            out=$(gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                --command='if grep -q "^EPISTEMIC_OK" ~/pipeline.log 2>/dev/null; then
                    echo DONE
                elif pgrep -f "run_pipeline.sh" > /dev/null 2>&1; then
                    echo RUNNING
                else
                    echo DIED
                fi' 2>/dev/null) || { echo UNREACHABLE; return; }
            echo "$out" | tr -d "[:space:]"
        }

        wait_for_completion() {
            # Poll probe_status every $POLL_INTERVAL_S seconds until DONE,
            # DIED, or PREEMPTED. Tolerates a few consecutive UNREACHABLE
            # probes (transient IAP glitches) before checking VM state.
            #   returns 0 -> DONE (EPISTEMIC_OK)
            #   returns 1 -> PREEMPTED (need to reacquire)
            #   returns 2 -> DIED (non-preempt failure - daemon exited
            #                without writing EPISTEMIC_OK)
            local POLL_INTERVAL_S="${POLL_INTERVAL_S:-90}"
            local probe_fail_count=0
            local probe_fail_max=5
            local last_status="?"
            while :; do
                sleep "$POLL_INTERVAL_S"
                local s
                s=$(probe_status)
                if [ "$s" != "$last_status" ]; then
                    log "probe: $s"
                    last_status="$s"
                fi
                case "$s" in
                    DONE) return 0 ;;
                    DIED)
                        # Could be a real Python failure OR a preempt that
                        # wiped the boot disk + the daemon. Check VM state
                        # to disambiguate before declaring non-preempt.
                        local state
                        state=$(vm_state)
                        if [ "$state" = "PREEMPTED" ] || [ "$state" = "MISSING" ]; then
                            return 1
                        fi
                        return 2
                        ;;
                    UNREACHABLE)
                        probe_fail_count=$((probe_fail_count + 1))
                        if [ "$probe_fail_count" -ge "$probe_fail_max" ]; then
                            local state
                            state=$(vm_state)
                            log "$probe_fail_max consecutive UNREACHABLE probes; vm state=$state"
                            if [ "$state" = "PREEMPTED" ] || [ "$state" = "MISSING" ]; then
                                return 1
                            fi
                            # VM is READY but we can't talk to it; treat
                            # as a transient and keep polling. Reset count
                            # so a brief outage does not give up.
                            probe_fail_count=0
                        fi
                        ;;
                    RUNNING)
                        probe_fail_count=0
                        ;;
                esac
            done
        }

        vm_state() {
            gcloud compute tpus tpu-vm describe "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" \
                --format="value(state)" 2>/dev/null \
                || echo "MISSING"
        }

        scp_back() {
            # Best-effort copy of checkpoints + eval JSONs + daemon-side logs
            # to ./results_tunix/seed_<N>/. Never fails the parent shell.
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/checkpoints/seed_${SEED}" "$SEED_DIR/" \
                >>"$LOG" 2>&1 || log "  (no checkpoints to copy)"
            gcloud alpha compute tpus tpu-vm scp --recurse \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/bohdi-lora/eval/seed_${SEED}" "$SEED_DIR/" \
                >>"$LOG" 2>&1 || log "  (no eval to copy)"
            # Pull the daemon-side logs so post-mortem doesn't require a live
            # SSH back to the VM (which is often already deleted by the time
            # we look). Each scp tolerates a missing file independently;
            # e.g. setup.log won't exist if setup_tpu.sh never ran.
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/pipeline.log" "${SEED_DIR}/pipeline.log" \
                >>"$LOG" 2>&1 || true
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/setup.log" "${SEED_DIR}/setup.log" \
                2>/dev/null || true
            gcloud alpha compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
                "${VM_NAME}:~/run_pipeline.log" "${SEED_DIR}/run_pipeline.log" \
                2>/dev/null || true
        }

        delete_vm() {
            gcloud compute tpus tpu-vm delete "$VM_NAME" \
                --zone="$ZONE" --project="$PROJECT" --quiet \
                >>"$LOG" 2>&1 || true
        }

        # Trap on EXIT: runs after the outer retry loop ends, no matter
        # how (success / preempt-give-up / SIGINT). Always tries to copy
        # whatever lives on the current VM and delete it.
        cleanup() {
            log "cleanup: copy results + delete VM"
            scp_back
            delete_vm
            log "cleaned up"
        }
        trap cleanup EXIT

        # outer retry loop.
        # The pipeline runs as a detached daemon on the VM (started by
        # launch_pipeline_detached + nohup + setsid). The local launcher
        # holds NO long-running SSH session; instead it polls the daemon
        # every POLL_INTERVAL_S seconds via short SSH connections that
        # are insulated from IAP tunnel drops. This is the structural fix
        # for the failure mode where multi-hour heredoc-over-IAP sessions
        # would routinely drop, kill the heredoc on the remote, and the
        # launcher would mis-classify the drop as a "non-preempt failure"
        # and delete the VM.
        preempt_attempt=0
        while :; do
            if ! try_create; then
                # Exit non-zero so the parent's `wait` sees this seed as a
                # real failure, not a successful no-op. Without this the
                # launcher itself reports exit 0 even when no work happened.
                log "exhausted create retries, giving up on this seed"
                exit 1
            fi
            push_tokens
            launch_pipeline_detached
            # Capture rc explicitly: `wait_for_completion; rc=$?` is broken
            # under `set -e` because a non-zero return from the function
            # (the PREEMPTED / DIED paths) kills the subshell *before*
            # `rc=$?` runs, so the case below, and the preempt-retry
            # branch, never fires. v18 hit this: VM was preempted at
            # 3min uptime, probe correctly detected PREEMPTED, but the
            # outer loop exited via the EXIT trap with no reacquisition.
            # The `if`-form is a tested context so set -e leaves it alone.
            if wait_for_completion; then rc=0; else rc=$?; fi
            case $rc in
                0)
                    log "pipeline complete (EPISTEMIC_OK)"
                    break
                    ;;
                1)
                    # PREEMPTED or MISSING - reacquire in same zone.
                    preempt_attempt=$((preempt_attempt + 1))
                    if [ "$preempt_attempt" -ge "$MAX_PREEMPT_RETRIES" ]; then
                        log "hit MAX_PREEMPT_RETRIES=$MAX_PREEMPT_RETRIES, giving up"
                        break
                    fi
                    log "preempted, reacquiring (attempt $preempt_attempt/$MAX_PREEMPT_RETRIES)"
                    # Most work is in GCS already (sidecar uploads); SCP
                    # is best-effort for anything not yet rsync'd.
                    scp_back
                    delete_vm
                    sleep 30
                    continue
                    ;;
                2)
                    # Daemon exited without EPISTEMIC_OK and the VM is
                    # still READY - a real Python/pipeline failure (not
                    # a tunnel drop, which the daemon survives now).
                    log "daemon exited without completion marker, non-preempt failure, not retrying"
                    break
                    ;;
            esac
        done
    ) &

    echo "$!" >> "$PID_FILE"
    sleep 4   # stagger gcloud creates a bit so we don't hammer the API
done

echo
echo "All $N_SEEDS jobs spawned. PIDs: $(cat "$PID_FILE")"
echo "Open the dashboard at http://localhost:8000 to watch progress."
echo
echo "Waiting for all VMs to finish..."

# Wait per-PID and aggregate exit codes. A bare `wait` would mask any
# subshell that exited non-zero (e.g. exhausted create retries) and let
# the launcher itself report success. Pair each PID with its seed/VM so
# the failure summary names the actual seed instead of just a PID.
SUBSHELL_PIDS=()
while IFS= read -r _pid_line; do
    [ -n "$_pid_line" ] && SUBSHELL_PIDS+=("$_pid_line")
done < "$PID_FILE"
overall_rc=0
failed_seeds=()
for ((i=0; i<N_SEEDS; i++)); do
    pid="${SUBSHELL_PIDS[$i]:-}"
    if [ -z "$pid" ]; then continue; fi
    # Capture rc on its own line: in `if ! wait; then rc=$?` the $? would
    # be 0 (the negated-test result), not the wait exit code we want.
    rc=0
    wait "$pid" || rc=$?
    if [ "$rc" -ne 0 ]; then
        overall_rc=1
        failed_seeds+=("seed=${SEED_ARR[$i]} vm=${VM_NAMES[$i]} rc=$rc")
        echo "ERROR: ${VM_NAMES[$i]} (seed ${SEED_ARR[$i]}) exited $rc" >&2
    fi
done

echo
echo "All seeds done. Results in $RESULTS_DIR/seed_*/"
ls -la "$RESULTS_DIR" 2>/dev/null || true

if [ "$overall_rc" -ne 0 ]; then
    echo
    echo "FAILED seeds:" >&2
    for f in "${failed_seeds[@]}"; do echo "  $f" >&2; done
    exit "$overall_rc"
fi
