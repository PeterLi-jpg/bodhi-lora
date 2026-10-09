# _common.sh: settings and helpers shared by run_cell.sh and run_baselines.sh.
# Source it; do not run it. It needs MODEL and BENCH set, and it cd's to the
# repo root.
#
# Written for bash 3.2 as well as bash 4+, so the dry-run tests also run on a
# Mac: no mapfile, no associative arrays, and every array that can be empty is
# expanded as ${a[@]+"${a[@]}"}.
# shellcheck disable=SC2034  # the arrays set here are read by the sourcing script

# ---- command-line flags -----------------------------------------------------------
ALLOW_SAME_FAMILY="${ALLOW_SAME_FAMILY:-0}"
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
    case "$arg" in
        --allow-same-family) ALLOW_SAME_FAMILY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg (expected --allow-same-family or --dry-run)"; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1     # repo root
REPO="$PWD"

# Two Python envs: training needs transformers 4.57.6, but vLLM 0.9.2 serving
# needs transformers < 4.54 (its ovis config registers "aimv2", which
# transformers 4.54+ also defines). requirements.txt builds .venv and
# requirements-infer.txt builds .venv-infer.
TRAIN_PY="${TRAIN_PY:-$REPO/.venv/bin/python}"
INFER_PY="${INFER_PY:-$REPO/.venv-infer/bin/python}"

MODEL="${MODEL:?set MODEL}"
BENCH="${BENCH:?set BENCH (healthbench|medqa|medquad|medicationqa|medmcqa|chatdoctor)}"
SEEDS="${SEEDS:-42 7 13 99 101}"
PROTOCOL="${PROTOCOL:-bodhi}"                        # bodhi | ask_first

# ---- profile ----------------------------------------------------------------------
# PROFILE sets the defaults below; any single variable can still be set by hand.
#   submitted        (default) the eval settings of the submitted MedGemma cell
#                    (legacy/modal_eval.py) and one eval-set policy for every
#                    benchmark: a held-out pool, and a separate random 200 per seed.
#   rebuttal-posted  the settings the posted rebuttal cells ran with (run_cell.sh
#                    at 61dada9), kept so those cells stay describable and
#                    re-runnable. legacy/README.md lists each difference.
PROFILE="${PROFILE:-submitted}"
case "$PROFILE" in
    submitted)
        : "${GRADER_MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
        : "${MAX_MODEL_LEN_EVAL:=4096}"
        : "${MAX_NUM_SEQS_EVAL:=32}"
        : "${EVAL_CONCURRENCY:=32}"
        : "${REQUEST_TIMEOUT:=900}"
        : "${DRAIN_SECONDS:=0}"
        : "${EVAL_IDS_POLICY:=per-seed}"     # per-seed | fixed
        : "${SPLIT_POLICY:=per-seed}"        # per-seed | once (train/val split)
        : "${SELECT:=seeded}"                # seeded | head (benchmark pool rows)
        : "${HOLDOUT_METHOD:=seeded}"        # seeded | head
        : "${HOLDOUT_N:=1000}"               # held-out pool (HealthBench Hard has 1000)
        : "${TAU_RULE:=fixed}"               # fixed | percentile | legacy-auto
        : "${CONTAMINATION_PROBE:=1}"
        ;;
    rebuttal-posted)
        : "${GRADER_MODEL:=NousResearch/Meta-Llama-3.1-8B-Instruct}"
        : "${MAX_MODEL_LEN_EVAL:=8192}"
        : "${MAX_NUM_SEQS_EVAL:=64}"
        : "${EVAL_CONCURRENCY:=16}"
        : "${REQUEST_TIMEOUT:=180}"
        : "${DRAIN_SECONDS:=30}"
        : "${EVAL_IDS_POLICY:=fixed}"
        : "${SPLIT_POLICY:=once}"
        : "${SELECT:=head}"
        : "${HOLDOUT_METHOD:=head}"
        : "${HOLDOUT_N:=200}"
        : "${TAU_RULE:=legacy-auto}"
        : "${CONTAMINATION_PROBE:=0}"
        ;;
    *) echo "unknown PROFILE: $PROFILE (submitted | rebuttal-posted)"; exit 2 ;;
esac
case "$TAU_RULE" in
    fixed|percentile|legacy-auto) ;;
    *) echo "unknown TAU_RULE: $TAU_RULE (fixed | percentile | legacy-auto)"; exit 2 ;;
esac
FILTER_MODEL="${FILTER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-}"                 # empty: the configs/families.yaml pin
GRADER_REVISION="${GRADER_REVISION:-}"
FILTER_REVISION="${FILTER_REVISION:-}"
GEN_MAX="${GEN_MAX:-4200}"                           # prompts to generate traces over
EVAL_N="${EVAL_N:-200}"                              # eval prompts per seed
SELECT_SEED="${SELECT_SEED:-0}"
HOLDOUT_SEED="${HOLDOUT_SEED:-0}"
MIN_SCORE="${MIN_SCORE:-0.4}"                        # fixed tau (the paper's value)
TAU_PERCENTILE="${TAU_PERCENTILE:-22}"               # used by percentile and legacy-auto
# EVAL_BASE_QUANT=match serves the base in 4-bit NF4 for every eval condition
# when the training config is QLoRA (model.quantization: 4bit), so the adapter
# is evaluated on the base it was trained on. bf16 is what the posted
# Mistral-24B cells did (trained on 4-bit, evaluated on bf16).
EVAL_BASE_QUANT="${EVAL_BASE_QUANT:-bf16}"

RESULTS_ROOT="${RESULTS_ROOT:-results_rebuttal}"
DATA_RAW="${DATA_RAW:-data/raw}"
LOG_DIR="${LOG_DIR:-logs}"
TAG="$(echo "${MODEL}" | tr '/:.' '___')"
WORK="${RESULTS_ROOT}/${BENCH}__${TAG}"
[ "$PROTOCOL" != "bodhi" ] && WORK="${WORK}__${PROTOCOL}"
SFT="${WORK}/sft"
mkdir -p "$WORK" "$SFT" "$DATA_RAW" "$LOG_DIR"

if [ "$DRY_RUN" != 1 ]; then
    # HF token: reuse the box's cached token if not exported (never printed).
    if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    fi
    : "${HF_TOKEN:?HF_TOKEN not set and no ~/.cache/huggingface/token found}"
fi

# vLLM: run the venv's GPU vLLM as a subprocess. The auto-detected docker path
# pulls a vllm/vllm-tpu image (TPU) that hangs on this GPU box. Pin tensor-parallel
# to 1 because each stage runs on a single CUDA_VISIBLE_DEVICES GPU; raise VLLM_TP
# to tensor-parallel generation/eval across several idle GPUs when they are free.
export BODHI_VLLM_MODE="${BODHI_VLLM_MODE:-subprocess}"
export BODHI_VLLM_ACCEL="${BODHI_VLLM_ACCEL:-gpu}"
export BODHI_VLLM_TP="${VLLM_TP:-1}"

# ---- GPU selection ------------------------------------------------------------------
# PIN_GPU (recommended when launching several cells at once): a space-separated list
# of GPU indices this cell owns, e.g. PIN_GPU="3 4". Auto-detection is RACY across
# concurrently-starting cells — vLLM takes ~2 min to allocate, so every cell launched
# in that window sees the same card as "idle" and they all pile onto it. Pinning
# removes the race; the caller is responsible for handing out disjoint sets.
#
# Without PIN_GPU we fall back to auto-detect (fine for a single cell at a time):
# idle = 0% util AND <2GB used, re-snapshotted per wave so we never grab a card
# another job has started using.
# Cooperative lease on the shared node (scripts/gpu_lease.sh + /data/GPUS.txt).
# Claim up front so another job can't take the same card during the ~2 min
# before vLLM allocates and the GPU still reads 0 MiB in nvidia-smi. Released
# automatically on exit; a SIGKILLed run is cleaned up by `gpu_lease.sh reconcile`.
LEASE="$REPO/scripts/gpu_lease.sh"
LEASE_OWNER="${LEASE_OWNER:-bohdi}"
if [ "$DRY_RUN" != 1 ] && [ -x "$LEASE" ] && [ -n "${PIN_GPU:-}" ]; then
    for g in $PIN_GPU; do
        CLAIM_PID=$$ bash "$LEASE" claim "$g" "$LEASE_OWNER" "rebuttal $BENCH" \
            || { echo "GPU $g is leased by someone else — aborting"; exit 1; }
    done
    trap 'for g in $PIN_GPU; do bash "$LEASE" release "$g" >/dev/null 2>&1; done' EXIT
fi

idle_gpus() {
    if [ "$DRY_RUN" = 1 ]; then
        echo 0
        return
    fi
    if [ -n "${PIN_GPU:-}" ]; then
        printf '%s\n' $PIN_GPU
        return
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<2000 {print $1}'
}

# run_gpu GPU CMD...: run one GPU stage on that GPU, or print it in a dry run.
run_gpu() {
    local gpu="$1"; shift
    if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY-RUN [gpu %s]:' "$gpu"; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" BODHI_VLLM_PORT="$((8000 + gpu))" "$@"
}

# run_gpu_py GPU SCRIPT ARGS...: like run_gpu for an inference entry point. In a
# dry run the script itself runs with --dry-run on the CPU (stub engines), so the
# prompt, wrapper and grading code is exercised and a *.dryrun.json is written.
run_gpu_py() {
    local gpu="$1"; shift
    if [ "$DRY_RUN" = 1 ]; then
        "$INFER_PY" "$@" --dry-run
        return
    fi
    run_gpu "$gpu" "$INFER_PY" "$@"
}

# ---- grader-family guard (EVAL-7) ------------------------------------------------
# The base must not share a family with the filter or the evaluator
# (configs/families.yaml). --allow-same-family waives it; the manifest records it.
family_guard() {
    local flag=()
    [ "$ALLOW_SAME_FAMILY" = 1 ] && flag=(--allow-same-family)
    "$INFER_PY" -m bodhi_lora.manifest check-family --base "$MODEL" \
        --filter "$FILTER_MODEL" --evaluator "$GRADER_MODEL" \
        ${flag[@]+"${flag[@]}"} --json-out "${WORK}/family_check.json" \
        || { echo "=== refused by the grader-family guard; see above ==="; exit 3; }
}

# ---- benchmark data, held-out pool and per-seed eval ids --------------------------
# Sets BENCH_JSONL, HOLDOUT_IDS, GEN_SRC, FILTER_DATA, EXCLUDE, EVAL_BENCH,
# AGG_DATA, EPI_DATA. Builds missing files on the CPU.
prepare_data() {
    if [ "$BENCH" = "healthbench" ]; then
        BENCH_JSONL="${DATA_RAW}/healthbench_hard.jsonl"
        # The per-seed eval draw and the leakage gate read the HealthBench
        # files before stage 1 would download them. generate_traces.py reads
        # data/raw itself, so keep DATA_RAW=data/raw for HealthBench cells.
        if [ ! -s "$BENCH_JSONL" ] || [ ! -s "${DATA_RAW}/healthbench.jsonl" ]; then
            "$INFER_PY" scripts/download_data.py
        fi
        # The pool is all 1000 Hard prompts; training excludes every one of them.
        HOLDOUT_IDS="$BENCH_JSONL"
        EXCLUDE=("${DATA_RAW}/healthbench_hard.jsonl" "${DATA_RAW}/hard_200_sample_ids.json")
        GEN_SRC=(--datasets healthbench_hard healthbench --exclude-ids "${EXCLUDE[@]}")
        FILTER_DATA=(--healthbench-data "${DATA_RAW}/healthbench_hard.jsonl" "${DATA_RAW}/healthbench.jsonl")
        AGG_DATA=(--healthbench "${DATA_RAW}/healthbench_hard.jsonl" "${DATA_RAW}/healthbench.jsonl")
    else
        local name
        case "$BENCH" in
            medqa)        name=medqa_open ;;
            medquad)      name=medquad ;;
            medicationqa) name=medicationqa ;;
            medmcqa)      name=medmcqa_open ;;
            chatdoctor)   name=chatdoctor ;;
            *) echo "unknown BENCH: $BENCH"; exit 1 ;;
        esac
        if [ "$SELECT" = head ]; then
            # The posted cells: first rows in HF order, file name without a suffix.
            local n_total=$((GEN_MAX + HOLDOUT_N + 500))
            BENCH_JSONL="${DATA_RAW}/${name}.jsonl"
        else
            local n_total=$((GEN_MAX + HOLDOUT_N + 500))
            BENCH_JSONL="${DATA_RAW}/${name}_s${SELECT_SEED}_n${n_total}.jsonl"
        fi
        if [ "$HOLDOUT_METHOD" = head ]; then
            HOLDOUT_IDS="${WORK}/eval_ids.json"      # where the posted cells kept it
        else
            HOLDOUT_IDS="${BENCH_JSONL%.jsonl}_holdout${HOLDOUT_N}_h${HOLDOUT_SEED}.json"
        fi
        if [ ! -s "$BENCH_JSONL" ]; then
            "$INFER_PY" -m bodhi_lora.data.loaders --benchmark "$BENCH" \
                --out "$BENCH_JSONL" --max "$n_total" --select "$SELECT" --select-seed "$SELECT_SEED"
        fi
        if [ ! -s "$HOLDOUT_IDS" ]; then
            "$INFER_PY" -m bodhi_lora.data.loaders --benchmark "$BENCH" \
                --out "$BENCH_JSONL" --holdout-only --holdout-n "$HOLDOUT_N" \
                --holdout-method "$HOLDOUT_METHOD" --holdout-seed "$HOLDOUT_SEED" \
                --holdout-out "$HOLDOUT_IDS"
        fi
        EXCLUDE=("$HOLDOUT_IDS")
        GEN_SRC=(--dataset-files "$BENCH_JSONL" --exclude-ids "$HOLDOUT_IDS")
        FILTER_DATA=(--healthbench-data "$BENCH_JSONL")
        AGG_DATA=(--healthbench "$BENCH_JSONL")
    fi
    EVAL_BENCH=(--benchmark-jsonl "$BENCH_JSONL")
    EPI_DATA=(--benchmark-jsonl "$BENCH_JSONL")
    [ "$BENCH" = "healthbench" ] && EPI_DATA=(--healthbench-data "$BENCH_JSONL")

    # Per-seed eval ids. per-seed: a separate Random(seed).sample(EVAL_N) of the
    # pool (bodhi_lora/data/splits.py, the submitted cell's rule).
    # fixed: one set for every seed (the posted cells).
    mkdir -p "${WORK}/eval_ids"
    local s
    for s in $SEEDS; do
        if [ "$EVAL_IDS_POLICY" = per-seed ]; then
            if [ "$BENCH" = "healthbench" ]; then
                "$INFER_PY" -m bodhi_lora.data.splits --healthbench-jsonl "$BENCH_JSONL" \
                    --seed "$s" --size "$EVAL_N" --output "$(eval_ids_for_seed "$s")"
            else
                "$INFER_PY" -m bodhi_lora.data.splits --pool-ids "$HOLDOUT_IDS" \
                    --seed "$s" --size "$EVAL_N" --output "$(eval_ids_for_seed "$s")"
            fi
        fi
    done
}

eval_ids_for_seed() {   # $1 = seed -> path of that seed's eval-id file
    if [ "$EVAL_IDS_POLICY" = per-seed ]; then
        echo "${WORK}/eval_ids/seed_$1.json"
    elif [ "$BENCH" = "healthbench" ]; then
        echo "${DATA_RAW}/hard_200_sample_ids.json"
    else
        echo "$HOLDOUT_IDS"
    fi
}

# build_eval_flags: set EVAL_FLAGS and EPI_FLAGS, the flags that every
# eval_healthbench.py and eval_epistemic.py call of this cell shares.
build_eval_flags() {
    EVAL_QUANT=none
    case "$EVAL_BASE_QUANT" in
        bf16) ;;
        match)
            local q
            q="$("$INFER_PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["model"].get("quantization") or "")' "$CONFIG")"
            case "$q" in
                4bit) EVAL_QUANT=bitsandbytes ;;
                "") EVAL_QUANT=none ;;
                *) echo "EVAL_BASE_QUANT=match: vLLM in-flight quantization is 4-bit only; $CONFIG has $q"; exit 2 ;;
            esac ;;
        *) echo "unknown EVAL_BASE_QUANT: $EVAL_BASE_QUANT (bf16 | match)"; exit 2 ;;
    esac
    EVAL_FLAGS=(--grader-model "$GRADER_MODEL"
                --max-model-len "$MAX_MODEL_LEN_EVAL" --max-num-seqs "$MAX_NUM_SEQS_EVAL"
                --concurrency "$EVAL_CONCURRENCY" --request-timeout "$REQUEST_TIMEOUT"
                --drain-seconds "$DRAIN_SECONDS" --base-quantization "$EVAL_QUANT")
    EPI_FLAGS=(--grader-model "$GRADER_MODEL"
               --max-model-len "$MAX_MODEL_LEN_EVAL" --max-num-seqs "$MAX_NUM_SEQS_EVAL"
               --concurrency "$EVAL_CONCURRENCY" --request-timeout "$REQUEST_TIMEOUT")
    if [ -n "$MODEL_REVISION" ]; then
        EVAL_FLAGS+=(--model-revision "$MODEL_REVISION")
    fi
    if [ -n "$GRADER_REVISION" ]; then
        EVAL_FLAGS+=(--grader-revision "$GRADER_REVISION")
        EPI_FLAGS+=(--grader-revision "$GRADER_REVISION")
    fi
}

# What a seed changes in this cell (REPRO-8). Teacher traces are generated once
# per cell with greedy decoding, so no seed varies them.
seed_varies() {
    local parts="LoRA init and data order"
    [ "$SPLIT_POLICY" = per-seed ] && parts="$parts, train/val split"
    [ "$EVAL_IDS_POLICY" = per-seed ] && parts="$parts, eval prompt draw"
    echo "$parts; teacher traces and grades are shared by all seeds"
}

# build_settings: set SETTINGS_ARGS, the --set flags every manifest of this
# cell records.
build_settings() {
    SETTINGS_ARGS=()
    local k
    for k in PROFILE BENCH MODEL MODEL_REVISION CONFIG PROTOCOL FILTER_MODEL FILTER_REVISION \
             GRADER_MODEL GRADER_REVISION SEEDS GEN_MAX EVAL_N EVAL_IDS_POLICY SPLIT_POLICY \
             SELECT SELECT_SEED HOLDOUT_METHOD HOLDOUT_N HOLDOUT_SEED TAU_RULE MIN_SCORE \
             TAU_PERCENTILE MAX_MODEL_LEN_EVAL MAX_NUM_SEQS_EVAL EVAL_CONCURRENCY \
             REQUEST_TIMEOUT DRAIN_SECONDS EVAL_BASE_QUANT EVAL_QUANT ALLOW_SAME_FAMILY \
             DRY_RUN BENCH_JSONL HOLDOUT_IDS; do
        SETTINGS_ARGS+=(--set "$k=${!k:-}")
    done
    SETTINGS_ARGS+=(--set "SEED_VARIES=$(seed_varies)")
}
