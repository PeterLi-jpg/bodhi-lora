#!/usr/bin/env bash
# Run a queue of commands on one GPU of the shared node, one after the other.
#
#   gpu_queue.sh --gpu 3 --owner bohdi --purpose "BODHI R3-R7" \
#       --queue configs/queues/tier_a.txt --log-dir /data/PROJECTS/bohdi-queue \
#       [--name tier_a] [--repo DIR] [--after SESSION [--after-marker FILE]] \
#       [--bundle FILE --commit SHA [--keep-local PATH]... [--allow-untracked PATH]...] \
#       [--env FILE] [--lease-tool PATH] [--lease-file PATH]
#
# The steps, in order:
#   a. With --after: wait until that tmux session has exited, then read its exit marker.
#      Stop unless the marker's first line is 0.
#   b. Claim the GPU with the node's lease tool, with this shell's PID as the lease PID.
#      Stop if the card holds memory, or if the tool refuses: it never gives out a card
#      that another job holds a lease on.
#   c. With --bundle and --commit: fetch the bundle into the checkout and switch it to
#      the commit. Stop if the checkout has changes, or untracked files outside the
#      --allow-untracked paths (the run outputs under results/, for example). This
#      check also runs at the start, so a checkout that cannot switch fails at once.
#   d. With --env: source the run env file.
#   e. Run each line of the queue file in the checkout, pinned to the GPU, and stop at
#      the first line that fails.
#   f. On every exit: release the GPU and write the exit marker <log-dir>/<name>.exit.
#
# docs/GPU_QUEUE.md is the manual: the queue file format, the logs, the exit codes and
# how to chain one queue after another.
#
# Exit codes: 0 every step finished; 2 bad arguments or setup; 3 the queue it waited for
# did not finish; 4 the claim failed; 5 the checkout switch failed; 6 the env file
# failed; 7 the queue file is bad; 128+N a signal; any other code is the failed step's.
#
# Settings for the tests (the defaults suit the node):
#   GPU_QUEUE_POLL_S        seconds between checks of the --after session (60)
#   GPU_QUEUE_STOP_GRACE_S  seconds a step gets to stop after SIGINT (120)
#   GPU_QUEUE_MEM_WAIT_S    seconds to wait for the card to empty before claim and release (60)
#
# Everything runs inside functions and the last line calls main, so bash has read the
# whole file before it runs any of it. A checkout switch that rewrites this file
# therefore cannot change the script while it runs.

# shellcheck disable=SC2317,SC2329  # the exit and signal handlers run from traps

POLL_S="${GPU_QUEUE_POLL_S:-60}"
STOP_GRACE_S="${GPU_QUEUE_STOP_GRACE_S:-120}"
MEM_WAIT_S="${GPU_QUEUE_MEM_WAIT_S:-60}"
# The lease tool counts a card that uses less than this many MiB as empty.
EMPTY_MIB=2000

GPU="" OWNER="" PURPOSE="" QUEUE="" NAME="" LOG_DIR="" REPO="$PWD"
AFTER="" AFTER_MARKER="" BUNDLE="" COMMIT="" ENV_FILE=""
# The node's lease tool and the file it keeps the leases in (its own default).
LEASE_TOOL=/data/PROJECTS/bohdi/gpu_lease.sh
LEASE_FILE="${GPU_LEASE_FILE:-/home/nvidia/GPUS.txt}"
KEEP=() ALLOW=()

QLOG="" MARKER="" REASON="" STEP=0 CHILD_PID="" CLAIM_TRIED=0
SAVED=() OWN=() CMDS=()

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# One line to the queue log and to the terminal. The queue log holds only these lines,
# never a step's output, so it is safe to read while a cell runs (docs/GPU_QUEUE.md).
log() {
    local line
    line="$(ts) $*"
    [ -n "$QLOG" ] && printf '%s\n' "$line" >>"$QLOG"
    printf '%s\n' "$line"
}

# Run a command and log its output, each line with a timestamp. Returns its exit code.
logged() {
    local out rc line
    out="$("$@" 2>&1)"
    rc=$?
    if [ -n "$out" ]; then
        while IFS= read -r line; do
            log "  $line"
        done <<<"$out"
    fi
    return "$rc"
}

# Log a list (a command's output kept in a variable), at most 20 lines of it.
log_lines() {
    local line
    while IFS= read -r line; do
        log "  $line"
    done < <(printf '%s\n' "$1" | head -n 20)
}

die() {
    local code=$1
    shift
    REASON="$*"
    log "stop: $*"
    exit "$code"
}

abspath() {
    case $1 in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

parse_args() {
    while [ $# -gt 0 ]; do
        case $1 in
            -h | --help) usage; exit 0 ;;
        esac
        if [ $# -lt 2 ]; then
            echo "gpu_queue.sh: $1 needs a value" >&2
            exit 2
        fi
        case $1 in
            --gpu) GPU=$2 ;;
            --owner) OWNER=$2 ;;
            --purpose) PURPOSE=$2 ;;
            --queue) QUEUE=$2 ;;
            --name) NAME=$2 ;;
            --log-dir) LOG_DIR=$(abspath "$2") ;;
            --repo) REPO=$(abspath "$2") ;;
            --after) AFTER=$2 ;;
            --after-marker) AFTER_MARKER=$(abspath "$2") ;;
            --bundle) BUNDLE=$(abspath "$2") ;;
            --commit) COMMIT=$2 ;;
            --keep-local) KEEP+=("$2") ;;
            --allow-untracked) ALLOW+=("$2") ;;
            --env) ENV_FILE=$(abspath "$2") ;;
            --lease-tool) LEASE_TOOL=$(abspath "$2") ;;
            --lease-file) LEASE_FILE=$(abspath "$2") ;;
            *)
                echo "gpu_queue.sh: unknown argument $1" >&2
                usage >&2
                exit 2
                ;;
        esac
        shift 2
    done
}

# The checks that need no lease, so a mistake shows at once and not after hours of
# waiting for the --after session.
validate() {
    case $GPU in
        '' | *[!0-9]*) die 2 "--gpu must be a GPU index, got '$GPU'" ;;
    esac
    case $OWNER in
        '' | *[[:space:]]*) die 2 "--owner must be one word, got '$OWNER'" ;;
    esac
    [ -n "$PURPOSE" ] || die 2 "--purpose is required"
    [ -x "$LEASE_TOOL" ] || die 2 "lease tool $LEASE_TOOL is not an executable file"
    [ -d "$REPO" ] || die 2 "--repo $REPO is not a directory"
    if [ -n "$AFTER" ]; then
        command -v tmux >/dev/null 2>&1 || die 2 "--after needs tmux, and tmux is not on PATH"
        [ "$AFTER" != "$NAME" ] || die 2 "--after names this queue itself ($NAME)"
        [ -n "$AFTER_MARKER" ] || AFTER_MARKER="$LOG_DIR/$AFTER.exit"
    fi
    if [ -n "$BUNDLE" ] || [ -n "$COMMIT" ]; then
        [ -n "$BUNDLE" ] && [ -n "$COMMIT" ] || die 2 "--bundle and --commit go together"
        [ -f "$BUNDLE" ] || die 2 "bundle $BUNDLE not found"
        git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die 2 "$REPO is not a git checkout"
        check_clean "at the start"
        logged git -C "$REPO" bundle verify "$BUNDLE" || die 5 "git bundle verify failed for $BUNDLE"
    fi
    [ "${#KEEP[@]}${#ALLOW[@]}" = 00 ] || [ -n "$COMMIT" ] ||
        die 2 "--keep-local and --allow-untracked apply only to a checkout switch (--commit)"
    # A --keep-local path is relative to the checkout, or absolute inside it.
    local i
    for i in "${!KEEP[@]}"; do
        case ${KEEP[$i]} in
            "$REPO"/*) KEEP[i]=${KEEP[$i]#"$REPO"/} ;;
            /*) die 2 "--keep-local ${KEEP[$i]} is not inside the checkout $REPO" ;;
        esac
    done
    [ -z "$ENV_FILE" ] || [ -f "$ENV_FILE" ] || die 2 "env file $ENV_FILE not found"
}

# a. Wait for the session before this queue, then read its marker.
wait_for_session() {
    log "waiting for tmux session $AFTER to exit (a check every ${POLL_S}s)"
    # '=' asks tmux for this exact session name, not a prefix match.
    while tmux has-session -t "=$AFTER" 2>/dev/null; do
        sleep "$POLL_S"
    done
    log "tmux session $AFTER is gone"
    [ -f "$AFTER_MARKER" ] || die 3 "$AFTER left no exit marker at $AFTER_MARKER; not starting"
    local status
    status="$(head -n 1 "$AFTER_MARKER")"
    [ "$status" = 0 ] || die 3 "$AFTER did not finish: its marker $AFTER_MARKER says '$status'; not starting"
    log "$AFTER finished (marker $AFTER_MARKER says 0)"
}

# True when the lease file shows the GPU's row with this queue's owner and PID. Only
# the index is read by position (the first field), so a change of column order in the
# node's copy of the tool does not break the check.
lease_is_mine() {
    [ -r "$LEASE_FILE" ] || return 1
    awk -v i="$GPU" -v o="$OWNER" -v p="$$" '
        $1 == i { for (f = 2; f <= NF; f++) { if ($f == o) own = 1; if ($f == p) pid = 1 } }
        END { exit !(own && pid) }' "$LEASE_FILE"
}

# b. Claim the GPU. The lease tool gives out a card whose row is free whatever the card
# holds (only its find checks memory), and a vLLM server that outlived a killed job holds
# a card with no lease. So the card must be empty first, after up to MEM_WAIT_S for the
# job before this queue to let go of it.
claim_gpu() {
    card_empty || die 4 "GPU $GPU is in use, whatever its lease row says (see the nvidia-smi table above); nothing ran"
    log "claiming GPU $GPU for $OWNER, pid $$: $PURPOSE"
    CLAIM_TRIED=1
    # The lease records CLAIM_PID. This shell lives as long as the queue, so the lease
    # stays live between steps, and the tool counts it stale only after the queue is
    # gone and the card is empty.
    if ! logged env CLAIM_PID=$$ "$LEASE_TOOL" claim "$GPU" "$OWNER" "$PURPOSE"; then
        die 4 "the lease tool did not give GPU $GPU (another job holds it, or the tool failed); nothing ran"
    fi
    lease_is_mine || die 4 "the lease tool reported a claim, but $LEASE_FILE does not show GPU $GPU for $OWNER, pid $$"
    log "claimed GPU $GPU"
}

tracked() {
    git -C "$REPO" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}

# The files of one --keep-local argument, NUL-separated: the file itself, or every file
# git tracks under a folder.
keep_files() {
    if [ -d "$REPO/$1" ]; then
        git -C "$REPO" ls-files -z -- "$1"
    elif [ -f "$REPO/$1" ]; then
        printf '%s\0' "$1"
    fi
}

# Copy the --keep-local files out of the checkout before the switch. Two kinds:
#   - the host's own copy: a file marked skip-worktree, or one git does not track. It
#     comes back after the switch whatever the commit holds. A skip-worktree file is put
#     back to its committed version first, because git refuses to switch while such a
#     file differs from the index.
#   - a committed copy: a tracked file with no local change. It comes back only if the
#     commit no longer tracks it, which means the switch deleted it.
keep_save() {
    local bk=$1 arg f flag n
    SAVED=() OWN=()
    for arg in ${KEEP[@]+"${KEEP[@]}"}; do
        n=0
        while IFS= read -r -d '' f; do
            [ -f "$REPO/$f" ] || continue
            mkdir -p "$bk/$(dirname "$f")" || return 1
            cp -p "$REPO/$f" "$bk/$f" || return 1
            # Check the copy before anything overwrites the original.
            cmp -s "$REPO/$f" "$bk/$f" || return 1
            flag="$(git -C "$REPO" ls-files -v -- "$f" | cut -c1)"
            SAVED+=("$f")
            if [ "$flag" = S ] || [ -z "$flag" ]; then OWN+=(1); else OWN+=(0); fi
            if [ "$flag" = S ]; then
                git -C "$REPO" update-index --no-skip-worktree -- "$f" || return 1
                git -C "$REPO" checkout -q -- "$f" || return 1
            fi
            n=$((n + 1))
        done < <(keep_files "$arg")
        log "keep-local $arg: $n files copied to $bk"
    done
}

# A restored file that the commit does not track must not show as untracked: if no
# ignore rule covers it, it goes into the checkout's .git/info/exclude.
keep_ignored() {
    local exclude
    git -C "$REPO" check-ignore -q -- "$1" && return 0
    exclude="$(git -C "$REPO" rev-parse --git-path info/exclude)"
    case $exclude in /*) ;; *) exclude="$REPO/$exclude" ;; esac
    mkdir -p "$(dirname "$exclude")"
    grep -qxF "/$1" "$exclude" 2>/dev/null || printf '/%s\n' "$1" >>"$exclude"
    log "keep-local $1: listed in $exclude"
}

# Put the saved files back, by the rules of keep_save. Used after the switch, and after
# a failed switch, where it restores the checkout as it was.
keep_restore() {
    local bk=$1 i f back=0
    for ((i = 0; i < ${#SAVED[@]}; i++)); do
        f=${SAVED[$i]}
        if [ "${OWN[$i]}" = 0 ] && tracked "$f"; then
            continue
        fi
        # The switch removes a folder that it leaves empty.
        if ! { mkdir -p "$(dirname "$REPO/$f")" && cp -p "$bk/$f" "$REPO/$f"; }; then
            log "keep-local $f: could not copy it back; the saved copy is $bk/$f"
            continue
        fi
        back=$((back + 1))
        if tracked "$f"; then
            git -C "$REPO" update-index --skip-worktree -- "$f"
            log "keep-local $f: the host's copy is back, with skip-worktree"
        else
            keep_ignored "$f"
        fi
    done
    [ "${#SAVED[@]}" -eq 0 ] || log "keep-local: $back of ${#SAVED[@]} saved files copied back"
}

# The lines of git status that block a switch: any change to a tracked file, and any
# untracked file outside the --allow-untracked paths. Ignored files never show, and a
# skip-worktree file does not either. A failed git status prints its error, which also
# blocks.
checkout_dirt() {
    local status line prefix allowed
    if ! status="$(git -C "$REPO" status --porcelain 2>&1)"; then
        printf 'git status failed: %s\n' "$status"
        return
    fi
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        allowed=0
        if [ "${line:0:3}" = "?? " ]; then
            for prefix in ${ALLOW[@]+"${ALLOW[@]}"}; do
                case ${line:3} in "$prefix"*) allowed=1 ;; esac
            done
        fi
        [ "$allowed" = 1 ] || printf '%s\n' "$line"
    done <<<"$status"
}

check_clean() {
    local dirt
    dirt="$(checkout_dirt)"
    if [ -n "$dirt" ]; then
        log_lines "$dirt"
        die 5 "$REPO has changes or untracked files $1 (listed above); the checkout cannot switch"
    fi
}

# c. Switch the checkout to the commit from the bundle.
switch_checkout() {
    local target head ref refs=()
    check_clean "before the switch"
    while read -r _ ref; do
        [ -n "$ref" ] && refs+=("$ref")
    done < <(git -C "$REPO" bundle list-heads "$BUNDLE")
    [ "${#refs[@]}" -gt 0 ] || die 5 "$BUNDLE lists no refs"
    # The refs go to FETCH_HEAD only: no branch or remote ref of the checkout changes.
    logged git -C "$REPO" fetch --no-tags "$BUNDLE" "${refs[@]}" || die 5 "git fetch from $BUNDLE failed"
    target="$(git -C "$REPO" rev-parse --verify --quiet "$COMMIT^{commit}")" ||
        die 5 "commit $COMMIT is in neither $BUNDLE nor the checkout"
    head="$(git -C "$REPO" rev-parse HEAD)"
    if [ "$head" = "$target" ]; then
        log "checkout already at $target"
        return 0
    fi
    log "switching $REPO from $head to $target"
    if ! keep_save "$LOG_DIR/$NAME.kept"; then
        keep_restore "$LOG_DIR/$NAME.kept"
        die 5 "could not save the --keep-local files; the checkout stays at $head"
    fi
    if ! logged git -C "$REPO" checkout -q --detach "$target"; then
        keep_restore "$LOG_DIR/$NAME.kept"
        die 5 "git checkout $target failed (git's reason is above); the checkout stays at $head"
    fi
    keep_restore "$LOG_DIR/$NAME.kept"
    [ "$(git -C "$REPO" rev-parse HEAD)" = "$target" ] || die 5 "HEAD is not $target after the switch"
    check_clean "after the switch"
    log "checkout at $target, clean"
}

# d. Source the run env file. set -a exports every variable it sets, so the steps see
# them. Its values are never logged: it holds the Hugging Face token.
source_env() {
    log "sourcing $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    if ! . "$ENV_FILE"; then
        set +a
        die 6 "sourcing $ENV_FILE failed"
    fi
    set +a
}

# Succeeds when every --gpus in the line names this queue's GPU and no other. The
# runner sets CUDA_VISIBLE_DEVICES from --gpus for each stage, which would override
# the pin below, so a line for another GPU must not run.
gpus_flag_ok() {
    local words w expect=0
    read -r -a words <<<"$1"
    for w in "${words[@]}"; do
        if [ "$expect" = 1 ]; then
            [ "$w" = "$GPU" ] || return 1
            expect=0
            continue
        fi
        case $w in
            --gpus) expect=1 ;;
            --gpus=*) [ "${w#--gpus=}" = "$GPU" ] || return 1 ;;
        esac
    done
    [ "$expect" = 0 ]
}

# The queue file: one command per line; blank lines and lines that start with # are
# skipped. A relative path is read from the checkout, after the switch.
read_queue() {
    local path=$QUEUE line
    case $path in /*) ;; *) path="$REPO/$path" ;; esac
    [ -f "$path" ] || die 7 "queue file $path not found"
    CMDS=()
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        case $line in '' | '#'*) continue ;; esac
        gpus_flag_ok "$line" || die 7 "this line names a GPU other than $GPU: $line"
        CMDS+=("$line")
    done <"$path"
    [ "${#CMDS[@]}" -gt 0 ] || die 7 "$path holds no command"
    # The copy is the record of what this run ran, whatever happens to the file later.
    cp "$path" "$LOG_DIR/$NAME.queue.txt"
    log "queue file $path: ${#CMDS[@]} commands (copy: $LOG_DIR/$NAME.queue.txt)"
}

elapsed() {
    local s=$(($(date +%s) - $1))
    printf '%dh%02dm%02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

# e. Run the commands, each in its own process group, so a signal can reach every
# process of a step and nothing else.
run_steps() {
    local n=${#CMDS[@]} i rc start steplog cmd
    cd "$REPO" || die 2 "cannot cd to $REPO"
    # The pin: CUDA numbers the cards in PCI bus order, as nvidia-smi and the lease file
    # do, and each step sees only the claimed card. Set after the env file, so the env
    # file cannot change it.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU"
    set -m
    for ((i = 0; i < n; i++)); do
        STEP=$((i + 1))
        cmd=${CMDS[$i]}
        steplog="$LOG_DIR/$NAME.step$(printf %02d "$STEP").log"
        log "step $STEP/$n start: $cmd"
        printf '%s step %s/%s: %s\n' "$(ts)" "$STEP" "$n" "$cmd" >>"$steplog"
        start=$(date +%s)
        bash -o pipefail -c "$cmd" </dev/null >>"$steplog" 2>&1 &
        CHILD_PID=$!
        wait "$CHILD_PID"
        rc=$?
        CHILD_PID=""
        log "step $STEP/$n exit $rc after $(elapsed "$start") (output: $steplog)"
        [ "$rc" -eq 0 ] || die "$rc" "step $STEP/$n failed with exit $rc: $cmd"
    done
    REASON="$n of $n steps finished"
}

# Any process of the group still alive (a zombie is not).
group_alive() {
    ps -A -o pgid= -o stat= 2>/dev/null |
        awk -v g="$1" '$1 == g && $2 !~ /^Z/ { found = 1 } END { exit !found }'
}

# SIGINT, SIGTERM or SIGHUP (tmux kill-session) to the queue: stop the running step
# with SIGINT, which Python turns into KeyboardInterrupt, so the stage's engine stops
# its vLLM server (bodhi_lora/engine/vllm.py, __exit__); SIGKILL only after the grace
# time. Then exit, which releases the GPU.
on_signal() {
    local sig=$1 code=$2 waited=0
    trap '' INT TERM HUP
    log "got SIG$sig"
    if [ -n "$CHILD_PID" ]; then
        log "step $STEP: SIGINT to its process group $CHILD_PID"
        kill -INT -- "-$CHILD_PID" 2>/dev/null
        while group_alive "$CHILD_PID" && [ "$waited" -lt "$STOP_GRACE_S" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if group_alive "$CHILD_PID"; then
            log "step $STEP: still running after ${STOP_GRACE_S}s; SIGKILL to its process group"
            kill -KILL -- "-$CHILD_PID" 2>/dev/null
        fi
        wait "$CHILD_PID" 2>/dev/null
        die "$code" "stopped by SIG$sig during step $STEP"
    fi
    die "$code" "stopped by SIG$sig"
}

# Memory in use on the card, in MiB. A failed query counts as busy, as in the lease
# tool. Without nvidia-smi (a CPU test) there is nothing to check.
card_used_mib() {
    local mib
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null |
        head -n 1 | tr -dc '0-9')"
    echo "${mib:-99999}"
}

# Wait up to MEM_WAIT_S for the card to empty. A job killed hard can leave its vLLM
# server alive, because the engine starts the server in a session of its own.
card_empty() {
    local waited=0 mib
    while :; do
        mib="$(card_used_mib)"
        [ "$mib" -lt "$EMPTY_MIB" ] && return 0
        [ "$waited" -ge "$MEM_WAIT_S" ] && break
        sleep 5
        waited=$((waited + 5))
    done
    log "GPU $GPU still uses $mib MiB after ${waited}s:"
    logged nvidia-smi -i "$GPU"
    return 1
}

# f. Release the GPU, if the lease is this queue's. A card that still holds memory
# keeps its lease: the lease tool counts it stale, and so free, once this shell is gone
# and the card is empty, and until then nobody is given a card that is in use.
release_gpu() {
    [ "$CLAIM_TRIED" = 1 ] || return 0
    if ! lease_is_mine; then
        log "the lease of GPU $GPU does not name this queue; not releasing it"
        return 0
    fi
    if ! card_empty; then
        log "GPU $GPU lease kept: the card is not empty (see the nvidia-smi table above)"
        REASON="$REASON; lease of GPU $GPU kept, the card was not empty"
        return 0
    fi
    logged "$LEASE_TOOL" release "$GPU"
    if lease_is_mine; then
        log "release of GPU $GPU did not take; $LEASE_FILE still names this queue"
        REASON="$REASON; release of GPU $GPU failed"
    else
        log "released GPU $GPU"
    fi
}

on_exit() {
    local rc=$?
    trap '' INT TERM HUP
    [ -n "$REASON" ] || REASON="exit $rc"
    release_gpu
    printf '%s\n%s\n%s\n' "$rc" "$REASON" "$(ts)" >"$MARKER"
    log "queue $NAME: exit $rc ($REASON); marker $MARKER"
}

main() {
    parse_args "$@"
    if [ -z "$QUEUE" ] || [ -z "$LOG_DIR" ]; then
        echo "gpu_queue.sh: --queue and --log-dir are required" >&2
        usage >&2
        exit 2
    fi
    if [ -z "$NAME" ]; then
        NAME="$(basename "$QUEUE")"
        NAME="${NAME%.*}"
    fi
    mkdir -p "$LOG_DIR" || exit 2
    QLOG="$LOG_DIR/$NAME.log"
    MARKER="$LOG_DIR/$NAME.exit"
    # Overwrite any marker of an earlier run at once: a queue chained after this one
    # must never read an old 0 while this run is still going, or after it was killed.
    printf 'running\nstarted %s, pid %s\n' "$(ts)" "$$" >"$MARKER" || exit 2
    trap on_exit EXIT
    trap 'on_signal INT 130' INT
    trap 'on_signal TERM 143' TERM
    trap 'on_signal HUP 129' HUP
    # A closed output pipe must not kill the queue before it releases the GPU. A caught
    # signal, unlike an ignored one, is back to its default in each command it starts.
    trap ':' PIPE
    log "queue $NAME: start, pid $$, GPU $GPU, checkout $REPO"
    validate
    [ -z "$AFTER" ] || wait_for_session
    claim_gpu
    [ -z "$COMMIT" ] || switch_checkout
    cd "$REPO" || die 2 "cannot cd to $REPO"
    [ -z "$ENV_FILE" ] || source_env
    read_queue
    run_steps
    exit 0
}

main "$@"; exit $?
