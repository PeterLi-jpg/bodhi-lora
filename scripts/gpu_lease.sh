#!/usr/bin/env bash
# gpu_lease.sh — cooperative GPU reservation for the shared NVIDIA node.
#
# WHY THIS EXISTS
#   nvidia-smi cannot prevent a race: a process takes ~1-2 min between launch and
#   its first CUDA allocation, so the GPU reads 0 MiB the whole time and a second
#   agent grabs the same card. Declaring intent up front closes that window.
#
# USAGE
#   gpu_lease.sh status                     # show the table
#   gpu_lease.sh claim 3 bohdi "rebuttal medqa"   # claim one GPU (fails if busy)
#   gpu_lease.sh find 2 bohdi "my job"      # claim N free GPUs, prints indices
#   gpu_lease.sh release 3                  # give it back
#   gpu_lease.sh release-mine bohdi         # release everything owned by you
#   gpu_lease.sh reconcile                  # auto-free entries whose PID is dead
#                                           #   AND whose GPU is actually empty
#
# ALWAYS release when done. In scripts, make it automatic:
#   trap 'gpu_lease.sh release-mine myname' EXIT
#
# Concurrency: every mutation takes an flock, so simultaneous claims can't both
# win the same GPU. State lives in one human-readable table (LEASE_FILE).
set -uo pipefail

# Canonical shared path agreed with the group (Sebastian, 2026-07-27).
# Do NOT create a second copy elsewhere — two files would reintroduce exactly the
# inconsistency this protocol exists to prevent.
LEASE_FILE="${GPU_LEASE_FILE:-/home/nvidia/GPUS.txt}"
LOCK_FILE="${LEASE_FILE}.lock"
NGPU="${NGPU:-8}"

_init_if_missing() {
    [ -s "$LEASE_FILE" ] && return 0
    {
        echo "# GPU leases for this node. Claim BEFORE you start; release when done."
        echo "# Manage with scripts/gpu_lease.sh (atomic). Do not hand-edit while jobs run."
        printf "%-4s %-6s %-10s %-9s %-20s %s\n" "#idx" "state" "owner" "pid" "since" "purpose"
        for i in $(seq 0 $((NGPU - 1))); do
            printf "%-4s %-6s %-10s %-9s %-20s %s\n" "$i" "free" "-" "-" "-" "-"
        done
    } > "$LEASE_FILE"
    chmod 666 "$LEASE_FILE" 2>/dev/null || true
}

_row_state() { awk -v i="$1" '$1==i {print $2}' "$LEASE_FILE" 2>/dev/null | head -1; }
_row_pid()   { awk -v i="$1" '$1==i {print $4}' "$LEASE_FILE" 2>/dev/null | head -1; }
_row_owner() { awk -v i="$1" '$1==i {print $3}' "$LEASE_FILE" 2>/dev/null | head -1; }

_set_row() { # idx state owner pid purpose
    local idx="$1" state="$2" owner="$3" pid="$4" purpose="$5"
    local since="-"
    [ "$state" = "busy" ] && since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local tmp="${LEASE_FILE}.tmp.$$"
    awk -v i="$idx" -v s="$state" -v o="$owner" -v p="$pid" -v t="$since" -v u="$purpose" '
        /^#/ { print; next }
        $1==i { printf "%-4s %-6s %-10s %-9s %-20s %s\n", i, s, o, p, t, u; next }
        { print }
    ' "$LEASE_FILE" > "$tmp" && mv "$tmp" "$LEASE_FILE"
    chmod 666 "$LEASE_FILE" 2>/dev/null || true
}

# A lease is stale if its PID is gone AND the GPU holds no real memory.
# Both conditions, so we never free a card that is genuinely in use.
_is_stale() {
    local idx="$1" pid state mem
    state="$(_row_state "$idx")"; [ "$state" = "busy" ] || return 1
    pid="$(_row_pid "$idx")"
    [ "$pid" != "-" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 1
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null || echo 99999)"
    [ "${mem:-99999}" -lt 2000 ]
}

cmd_status() {
    _init_if_missing
    printf "%-4s %-6s %-10s %-9s %-20s %-28s %s\n" "GPU" "state" "owner" "pid" "since" "purpose" "actual"
    for i in $(seq 0 $((NGPU - 1))); do
        local mem util line
        mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null || echo '?')"
        util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$i" 2>/dev/null || echo '?')"
        line="$(awk -v i="$i" '$1==i' "$LEASE_FILE" | head -1)"
        local st ow pd sn pu
        st="$(echo "$line" | awk '{print $2}')"; ow="$(echo "$line" | awk '{print $3}')"
        pd="$(echo "$line" | awk '{print $4}')"; sn="$(echo "$line" | awk '{print $5}')"
        # fields 6..NF are the purpose; awk handles the multi-space padding that
        # `cut -d' '` would split into empty fields.
        pu="$(echo "$line" | awk '{for(i=6;i<=NF;i++) printf "%s%s", $i, (i<NF?" ":"")}')"
        local flag=""
        _is_stale "$i" && flag="  <-- STALE (pid gone, gpu empty): run reconcile"
        printf "%-4s %-6s %-10s %-9s %-20s %-28s %sMiB %s%%%s\n" \
            "$i" "$st" "$ow" "$pd" "$sn" "${pu:0:28}" "$mem" "$util" "$flag"
    done
}

cmd_claim() { # idx owner purpose
    local idx="$1" owner="${2:-unknown}" purpose="${3:--}"
    _init_if_missing
    if [ "$(_row_state "$idx")" = "busy" ] && ! _is_stale "$idx"; then
        echo "GPU $idx already claimed by $(_row_owner "$idx")" >&2
        return 1
    fi
    _set_row "$idx" busy "$owner" "${CLAIM_PID:-$PPID}" "$purpose"
    echo "claimed GPU $idx for $owner"
}

cmd_find() { # n owner purpose  -> claims n free GPUs atomically, prints indices
    local n="$1" owner="${2:-unknown}" purpose="${3:--}" got=()
    _init_if_missing
    for i in $(seq 0 $((NGPU - 1))); do
        [ "${#got[@]}" -ge "$n" ] && break
        local st; st="$(_row_state "$i")"
        if [ "$st" = "free" ] || _is_stale "$i"; then
            local mem; mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null || echo 99999)"
            # Belt and braces: only take it if the card is also physically empty.
            if [ "${mem:-99999}" -lt 2000 ]; then
                _set_row "$i" busy "$owner" "${CLAIM_PID:-$PPID}" "$purpose"
                got+=("$i")
            fi
        fi
    done
    [ "${#got[@]}" -eq 0 ] && { echo "no free GPUs" >&2; return 1; }
    echo "${got[*]}"
}

cmd_release() { _init_if_missing; _set_row "$1" free "-" "-" "-"; echo "released GPU $1"; }

cmd_release_mine() {
    local owner="$1"
    _init_if_missing
    for i in $(seq 0 $((NGPU - 1))); do
        [ "$(_row_owner "$i")" = "$owner" ] && _set_row "$i" free "-" "-" "-" && echo "released GPU $i"
    done
}

cmd_reconcile() {
    _init_if_missing
    local freed=0
    for i in $(seq 0 $((NGPU - 1))); do
        if _is_stale "$i"; then
            echo "GPU $i: stale lease from $(_row_owner "$i") (pid $(_row_pid "$i") gone, card empty) -> free"
            _set_row "$i" free "-" "-" "-"; freed=$((freed + 1))
        fi
    done
    echo "reconciled; $freed lease(s) freed"
}

main() {
    local sub="${1:-status}"; shift || true
    if [ "$sub" = "status" ]; then cmd_status; return; fi   # read-only, no lock

    _init_if_missing
    # Hold the lock on a file descriptor rather than passing a command STRING to
    # `flock -c`: string form re-parses the arguments, which breaks any purpose
    # containing spaces or parentheses (and would be an injection hazard).
    exec 9>"$LOCK_FILE"
    flock 9 || { echo "could not acquire $LOCK_FILE" >&2; exit 1; }
    case "$sub" in
        claim)        cmd_claim "$@" ;;
        find)         cmd_find "$@" ;;
        release)      cmd_release "$@" ;;
        release-mine) cmd_release_mine "$@" ;;
        reconcile)    cmd_reconcile ;;
        *) echo "usage: gpu_lease.sh {status|claim IDX OWNER PURPOSE|find N OWNER PURPOSE|release IDX|release-mine OWNER|reconcile}" >&2; exit 2 ;;
    esac
    # fd 9 (and the lock) are released when the script exits
}
main "$@"
