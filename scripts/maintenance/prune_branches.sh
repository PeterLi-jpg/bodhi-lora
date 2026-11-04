#!/usr/bin/env bash
# List, and with --apply delete, branches that should not live on.
#
# Prepared on 2026-10-07 for the branch policy in docs/CONVENTIONS.md. It is a
# dry run by default: it only reads refs and prints what it would delete.
#
# Deleted with --apply, locally and on the remote:
#   scratch   claude/*, worktree-agent*  (scratch branches, never pushed by policy)
#   temp      _backup_main, _backup_origin_main, data-temp, results-temp
#             (results-temp holds MedGemma adapter weights and optimizer state)
#   merged    branches whose tip is already in the mainline
# Listed for review only, never deleted:
#   unmerged  every other branch not in the mainline
# Never touched: the mainline, main, master, the current branch, branches
# checked out in a worktree, cleanup/*, rebuttal/generality-experiments.
#
# Usage:
#   scripts/maintenance/prune_branches.sh [--mainline REF] [--remote NAME]
#   scripts/maintenance/prune_branches.sh --apply [--mainline REF] [--remote NAME]
#
# The dry run reads the remote-tracking refs you already have, so run
# `git fetch --prune <remote>` first for current remote counts. --apply runs
# that fetch itself, shows the plan again and asks before deleting anything.
set -euo pipefail

mainline="main"
remote="origin"
apply=0
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) apply=1; shift ;;
        --mainline) mainline="${2:?--mainline needs a ref}"; shift 2 ;;
        --remote) remote="${2:?--remote needs a name}"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

git rev-parse --verify -q "${mainline}^{commit}" >/dev/null \
    || { echo "mainline ref not found: $mainline" >&2; exit 2; }

if [ "$apply" -eq 1 ]; then
    git fetch --prune "$remote"
fi

current="$(git symbolic-ref -q --short HEAD || true)"
# Branches checked out in any worktree cannot be deleted and must not be.
in_worktree="$(git worktree list --porcelain | sed -n 's#^branch refs/heads/##p')"

is_protected() {
    local b="$1"
    case "$b" in
        HEAD|main|master|cleanup/*|rebuttal/generality-experiments) return 0 ;;
    esac
    [ "$b" = "${mainline#refs/heads/}" ] && return 0
    [ "$b" = "$current" ] && return 0
    printf '%s\n' "$in_worktree" | grep -qxF -- "$b" && return 0
    return 1
}

category() {
    case "$1" in
        claude/*|worktree-agent*) echo scratch ;;
        _backup_main|_backup_origin_main|data-temp|results-temp) echo temp ;;
        *) echo other ;;
    esac
}

merged_local="$(git branch --format='%(refname:short)' --merged "$mainline")"
merged_remote="$(git branch -r --format='%(refname)' --merged "$mainline")"

# Plain arrays only: macOS ships bash 3.2, which has no associative arrays.
# Entries are "where|category|branch"; unmerged ones also carry the tip date.
plan=()

while read -r b; do
    [ -n "$b" ] || continue
    is_protected "$b" && continue
    c="$(category "$b")"
    if [ "$c" = other ]; then
        if printf '%s\n' "$merged_local" | grep -qxF -- "$b"; then c=merged; else c=unmerged; fi
    fi
    plan+=("local|$c|$b|$(git log -1 --format='%cs' "refs/heads/$b")")
done < <(git branch --format='%(refname:short)')

# Full ref names: the short form of refs/remotes/origin/HEAD is just "origin".
while read -r rb; do
    [ -n "$rb" ] || continue
    b="${rb#refs/remotes/"$remote"/}"
    [ "$b" = HEAD ] && continue
    is_protected "$b" && continue
    c="$(category "$b")"
    if [ "$c" = other ]; then
        if printf '%s\n' "$merged_remote" | grep -qxF -- "$rb"; then c=merged; else c=unmerged; fi
    fi
    plan+=("remote|$c|$b|$(git log -1 --format='%cs' "$rb")")
done < <(git for-each-ref --format='%(refname)' "refs/remotes/$remote/")

# Print plan entries for one location, either to delete or to review.
show() {   # $1=local|remote  $2=delete|review
    local e where c b d
    for e in ${plan[@]+"${plan[@]}"}; do
        IFS='|' read -r where c b d <<< "$e"
        [ "$where" = "$1" ] || continue
        if [ "$2" = review ] && [ "$c" = unmerged ]; then
            printf '  %-7s %s  %s\n' "$where" "$d" "$b"
        elif [ "$2" = delete ] && [ "$c" != unmerged ]; then
            printf '  %-8s %s\n' "$c" "$b"
        fi
    done
}
count() {   # $1=local|remote  $2=category
    local e n=0
    for e in ${plan[@]+"${plan[@]}"}; do
        case "$e" in "$1|$2|"*) n=$((n + 1)) ;; esac
    done
    echo "$n"
}
n_del_local=$(( $(count local scratch) + $(count local temp) + $(count local merged) ))
n_del_remote=$(( $(count remote scratch) + $(count remote temp) + $(count remote merged) ))

echo "mainline: $mainline   remote: $remote   mode: $([ "$apply" -eq 1 ] && echo APPLY || echo dry-run)"
echo
echo "== would delete, local ($n_del_local)"
show local delete
echo "== would delete, remote $remote ($n_del_remote)"
show remote delete
echo "== unmerged, review by hand, not deleted ($(( $(count local unmerged) + $(count remote unmerged) )))"
show local review
show remote review
echo
echo "== counts"
for where in local remote; do
    for c in scratch temp merged unmerged; do
        printf '  %-7s %-9s %s\n' "$where" "$c" "$(count "$where" "$c")"
    done
done

if [ "$apply" -eq 0 ]; then
    echo
    echo "Dry run. Nothing was deleted. Re-run with --apply to delete the branches above."
    exit 0
fi

echo
read -r -p "Delete $n_del_local local and $n_del_remote remote branches? Type 'delete' to go on: " answer
[ "$answer" = delete ] || { echo "Stopped. Nothing deleted."; exit 1; }

names=()
for e in ${plan[@]+"${plan[@]}"}; do
    IFS='|' read -r where c b _ <<< "$e"
    [ "$c" = unmerged ] && continue
    if [ "$where" = local ]; then
        # Scratch and temp branches are unmerged by nature, so they need -D.
        if [ "$c" = merged ]; then git branch -d -- "$b"; else git branch -D -- "$b"; fi
    else
        names+=("$b")
    fi
done
# One push per 50 refs keeps each command short.
for ((i = 0; i < ${#names[@]}; i += 50)); do
    git push "$remote" --delete -- "${names[@]:i:50}"
done
echo "Done. Check with: git branch -a"
