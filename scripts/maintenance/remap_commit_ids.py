#!/usr/bin/env python3
"""After the history purge: rewrite the commit ids that tracked files cite.

git filter-repo gives every rewritten commit a new id and writes old -> new to
filter-repo/commit-map in the repository it ran in. It rewrites the ids in commit
messages, not in files, so the docs, configs, tests and run manifests would still cite
ids that no longer exist. scripts/check_doc_paths.py checks every <commit>:<path>
reference with git cat-file, so make check, CI and hooks/pre-commit would then fail, and
no later commit could pass the hook.

Run it in a working clone of the rewritten repository, before the force-push
(purge_history.md, step 7):

    python3 scripts/maintenance/remap_commit_ids.py <commit-map>            # list only
    python3 scripts/maintenance/remap_commit_ids.py <commit-map> --apply    # rewrite

A token is replaced when it is 7 to 40 hex digits standing alone and is the prefix of
exactly one old id in the map whose new id differs. It keeps its length, unless the
shorter new id would be ambiguous in the clone; then it grows until it is not. Every
replacement is printed with its file and line, so read the list before --apply. Files
that are not UTF-8 text are skipped.
"""
from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def read_map(path: Path) -> dict[str, str]:
    """old -> new for the commits whose id changed; filter-repo writes a header line, and
    all zeros for a commit it dropped."""
    out = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[0] != parts[1] and set(parts[1]) != {"0"}:
            out[parts[0]] = parts[1]
    return out


@lru_cache(maxsize=None)
def short(new: str, length: int) -> str:
    """The new id at the token's length, longer only if that prefix is ambiguous here."""
    for n in range(length, 41):
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{new[:n]}^{{commit}}"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == new:
            return new[:n]
    return new


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mapping = read_map(Path(sys.argv[1]))
    apply = "--apply" in sys.argv[2:]
    # Every token is at least 7 digits, so the first 7 pick the candidates.
    by_prefix: dict[str, list[str]] = {}
    for old in mapping:
        by_prefix.setdefault(old[:7], []).append(old)
    changed_files = replacements = 0
    for name in git("ls-files", "-z").split("\0"):
        path = Path(name)
        if not name or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.split("\n")
        hit = False
        for i, line in enumerate(lines):
            def swap(m: re.Match) -> str:
                nonlocal hit, replacements
                tok = m.group(0)
                matches = [o for o in by_prefix.get(tok[:7], []) if o.startswith(tok)]
                if len(matches) != 1:
                    return tok
                new = short(mapping[matches[0]], len(tok))
                print(f"{name}:{i + 1}: {tok} -> {new}")
                hit = True
                replacements += 1
                return new
            lines[i] = TOKEN.sub(swap, line)
        if hit:
            changed_files += 1
            if apply:
                path.write_text("\n".join(lines), encoding="utf-8")
    verb = "rewrote" if apply else "would rewrite"
    print(f"remap_commit_ids: {verb} {replacements} ids in {changed_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
