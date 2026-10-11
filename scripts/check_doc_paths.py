#!/usr/bin/env python3
"""Check that every repo path the docs mention exists.

Docs drifted when code paths were deleted (audit HYG-14): README and
KNOWN_ISSUES pointed at TPU launchers that were gone. This check fails CI when
a doc names a path that is not there.

What it reads: README.md, CONTRIBUTING.md, results/README.md and docs/**/*.md. In each
it takes inline code spans,
markdown link targets and the words inside fenced code blocks, and keeps the
ones that look like repo paths: the first path part is a top-level entry of
the repo (or a root file such as setup.sh). It also reads the source comments of
paper/generated/*.tex, where each macro and table names the file it came from, so a
reader who follows a number's source does not reach a path that is gone.

A path passes when it is one of these:
  - tracked in git, or a directory that holds tracked files
  - ignored by .gitignore (outputs and local data, such as eval/ or data/raw/)
  - written as <commit>:<path> and present at that commit (for history notes)
  - in PENDING below
Placeholders such as seed_<N>, * and {a,b} are expanded before the check.

Usage: python scripts/check_doc_paths.py [--verbose]
Exit 1 if any path is missing.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = ["README.md", "CONTRIBUTING.md", "results/README.md"]
DOC_GLOBS = ["docs/**/*.md"]

# Paths the docs name that are not in the repo yet. Each entry is a promise, not
# a file: the check warns when one exists, so the entry can be removed.
PENDING: dict[str, str] = {}

TOP_FILES_EXT = (".sh", ".md", ".txt", ".py", ".toml", ".cfg", ".yaml", ".yml")
# A relative path ending in one of these is checked even when its first folder is
# gone from the repo. That is how a doc pointing at a deleted tpu/ launcher fails.
FILE_EXT = TOP_FILES_EXT + (".json", ".jsonl", ".tex", ".bib", ".xlsx", ".pdf", ".csv", ".ipynb")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s*(>\s*)*(```|~~~)")  # also fences inside a blockquote
COMMIT_PATH = re.compile(r"^([0-9a-f]{7,40}):(.+)$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


TRACKED = set(git("ls-files").splitlines())
TRACKED_DIRS: set[str] = set()
for p in TRACKED:
    parts = p.split("/")
    for i in range(1, len(parts)):
        TRACKED_DIRS.add("/".join(parts[:i]) + "/")
TOP = {p.split("/", 1)[0] for p in TRACKED} | {k.split("/", 1)[0] for k in PENDING}
# Top-level output dirs that are ignored on purpose but still worth checking.
TOP |= {"checkpoints", "logs", "results", "outputs", "paper"}


def expand(path: str) -> list[str]:
    """Turn placeholders into concrete samples and glob patterns."""
    path = re.sub(r"<[^>/]+>", "*", path)
    m = re.search(r"\{([^{}]+)\}", path)
    if m:
        out = []
        for alt in m.group(1).split(","):
            out += expand(path[:m.start()] + alt.strip() + path[m.end():])
        return out
    return [path]


def is_ignored(path: str) -> bool:
    sample = path.replace("*", "x").rstrip("/")
    # A path that does not exist yet may be a directory: a rule like
    # "eval/figures/" only matches it with something inside.
    for probe in (sample, sample + "/x"):
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=ROOT)
        if r.returncode == 0:
            return True
    return False


def is_branch(name: str) -> bool:
    for ref in (f"refs/heads/{name}", f"refs/remotes/origin/{name}"):
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=ROOT, capture_output=True)
        if r.returncode == 0:
            return True
    return False


def exists(path: str) -> bool:
    m = COMMIT_PATH.match(path)
    if m:
        r = subprocess.run(["git", "cat-file", "-e", f"{m.group(1)}:{m.group(2).rstrip('/')}"],
                           cwd=ROOT, capture_output=True)
        return r.returncode == 0
    for p in expand(path):
        if any(p == k or (k.endswith("/") and p.startswith(k)) for k in PENDING):
            continue
        if "*" in p:
            pat = p.rstrip("/")
            if any(fnmatch.fnmatch(t, pat) or fnmatch.fnmatch(t, pat + "/*") for t in TRACKED):
                continue
        elif p.rstrip("/") in TRACKED or (p.rstrip("/") + "/") in TRACKED_DIRS:
            continue
        if is_ignored(p) or is_branch(p):
            continue
        return False
    return True


def candidate(tok: str, doc_dir: Path, is_link: bool, in_fence: bool = False) -> str | None:
    tok = tok.strip().strip("'\"").rstrip(".,;:)")
    tok = tok.split("#", 1)[0]
    tok = re.sub(r":\d+(-\d+)?$", "", tok)  # file:line references
    if not tok or "://" in tok or tok.startswith(("-", "$", "~", "/", "<", "@", "=")):
        return None
    if " " in tok or "=" in tok:
        return None
    if COMMIT_PATH.match(tok):
        return tok
    if is_link:
        # Markdown links resolve against the doc's own folder.
        resolved = (ROOT / doc_dir / tok).resolve()
        try:
            tok = str(resolved.relative_to(ROOT))
        except ValueError:
            return None
        if (ROOT / tok).is_dir() and not tok.endswith("/"):
            tok += "/"
        return tok  # a link always names a repo path
    if tok.startswith("./"):
        tok = tok[2:]
    first = tok.split("/", 1)[0]
    if "/" in tok and first in TOP:
        return tok
    if "/" in tok and not first.startswith(".") and tok.endswith(FILE_EXT):
        return tok
    if not in_fence and re.fullmatch(r"[A-Za-z0-9_-]+/", tok):
        # A bare top-level folder in prose, such as `legacy/`. Inside a code
        # block the same shape is usually a line of a folder tree.
        return tok
    if "/" not in tok and (tok in TOP or tok in PENDING) and (tok.endswith(TOP_FILES_EXT) or tok in PENDING
                                                               or tok in {"LICENSE", "NOTICE", "Makefile"}):
        return tok
    return None


def doc_files() -> list[Path]:
    files = [ROOT / d for d in DOCS if (ROOT / d).exists()]
    for g in DOC_GLOBS:
        files += sorted(ROOT.glob(g))
    return files


def main() -> int:
    verbose = "--verbose" in sys.argv
    missing, checked = [], 0
    for doc in doc_files():
        rel = doc.relative_to(ROOT)
        in_fence = False
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            toks = []
            if in_fence:
                # Split on = too, so VAR=path in a shell line yields the path.
                toks = [(t, False) for t in re.split(r"[\s\"'()=]+", line)]
            else:
                toks = [(t, False) for t in CODE_SPAN.findall(line)]
                toks += [(t, True) for t in LINK.findall(line)]
            for raw, is_link in toks:
                path = candidate(raw, rel.parent, is_link, in_fence)
                if path is None:
                    continue
                checked += 1
                ok = exists(path)
                if verbose:
                    print(f"{'ok  ' if ok else 'MISS'} {rel}:{lineno}: {path}")
                if not ok:
                    missing.append(f"{rel}:{lineno}: {path}")
    # The source comments of the generated numbers and tables. Every macro names the same
    # few files, so each path is checked once.
    seen: dict[str, bool] = {}
    for tex in sorted((ROOT / "paper" / "generated").glob("*.tex")):
        rel = tex.relative_to(ROOT)
        for lineno, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
            if "% " not in line:
                continue
            for raw in re.split(r"[\s()]+", line.split("% ", 1)[1]):
                path = candidate(raw, rel.parent, False, in_fence=True)
                if path is None:
                    continue
                if path not in seen:
                    seen[path] = exists(path)
                    checked += 1
                if not seen[path]:
                    missing.append(f"{rel}:{lineno}: {path}")
    stale = [k for k in PENDING if (k.rstrip("/") in TRACKED or k in TRACKED_DIRS)]
    for k in stale:
        print(f"note: PENDING entry now exists, remove it from the list: {k}")
    if missing:
        print(f"check_doc_paths: {len(missing)} missing of {checked} paths")
        for m in missing:
            print("  " + m)
        return 1
    print(f"check_doc_paths: OK, {checked} paths in {len(doc_files())} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
