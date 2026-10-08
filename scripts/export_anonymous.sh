#!/usr/bin/env bash
# Build a squashed, history-free, anonymized copy of this repo for reviewers.
#
# Anonymity is done here, in a fresh copy, never by editing the real repo.
# The copy holds the tracked files of one commit, with no git history, after
# these steps:
#   1. export the tracked files of REF (default HEAD) with `git archive`
#   2. drop paths that reviewers do not need (EXCLUDE below)
#   3. redact identities in every text file and inside .xlsx/.docx members
#   4. scan everything that is left; exit 1 if any identity remains
#   5. make one commit by "Anonymous Authors" (skip with --no-git)
#
# Usage:
#   scripts/export_anonymous.sh OUT_DIR [--ref REF] [--deny-file FILE] [--no-git]
#
#   OUT_DIR     must be outside the repo and must not exist yet, or be empty
#   --deny-file extra tokens to block, one per line (rater and co-author
#               names). Keep this file outside the repo; names never go in git.
#
# Only commit the work first: uncommitted edits are not exported.
# A failed scan leaves OUT_DIR in place for inspection. Do not share it.
set -euo pipefail

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

out=""
ref="HEAD"
deny_file=""
make_git=1
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) ref="${2:?--ref needs a value}"; shift 2 ;;
        --deny-file) deny_file="${2:?--deny-file needs a path}"; shift 2 ;;
        --no-git) make_git=0; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *) [ -z "$out" ] || usage; out="$1"; shift ;;
    esac
done
[ -n "$out" ] || usage

repo="$(git rev-parse --show-toplevel)"
if [ -n "$deny_file" ] && [ ! -r "$deny_file" ]; then
    echo "cannot read --deny-file $deny_file" >&2; exit 2
fi
mkdir -p "$out"
out="$(cd "$out" && pwd -P)"
case "$out/" in
    "$(cd "$repo" && pwd -P)/"*) echo "OUT_DIR must be outside the repo: $out" >&2; exit 2 ;;
esac
if [ -n "$(ls -A "$out")" ]; then
    echo "OUT_DIR is not empty: $out" >&2; exit 2
fi

# Paths reviewers do not need. Internal planning, the previous venue's
# replies and frozen copy, the decision log, and repo tooling whose source
# names the identities it redacts.
EXCLUDE=(
    hooks
    scripts/maintenance
    scripts/export_anonymous.sh
    docs/DECISIONS.md
    paper/submitted
    rebuttal/PLAN.md
    rebuttal/COVERAGE.md
    rebuttal/responses
)

sha="$(git -C "$repo" rev-parse --verify "${ref}^{commit}")"
echo "export: ${ref} (${sha:0:9}) -> $out"
git -C "$repo" archive --format=tar "$sha" | tar -x -C "$out"
for p in "${EXCLUDE[@]}"; do
    rm -rf "${out:?}/$p"
done

# Steps 3 and 4. Python handles the binary formats and the regexes.
python3 - "$out" "$deny_file" <<'PY'
import io
import re
import sys
import zipfile
from pathlib import Path

out = Path(sys.argv[1])
deny_file = sys.argv[2]

def keep_case(word):
    # bohdi -> bodhi with the original capitals, so BOHDI_DIR stays an
    # env var name that every exported file agrees on.
    def repl(m):
        s = m.group(0)
        if s.isupper():
            return word.upper()
        if s[0].isupper():
            return word.capitalize()
        return word
    return repl

# Order matters: specific forms before the bare handle or username.
REDACT = [
    (re.compile(r"github\.com([/:])(PeterLi-jpg|sebasmos)/", re.I), r"github.com\1anonymous/"),
    (re.compile(r"\b(PeterLi-jpg|sebasmos)\b", re.I), "anonymous"),
    (re.compile(r"\banqipeter[\w.+-]*@[\w.-]+", re.I), "anonymous@example.org"),
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)*\.local(domain)?\b", re.I), "anonymous@example.org"),
    (re.compile(r"/Users/[^/\s\"'`)]+"), "/home/user"),
    (re.compile(r"\banqipeter\w*", re.I), "user"),
    (re.compile(r"gs://[\w.-]*tokyo-micron[\w.-]*", re.I), "gs://anonymous-bucket"),
    (re.compile(r"tokyo-micron-\d+-\w+", re.I), "anonymous-gcp-project"),
    (re.compile(r"\bmit-critical-data-\d+\b", re.I), "gpu-host"),
    (re.compile(r"\bbrev-[a-z0-9]{6,}\b", re.I), "gpu-host"),
    (re.compile(r"BODHI: GPU Accelerated Epistemic AI", re.I), "ANONYMOUS-ORG"),
    (re.compile(r"MIT Critical Data", re.I), "ANONYMOUS-LAB"),
    (re.compile(r"Substrate Labs", re.I), "ANONYMOUS-LAB"),
    (re.compile(r"\bPeter Li\b"), "Anonymous Authors"),
    # The repo slug misspells BODHI. The misspelling finds the public repo.
    (re.compile(r"bohdi", re.I), keep_case("bodhi")),
]

# Anything that still matches after redaction fails the export.
GATE = [
    ("github handle", re.compile(r"PeterLi-jpg|sebasmos", re.I)),
    ("author name", re.compile(r"\bPeter Li\b|\bAnqi\b", re.I)),
    ("local username", re.compile(r"anqipeter", re.I)),
    ("lab or host", re.compile(r"\bmit[ -]critical|substrate labs", re.I)),
    ("brev host", re.compile(r"\bbrev-[a-z0-9]{6,}\b", re.I)),
    ("gcp project or bucket", re.compile(r"tokyo-micron", re.I)),
    ("home path", re.compile(r"/Users/")),
    ("machine hostname", re.compile(r"\.localdomain\b|@[\w-]+(\.[\w-]+)*\.local\b", re.I)),
    ("affiliation", re.compile(r"YCRG|King's College|Makerere", re.I)),
    ("repo slug", re.compile(r"bohdi", re.I)),
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(
    r"^(anonymous@example\.org|[\w.+-]+@example\.(org|com|invalid)"
    r"|redacted[\w.-]*@[\w.-]+|garnett@wustl\.edu)$", re.I)  # garnett: NeurIPS style file author

deny = []
if deny_file:
    for line in Path(deny_file).read_text().splitlines():
        tok = line.strip()
        if tok and not tok.startswith("#"):
            # Letters on either side end the match; _ and digits do not,
            # so filled_<name>.xlsx is caught.
            deny.append(re.compile(r"(?<![A-Za-z])" + re.escape(tok) + r"(?![A-Za-z])", re.I))

# Model generations on public benchmark prompts. Short name tokens occur there
# as ordinary words, and these files hold no rater or author text, so the
# deny-file tokens skip these files. The built-in patterns still scan them.
MODEL_OUTPUT = re.compile(r"^(results_modal/seed_[^/]+|results_rebuttal/.+/eval)/")

OFFICE = {".xlsx", ".docx", ".pptx"}

def redact_text(s):
    for pat, rep in REDACT:
        s = pat.sub(rep, s)
    return s

def redact_file(path):
    if path.suffix.lower() in OFFICE:
        src = path.read_bytes()
        buf = io.BytesIO()
        changed = False
        with zipfile.ZipFile(io.BytesIO(src)) as zin, zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.endswith((".xml", ".rels")):
                    new = redact_text(data.decode("utf-8")).encode("utf-8")
                    changed |= new != data
                    data = new
                zout.writestr(info, data, compress_type=info.compress_type)
        if changed:
            path.write_bytes(buf.getvalue())
        return changed
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False  # binary; the gate still scans its raw bytes
    new = redact_text(text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False

def texts_of(path):
    """Yield (label, text) for everything a reader could see in the file."""
    raw = path.read_bytes()
    if path.suffix.lower() in OFFICE:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                yield f"{path.relative_to(out)}!{name}", z.read(name).decode("utf-8", "replace")
    else:
        yield str(path.relative_to(out)), raw.decode("utf-8", "replace")

files = [p for p in sorted(out.rglob("*")) if p.is_file() and ".git" not in p.relative_to(out).parts]
n_changed = sum(redact_file(p) for p in files)
print(f"redact: {n_changed} of {len(files)} files changed")

problems = []
for p in files:
    for label, text in texts_of(p):
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in GATE:
                m = pat.search(line)
                if m:
                    problems.append(f"{label}:{lineno}: {name}: {m.group(0)!r}")
            for m in EMAIL.finditer(line):
                if not EMAIL_OK.match(m.group(0)):
                    problems.append(f"{label}:{lineno}: email: {m.group(0)!r}")
            if MODEL_OUTPUT.match(label):
                continue
            for k, pat in enumerate(deny, 1):
                for m in pat.finditer(line):
                    # An all-caps hit is an acronym (a guideline body, say),
                    # not a name. Names are written Capitalized or lower case.
                    if not m.group(0).isupper():
                        # Never print the token itself.
                        problems.append(f"{label}:{lineno}: deny-file token #{k}")

if problems:
    print(f"gate: FAIL, {len(problems)} hits")
    for line in problems[:200]:
        print("  " + line)
    sys.exit(1)
print(f"gate: PASS ({len(files)} files, {len(GATE)} built-in patterns, emails, {len(deny)} deny-file tokens)")
PY

if [ "$make_git" -eq 1 ]; then
    # One commit, anonymous identity, no hooks or signing from the caller's config.
    # The fixed date hides when the export was made, and makes the commit id
    # depend only on the content.
    GIT_AUTHOR_NAME="Anonymous Authors" GIT_AUTHOR_EMAIL="anonymous@example.org" \
    GIT_COMMITTER_NAME="Anonymous Authors" GIT_COMMITTER_EMAIL="anonymous@example.org" \
    GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
        bash -c 'cd "$1" && git init -q -b main && git add -A \
            && git -c core.hooksPath=/dev/null -c commit.gpgsign=false commit -q -m "Anonymous code release"' _ "$out"
    echo "git: one commit in $out ($(git -C "$out" rev-parse --short HEAD))"
fi
echo "export done: $out"
