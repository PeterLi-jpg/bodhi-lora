"""Content rules for the history purge described in purge_history.md.

git filter-repo calls Purger.transform(path, data) for every file in every
commit, through the --file-info-callback given in purge_history.md. Rules:

  names        Each "Token==>Replacement" line of the map file replaces that
               token wherever no letter touches it on either side (so
               filled_<token>.xlsx is caught). All-caps hits are acronyms and
               stay. Model outputs, training traces, tokenizers, checkpoints
               and vendored code are skipped (SKIP), because short tokens
               occur there as ordinary words.
  author block An uncommented \\author{...} in a .tex file that is not
               already anonymous becomes \\author{Anonymous Authors}.
  .xlsx        The name rule also runs inside each XML member.

The map file holds real names, so it lives outside the repo and never in git.

Test the rules on this repo without rewriting anything:
  python3 scripts/maintenance/purge_callback.py selftest MAP_FILE
Write the matching --replace-message file for commit messages:
  python3 scripts/maintenance/purge_callback.py messages MAP_FILE > messages.txt
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
import zipfile
from collections import Counter

# Model outputs and training traces on public prompts, tokenizer vocabularies,
# checkpoints and vendored code. Short tokens occur there as ordinary words.
# results/<cell>/seed_*/ and shared/ hold the same outputs since the results root
# (results/PROVENANCE.md); a cell's validation/ and analysis/ are still read.
SKIP = re.compile(rb"^(results_modal/seed_[^/]+/|results_rebuttal/|third_party/|checkpoints/"
                  rb"|data/sft[^/]*/|eval/|results/[^/]+/(?:seed_[^/]+|shared)/)|(^|/)tokenizer[^/]*$")
# Only text-like files are read. Weights and optimizer states are skipped
# unread, which keeps the rewrite fast.
TEXT_EXT = (b".md", b".txt", b".tex", b".bib", b".py", b".sh", b".json", b".jsonl",
            b".yaml", b".yml", b".toml", b".cfg", b".csv", b".html")
# The selftest reports any affiliation left in a .tex blob. The second pattern is
# case-sensitive and allows MIT as a license name (as scripts/export_anonymous.sh does).
AFFILIATION = re.compile(rb"MIT Critical Data|YCRG|King's College|Makerere|Chapel Hill"
                         rb"|North Carolina|Massachusetts Institute", re.I)
AFFILIATION_CAPS = re.compile(rb"\bUNC\b|\bMIT\b(?!\s*(?:[Ll]icen[cs]e|, see|\|))")


def read_map(path: str) -> list[tuple[str, str]]:
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token, sep, repl = line.partition("==>")
        if not sep or not token.strip() or not repl.strip():
            raise SystemExit(f"bad map line (want Token==>Replacement): line {len(pairs) + 1}")
        pairs.append((token.strip(), repl.strip()))
    return pairs


class Purger:
    def __init__(self, map_path: str):
        self.rules = []
        for token, repl in read_map(map_path):
            pat = re.compile(rb"(?<![A-Za-z])" + re.escape(token.encode()) + rb"(?![A-Za-z])", re.I)
            self.rules.append((pat, repl.encode()))

    def _names(self, data: bytes) -> bytes:
        for pat, repl in self.rules:
            # Keep all-caps hits: those are acronyms, not names.
            data = pat.sub(lambda m: m.group(0) if m.group(0).isupper() else repl, data)
        return data

    @staticmethod
    def _author_block(data: bytes) -> bytes:
        out, pos = [], 0
        for m in re.finditer(rb"\\author\s*\{", data):
            if m.start() < pos:
                continue
            line_start = data.rfind(b"\n", 0, m.start()) + 1
            if b"%" in data[line_start:m.start()]:
                continue  # commented out
            depth, i = 0, m.end() - 1
            while i < len(data):
                c = data[i:i + 1]
                if c == b"\\":
                    i += 2
                    continue
                if c == b"{":
                    depth += 1
                elif c == b"}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            block = data[m.start():i + 1]
            if b"Anonymous" in block:
                continue
            out += [data[pos:m.start()], b"\\author{Anonymous Authors}"]
            pos = i + 1
        return b"".join(out) + data[pos:] if out else data

    @staticmethod
    def wants(path: bytes) -> bool:
        """True if transform() may change this path. The callback checks this
        before it reads the blob, so weights and images are never loaded."""
        if path is None or SKIP.match(path):
            return False
        low = path.lower()
        name = low.rsplit(b"/", 1)[-1]
        # Files with no extension (LICENSE, Makefile) are text too.
        return low.endswith(TEXT_EXT) or low.endswith(b".xlsx") or b"." not in name

    def transform(self, path: bytes, data: bytes) -> bytes:
        if not self.wants(path):
            return data
        low = path.lower()
        if low.endswith(b".xlsx"):
            return self._xlsx(data)
        new = self._names(data)
        if low.endswith(b".tex"):
            new = self._author_block(new)
        return new

    def _xlsx(self, data: bytes) -> bytes:
        try:
            zin = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return data
        buf, changed = io.BytesIO(), False
        with zin, zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                member = zin.read(info.filename)
                if info.filename.endswith(".xml"):
                    new = self._names(member)
                    changed |= new != member
                    member = new
                zout.writestr(info, member, compress_type=info.compress_type)
        return buf.getvalue() if changed else data


def messages(map_path: str) -> None:
    """Print a --replace-message file. Capitalized and lower-case forms only,
    so acronyms in messages stay."""
    for token, repl in read_map(map_path):
        forms = sorted({token, token.capitalize(), token.lower()})
        alt = "|".join(re.escape(f) for f in forms)
        print(f"regex:(?<![A-Za-z])(?:{alt})(?![A-Za-z])==>{repl}")


def selftest(map_path: str) -> int:
    """Apply the rules to every blob reachable from any ref. Rewrites nothing.
    Prints counts only, never the tokens."""
    purger = Purger(map_path)
    objs = subprocess.run(["git", "rev-list", "--objects", "--all"],
                          capture_output=True, check=True).stdout.splitlines()
    first_path = {}
    for line in objs:
        sha, _, path = line.partition(b" ")
        if path and sha not in first_path:
            first_path[sha] = path
    check = subprocess.run(["git", "cat-file", "--batch-check"], input=b"\n".join(first_path),
                           capture_output=True, check=True).stdout.splitlines()
    blobs = [ln.split()[0] for ln in check if ln.split()[1:2] == [b"blob"]]
    changed, residual, read = Counter(), Counter(), 0
    for sha in blobs:
        path = first_path[sha]
        low = path.lower()
        if not purger.wants(path):
            continue
        data = subprocess.run(["git", "cat-file", "blob", sha], capture_output=True, check=True).stdout
        read += 1
        new = purger.transform(path, data)
        if new != data:
            changed[path.decode()] += 1
        text = new
        if low.endswith(b".xlsx"):
            with zipfile.ZipFile(io.BytesIO(new)) as z:
                text = b"".join(z.read(n) for n in z.namelist() if n.endswith(".xml"))
        left = sum(1 for pat, _ in purger.rules for m in pat.finditer(text) if not m.group(0).isupper())
        if left:
            residual[path.decode()] += left
        if low.endswith(b".tex") and (AFFILIATION.search(text) or AFFILIATION_CAPS.search(text)):
            residual[path.decode() + " (affiliation)"] += 1
    # Some paths contain a name (Grading-<name>.xlsx), so print them masked.
    def mask(p):
        return purger._names(p.encode()).decode()
    print(f"blobs read: {read}; blobs changed: {sum(changed.values())} in {len(changed)} paths")
    for p, n in sorted(changed.items()):
        print(f"  changed {n:3d}  {mask(p)}")
    print(f"residual hits after rules: {sum(residual.values())}")
    for p, n in sorted(residual.items()):
        print(f"  residual {n:3d}  {mask(p)}")
    return 1 if residual else 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("selftest", "messages"):
        sys.exit(__doc__)
    if sys.argv[1] == "messages":
        messages(sys.argv[2])
    else:
        sys.exit(selftest(sys.argv[2]))
