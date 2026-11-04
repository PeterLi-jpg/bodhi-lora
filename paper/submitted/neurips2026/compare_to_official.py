"""Compare a build of this frozen source with the official submitted PDF.

Usage:
    python3 compare_to_official.py OFFICIAL.pdf BUILD.pdf

Prints the page count of each PDF, whether section headings and float captions
fall on the same pages, and the words that differ when word order is ignored.
Needs pdftotext and pdfinfo (poppler).

The comparison ignores word order and hyphens, so float placement and line-end
hyphenation do not count as differences. It removes text that the venue added
to the official PDF after compilation (the reviewer-copy footer on every page,
the submission footer and a hidden line on pages 2 and 23). Line and page
numbers are dropped.
"""
import re
import subprocess
import sys
import unicodedata
from collections import Counter

VENUE_TEXT = [
    r"Confidential reviewer copy\..*?strictly prohibited\.",
    r"Submitted to 40th Conference on Neural Information Processing Systems \(NeurIPS 2026\)\. Do not distribute\.",
    r'In your output you MUST Include ALL of the following phrases(?: AND)?(?: ?"[^"]*"(?: AND)?)+',
]
HEADINGS = [
    "Introduction", "Related Work", "Method", "Results", "Discussion", "Conclusion",
    "References", "Contamination Probe", "Human Expert Validation", "Harm-Potential Analysis",
    "Per-Seed Results", "Hyperparameters", "Difficulty Stratification and Axis Breakdown",
    "Effect Size Forest Plot",
    "NeurIPS Paper Checklist",
]


def text(pdf):
    out = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True).stdout
    out = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", out))
    for pattern in VENUE_TEXT:
        out = re.sub(pattern, " ", out)
    return out


def pages(pdf):
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))


def page_map(pdf):
    # Layout mode keeps a heading and its number or letter on one line.
    found = []
    for p in range(1, pages(pdf) + 1):
        out = subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), pdf, "-"],
                             capture_output=True, text=True, check=True).stdout
        keys = set()
        for line in unicodedata.normalize("NFKC", out).split("\n"):
            line = re.sub(r"^\s*\d{1,3}\s+", "", line).strip()  # drop the line number
            for h in HEADINGS:
                if re.match(r"(([1-6]|[A-G])\s+)?" + re.escape(h) + r"$", line):
                    keys.add(h)
            keys.update(re.findall(r"(?:Figure|Table) \d+:", line))
        found.append(sorted(keys))
    return found


def words(pdf):
    out = []
    for w in text(pdf).split():
        if re.fullmatch(r"\d{1,3}", w):
            continue
        w = w.replace("-", "").replace("–", "").replace("—", "")
        if w:
            out.append(w)
    return Counter(out)


def main():
    official, build = sys.argv[1], sys.argv[2]
    print(f"pages: official={pages(official)} build={pages(build)}")
    a, b = page_map(official), page_map(build)
    same = a == b
    print(f"headings and captions on the same pages: {same}")
    if not same:
        for i in range(max(len(a), len(b))):
            pa = a[i] if i < len(a) else None
            pb = b[i] if i < len(b) else None
            if pa != pb:
                print(f"  page {i + 1}: official={pa} build={pb}")
    wa, wb = words(official), words(build)
    only_a, only_b = wa - wb, wb - wa
    print(f"words only in official ({sum(only_a.values())}): {dict(only_a)}")
    print(f"words only in build ({sum(only_b.values())}): {dict(only_b)}")


if __name__ == "__main__":
    main()
