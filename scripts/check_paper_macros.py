"""Check that every \\stat macro the paper uses is defined in paper/generated/numbers.tex.

`make numbers` leaves out the macros of a section whose input is missing. LaTeX then
stops on the first undefined one, with a message that does not name the input. This
check names every missing macro at once, and it needs no TeX install, so CI can run it.

Usage:
    python3 scripts/check_paper_macros.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Files that use \stat macros. The figure source is built on its own, but reads the
# same numbers.tex.
SOURCES = ("paper/main.tex", "paper/checklist.tex", "paper/figures/pipeline_standalone.tex")
NUMBERS = ROOT / "paper" / "generated" / "numbers.tex"


def main() -> int:
    defined = set(re.findall(r"\\newcommand\{\\(stat[A-Za-z]+)\}", NUMBERS.read_text()))
    missing = {}
    for rel in SOURCES:
        path = ROOT / rel
        if not path.is_file():
            continue
        # Drop TeX comments first, so a macro named only in a comment does not count.
        text = re.sub(r"(?<!\\)%.*", "", path.read_text())
        for name in sorted(set(re.findall(r"\\(stat[A-Za-z]+)", text)) - defined):
            missing.setdefault(name, []).append(rel)
    if missing:
        for name, where in sorted(missing.items()):
            print(f"undefined \\{name} (used in {', '.join(where)})")
        print(f"check_paper_macros: {len(missing)} macro(s) not in {NUMBERS.name}; "
              "a make numbers input is probably missing (see paper/generated/README.md)")
        return 1
    print(f"check_paper_macros: OK, every \\stat macro used is defined ({len(defined)} defined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
