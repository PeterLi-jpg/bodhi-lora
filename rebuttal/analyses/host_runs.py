"""The 2026-07 runs of the GPU host, found in the results root.

The host named each run <bench>__<tag>, where <tag> is the HF repo with "/", ":" and "."
replaced by "_", and the analyses of the discussion key their outputs by that name.
scripts/paper_stats.py reads those keys (Appendix J run history), so the analyses keep
them. The runs now live in cell folders (results/PROVENANCE.md). Each cell kept the
host's logs, and its generation log gen_<bench>_<tag>.log names the run.
"""

import json
from pathlib import Path


def host_runs(results_root: Path) -> dict[str, Path]:
    """Run name -> cell folder, for every cell that holds a host generation log."""
    runs = {}
    for log in sorted(Path(results_root).glob("*/logs/gen_*.log")):
        bench, _, tag = log.stem.removeprefix("gen_").partition("_")
        name = f"{bench}__{tag}"
        if name in runs:
            raise SystemExit(f"host_runs: {name} has a generation log in two cells: "
                             f"{runs[name].name} and {log.parents[1].name}")
        runs[name] = log.parents[1]
    return dict(sorted(runs.items()))


def host_folders(results_root: Path) -> set[str]:
    """The runs that wrote a folder on the host (results_rebuttal/<run>/ before the move).

    A run that stopped before it wrote one left only logs. Only the move's record,
    provenance.json, still tells the two apart.
    """
    entries = json.loads((Path(results_root) / "provenance.json").read_text())
    return {e["old"].split("/")[1] for e in entries if e["old"].startswith("results_rebuttal/")}
