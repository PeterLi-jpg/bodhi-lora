"""The paper's numbers, tables and figures from the analysis outputs.

The interface below is fixed for the analysis lane, which builds it. Until then
`make numbers` and `make figures` run scripts/paper_stats.py and
scripts/build_paper_figures.py.
"""
from pathlib import Path


def report(reg, report_cfg: dict, out_dir: Path) -> None:
    """Analysis outputs -> paper/generated/."""
    raise NotImplementedError("the analysis lane builds bodhi_lora.report.report")
