"""The uniform analysis of every cell (docs/ARCHITECTURE.md section 7).

The interfaces below are fixed for the analysis lane, which builds them. Until then the
paper's numbers come from scripts/paper_stats.py and the analyses in rebuttal/analyses/.
"""


def extract(reg, cell_id: str) -> None:
    """Raw stage outputs of a cell -> each seed's table.csv."""
    raise NotImplementedError("the analysis lane builds bodhi_lora.analysis.extract")


def analyze_cell(reg, cell_id: str) -> dict:
    """A cell's tables -> its analysis/stats.json."""
    raise NotImplementedError("the analysis lane builds bodhi_lora.analysis.analyze_cell")


def analyze_cross(reg, report_cfg: dict) -> dict:
    """Every cell's stats.json -> the cross-cell statistics."""
    raise NotImplementedError("the analysis lane builds bodhi_lora.analysis.analyze_cross")
