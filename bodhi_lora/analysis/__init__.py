"""The uniform analysis of every cell (docs/ARCHITECTURE.md section 7).

    extract_cell(reg, cell_id) raw stage outputs -> each seed's table.csv and the cell's
                               prompts.csv (extract.py)
    analyze_cell(reg, cell_id) the tables -> the cell's analysis/stats.json (cell.py)
    analyze_cross(reg, cfg)    every cell's stats.json -> <results root>/cross_cell/stats.json
                               (cross.py)

The project roots default to configs/project.yaml and the report config to
configs/report.yaml; tests pass their own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from bodhi_lora.analysis.config import ReportConfig, load_report_config
from bodhi_lora.paths import CONFIGS_DIR, Project, project as load_project


def report_config(configs_dir: Path = CONFIGS_DIR) -> ReportConfig:
    return load_report_config(Path(configs_dir) / "report.yaml")


# Not named extract: importing the submodule bodhi_lora.analysis.extract sets the
# package attribute of that name to the module, which would hide a function called extract.
def extract_cell(reg, cell_id: str, proj: Optional[Project] = None, cfg: Optional[ReportConfig] = None) -> dict:
    """Raw stage outputs of a cell -> each seed's table.csv."""
    from bodhi_lora.analysis.extract import extract as _extract

    return _extract(reg, cell_id, proj or load_project(), cfg or report_config())


def analyze_cell(reg, cell_id: str, proj: Optional[Project] = None, cfg: Optional[ReportConfig] = None) -> dict:
    """A cell's tables -> its analysis/stats.json."""
    from bodhi_lora.analysis.cell import analyze

    return analyze(reg, cell_id, proj or load_project(), cfg or report_config())


def analyze_cross(reg, report_cfg: Optional[ReportConfig] = None, proj: Optional[Project] = None) -> dict:
    """Every cell's stats.json -> the cross-cell statistics."""
    from bodhi_lora.analysis.cross import analyze_cross as _cross

    return _cross(reg, proj or load_project(), report_cfg or report_config())


def analyzable(reg, cfg: ReportConfig) -> list[str]:
    """The cells whose status the report config analyzes, in id order."""
    return sorted(cid for cid, c in reg.cells.items() if c.report.status in cfg.statuses)
