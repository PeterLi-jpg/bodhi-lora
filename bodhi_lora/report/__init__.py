"""The report: every number the paper prints, from the uniform analysis
(docs/ARCHITECTURE.md section 7).

    report(reg, cfg, proj) -> Report    every analyzed cell's stats.json and the
                                        cross-cell stats.json -> macros, tables, record
    write(rep, out_dir)                 paper/generated/{numbers.tex, numbers.json, tab_*.tex}

`python -m bodhi_lora report` (make numbers) runs both. It reads only the analysis
outputs, never the raw outputs, and treats every cell the same way: a new cell needs a
record and its stats.json, not code. Macro names follow one rule. A cell's macros start
with its record's report.token, so \\statHBMistralJulActiveInquiryLoRAvsBaseDiff is the
LoRA - Base difference in active inquiry of the cell whose token is HBMistralJul; a
group's with Group<token> (configs/report.yaml). A value the analysis does not have gets
no macro, and a table shows "--".

Tables:
    tab_cell_<token>.tex     every analyzed cell (bodhi_lora/report/cell.py)
    tab_<kind>_<token>.tex   the detail tables of each cell in report.detail_cells
    tab_<name>.tex           each table of report.cross_tables, one row per cross cell
    tab_cell_detail_<n>.tex  every cross cell's metrics by condition (report.cell_detail_tables)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.paths import Project
from bodhi_lora.registry import Registry
from bodhi_lora.report.cell import cell_macros, cell_table, detail_tables
from bodhi_lora.report.cross import (cell_detail_tables, count_macros, cross_table, grader_macros, group_macros,
                                     posted_macros)
from bodhi_lora.report.latex import MISSING


@dataclass
class Report:
    # (macro name without the \stat prefix, value, source comment), in emission order.
    macros: list[tuple[str, str, str]] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    _names: set = field(default_factory=set)

    def add(self, name: str, value: str, source: str) -> None:
        if value == MISSING:
            return
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"a macro name is letters only: {name!r}")
        if name in self._names:
            raise ValueError(f"macro \\stat{name} is made twice; two tokens collide")
        self._names.add(name)
        self.macros.append((name, value, source))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(reg: Registry, cfg: ReportConfig, proj: Project) -> Report:
    """Macros, tables and a record from every analyzed cell's stats.json and the
    cross-cell statistics. A cell of cross_cells without its stats.json, or a missing
    cross-cell stats.json, stops the report: the paper cannot be complete without them."""
    from bodhi_lora.analysis import analyzable

    rep = Report()
    order = list(cfg.cross_cells) + [c for c in analyzable(reg, cfg) if c not in cfg.cross_cells]
    all_stats, cells_rec, missing, inputs = {}, {}, {}, {}
    for cid in order:
        path = proj.cell(cid).analysis() / "stats.json"
        if not path.is_file():
            if cid in cfg.cross_cells or cid in cfg.detail_cells:
                raise FileNotFoundError(f"{proj.rel(path)} is missing; run make analyze")
            missing[cid] = "no stats.json (the cell has no outputs in this checkout)"
            continue
        stats = json.loads(path.read_text())
        cell = reg.cells[cid]
        src = proj.rel(path)
        all_stats[cid] = stats
        inputs[src] = _sha256(path)
        cell_macros(rep, stats, cell, cfg, src)
        rep.tables[f"tab_cell_{cell.report.token}"] = cell_table(stats, cell, cfg, src)
        if cid in cfg.detail_cells:
            rep.tables.update(detail_tables(stats, cell, cfg, src))
        cells_rec[cid] = {"token": cell.report.token, "label": cell.report.label, "stats": src,
                          "in_cross_cells": cid in cfg.cross_cells, "detail": cid in cfg.detail_cells,
                          "set_aside": cell.report.set_aside, "family_waiver": cell.grading.same_family_waiver}
    cross_path = proj.results_root / "cross_cell" / "stats.json"
    if not cross_path.is_file():
        raise FileNotFoundError(f"{proj.rel(cross_path)} is missing; run make analyze")
    cross = json.loads(cross_path.read_text())
    src = proj.rel(cross_path)
    inputs[src] = _sha256(cross_path)
    members = {gid: g["cells"] for gid, g in cross["groups"].items()}
    group_macros(rep, cross, all_stats, cfg, src)
    count_macros(rep, all_stats, members, reg, cfg, cross["groups"])
    grader_macros(rep, cross, reg, cfg, src)
    posted_macros(rep, cross, cfg, src)
    every = f"{proj.rel(proj.results_root)}/*/analysis/stats.json"
    for name, spec in cfg.cross_tables.items():
        rep.tables[name] = cross_table(spec, all_stats, members, reg, cfg, every)
    rep.tables.update(cell_detail_tables(all_stats, members, reg, cfg, f"{every} (conditions, contrasts)"))
    rep.record = {"cells": cells_rec, "not_reported": missing, "inputs": dict(sorted(inputs.items())),
                  "tables": sorted(rep.tables), "macros": len(rep.macros)}
    return rep


def write(rep: Report, out_dir: Path) -> list[str]:
    """numbers.tex, numbers.json and every table into out_dir. A table the report no
    longer makes is removed, so no stale table survives. Returns the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out_dir.glob("tab_*.tex")):
        if stale.stem not in rep.tables:
            stale.unlink()
    lines = ["% Generated by python -m bodhi_lora report (make numbers). Do not edit by hand.",
             "% Each macro carries a comment naming the statistic it comes from.", ""]
    lines += [f"\\newcommand{{\\stat{name}}}{{{value}}} % {source}" for name, value, source in rep.macros]
    (out_dir / "numbers.tex").write_text("\n".join(lines) + "\n")
    (out_dir / "numbers.json").write_text(json.dumps(rep.record, indent=2, sort_keys=True) + "\n")
    for name, text in sorted(rep.tables.items()):
        (out_dir / f"{name}.tex").write_text(text)
    return ["numbers.tex", "numbers.json"] + [f"{name}.tex" for name in sorted(rep.tables)]
