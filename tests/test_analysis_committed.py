"""Every committed results/<cell>/analysis/stats.json is what analyze writes from the
cell's tracked tables, and the cross-cell stats.json is what analyze_cross writes from
them. A change to the analysis code that is not followed by `make analyze` fails here."""
import json
from dataclasses import replace

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from bodhi_lora.analysis import analyzable, report_config  # noqa: E402
from bodhi_lora.analysis.cell import analyze_tables, load_tables, round6  # noqa: E402
from bodhi_lora.analysis.cross import analyze_cross  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

REG = load_registry(CONFIGS_DIR)
CFG = report_config()
PROJ = project(CONFIGS_DIR)
CELLS = [c for c in analyzable(REG, CFG) if (PROJ.cell(c).analysis() / "stats.json").is_file()]


@pytest.mark.parametrize("cell_id", CELLS)
def test_committed_cell_stats_are_current(cell_id):
    paths = PROJ.cell(cell_id)
    table, prompts, meta = load_tables(REG.cells[cell_id], paths)
    fresh = round6(analyze_tables(REG, REG.cells[cell_id], table, prompts, meta, paths, CFG))
    committed = json.loads((paths.analysis() / "stats.json").read_text())
    assert json.loads(json.dumps(fresh, sort_keys=True)) == committed


def test_committed_cross_cell_stats_are_current(tmp_path):
    committed_path = PROJ.results_root / "cross_cell" / "stats.json"
    if not committed_path.is_file() or not all(c in CELLS for c in CFG.cross_cells):
        pytest.skip("the cross-cell statistics are not in this checkout")
    # analyze_cross writes under results_root/cross_cell; point it at a copy of what it reads.
    for cid in CFG.cross_cells:
        dst = tmp_path / cid / "analysis" / "stats.json"
        dst.parent.mkdir(parents=True)
        dst.write_text((PROJ.cell(cid).analysis() / "stats.json").read_text())
    for gc in CFG.grader_comparisons:
        for cid in gc.cells:
            src = PROJ.cell(cid).table(gc.seed)
            if src.is_file():
                dst = tmp_path / src.relative_to(PROJ.results_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())
    fresh = analyze_cross(REG, replace(PROJ, results_root=tmp_path), CFG)
    assert json.loads(json.dumps(fresh, sort_keys=True)) == json.loads(committed_path.read_text())


@pytest.mark.parametrize("cell_id", CELLS)
def test_prompts_tables_hold_only_the_listed_labels(cell_id):
    """The plan keeps the labels that configs/report.yaml does not list (the physician
    labels among them) unseen until its adoption, so no tracked prompts.csv carries them
    next to the outcome tables (docs/ANALYSIS_PLAN.md section 12, 2026-10-10)."""
    listed = {lab for spec in CFG.discrimination for lab in spec.labels}
    header = (PROJ.cell(cell_id).analysis() / "prompts.csv").read_text().split("\n", 1)[0].split(",")
    extra = sorted(c for c in header if c.startswith("label_") and c.removeprefix("label_") not in listed)
    assert not extra, f"{cell_id}: prompts.csv has unlisted label columns {extra}"
