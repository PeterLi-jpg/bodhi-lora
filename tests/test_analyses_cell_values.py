"""Tests for rebuttal/analyses/cell_values.py.

The eval files of the added cells are copies from the shared GPU host (PROV-1). The
test builds a results root whose added cell holds the submitted cell's committed files,
so the script runs unchanged on the real copies.
"""

import copy
import importlib.util
import json
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
POSTED = ROOT / "rebuttal" / "posted_values.json"
M = ROOT / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05"


def _load():
    spec = importlib.util.spec_from_file_location(
        "cell_values", ROOT / "rebuttal" / "analyses" / "cell_values.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_cell(root, cell_id):
    """The submitted cell's eval and grade files linked into <root>/<cell_id>/."""
    for seed_dir in sorted(M.glob("seed_*")):
        for stage in ("eval", "grade"):
            d = root / cell_id / seed_dir.name / stage
            d.mkdir(parents=True)
            for f in sorted((seed_dir / stage).glob("*.json")):
                if f.name != "manifest.json":
                    (d / f.name).symlink_to(f)


def test_every_posted_cell_has_a_registry_cell():
    cv = _load()
    posted = {c["id"] for c in json.loads(POSTED.read_text())["cells"]}
    assert set(cv.POSTED_CELL) == posted
    assert all((ROOT / "configs" / "cells" / f"{c}.yaml").is_file() for c in cv.POSTED_CELL.values())


def test_submitted_cell_matches_posted_and_missing_cells_are_named(tmp_path):
    cv = _load()
    out = tmp_path / "generated_values.json"
    res = cv.run(POSTED, tmp_path / "no_cells", {"healthbench__medgemma-27b": M}, out)
    assert [c["id"] for c in res["cells"]] == ["healthbench__medgemma-27b"]
    assert len(res["cells"][0]["diff_vs_posted"]) == 8
    assert res["mismatches"] == []
    assert len(res["missing"]) == 7
    assert res["cells"][0]["seeds"] == [7, 13, 42, 99, 101]
    # Before the sync there is no added cell, so no file is left in the tree (G9).
    assert not out.exists()


def test_synced_cell_layout_is_read_and_a_wrong_posted_value_is_flagged(tmp_path):
    cv = _load()
    posted = json.loads(POSTED.read_text())
    main = next(c for c in posted["cells"] if c["id"] == "healthbench__medgemma-27b")
    # A synced Mistral cell whose files are, for the test, the MedGemma outputs. The
    # posted values are MedGemma's, with one LoRA aggregate changed to a wrong value.
    fake = copy.deepcopy(main)
    fake["id"] = "healthbench__mistral-small-24b"
    fake["conditions"]["lora"]["aggregate"]["mean"] = 0.411
    posted["cells"] = [fake]
    p = tmp_path / "posted.json"
    p.write_text(json.dumps(posted))
    root = tmp_path / "results"
    _fake_cell(root, "healthbench__mistral-small-24b__bodhi__2026-07")

    res = cv.run(p, root, {}, tmp_path / "gv.json")
    cell = res["cells"][0]
    # A folder outside the repository is named by its last part only (no absolute paths).
    assert cell["dir"] == "healthbench__mistral-small-24b__bodhi__2026-07"
    assert abs(cell["conditions"]["base"]["aggregate"]["mean"] - 0.459) <= 0.0005
    assert [(m["condition"], m["metric"]) for m in res["mismatches"]] == [("lora", "aggregate")]
    # Askers minus non-askers among LoRA's graded responses: every scored LoRA response
    # (985 in the submitted cell) falls in one of the two groups.
    gap = cell["lora_askers_minus_non_askers"]
    assert gap["n_ask"] + gap["n_no_ask"] == 985
    assert gap["seeds_with_grades"] == [7, 13, 42, 99, 101]
    assert abs(gap["gap"]) < 0.05
    # The paired LoRA - Base test runs on the unrounded seed means, before the output
    # rounds them to 6 significant digits.
    agg = cell["conditions"]
    seeds = [str(s) for s in cell["seeds"]]
    lora = [agg["lora"]["aggregate"]["per_seed"][s] for s in seeds]
    base = [agg["base"]["aggregate"]["per_seed"][s] for s in seeds]
    assert cell["aggregate_lora_minus_base"]["p"] == stats.ttest_rel(lora, base).pvalue
    written = json.loads((tmp_path / "gv.json").read_text())["cells"][0]["aggregate_lora_minus_base"]
    assert written["p"] == float(f"{cell['aggregate_lora_minus_base']['p']:.6g}")

