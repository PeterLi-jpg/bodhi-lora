"""From a run to the paper's numbers, on outputs the package's own stages wrote.

The smoke cell and the paired probe cell run on the stub engine into a scratch results
root. Then extract, analyze, the cross-cell step and the report run on those outputs,
with a report config that puts both cells in the cross-cell tables. This is the path a
new cell takes: no step may need a code change for it (docs/ARCHITECTURE.md section 7).
"""
import json
from dataclasses import replace

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("bodhi")
pytest.importorskip("pandas")

from bodhi_lora import analysis  # noqa: E402
from bodhi_lora.analysis.config import load_report_config  # noqa: E402
from bodhi_lora.paths import project  # noqa: E402
from bodhi_lora.pipeline.runner import run_cell  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from bodhi_lora.report import report, write  # noqa: E402
from tests.fixtures.pipeline.registry import fixture_configs  # noqa: E402

SMOKE = "tiny-rubric__tiny-random-llama__bodhi"
PAIRED = "tiny-paired__tiny-random-llama__bodhi"


def _report_config(configs):
    """The real report config, with the two stub cells as the cross cells. Groups that
    list recorded cells, and the counts over them, have nothing to count here."""
    data = yaml.safe_load((configs / "report.yaml").read_text())
    data["statuses"] = ["planned"]
    data["cross_cells"] = [SMOKE, PAIRED]
    data["detail_cells"] = [SMOKE]
    data["groups"] = {"all": {"token": "All", "description": "every stub cell", "where": {"status": ["planned"]}},
                      "kept": {"token": "Kept", "description": "not set aside", "where": {"set_aside": False}}}
    data["counts"] = [{"token": "AskRose", "group": "all", "contrast": "lora - base",
                       "metrics": ["active_inquiry"], "direction": "up"}]
    data["compares"], data["ranges"], data["grader_comparisons"] = [], [], []
    data["posted"] = None
    data["families"] = []
    for table in data["cross_tables"].values():
        table["blocks"] = ["kept"] if table.get("blocks") else []
        table["leave_out"] = []
    data["cell_detail_tables"]["blocks"] = ["kept"]
    (configs / "report.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return load_report_config(configs / "report.yaml")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    configs = fixture_configs(tmp_path_factory.mktemp("e2e"))
    reg, proj = load_registry(configs), project(configs)
    # A stub run may not write under the results root (runner), so the cells share a scratch root.
    root = proj.scratch_root / "stub"
    run_cell(reg, proj, SMOKE, mode="stub", stages=("traces", "filter", "eval", "grade"), configs_dir=configs,
             results_root=root)
    run_cell(reg, proj, PAIRED, mode="stub", stages=("eval", "grade", "classify"), configs_dir=configs,
             results_root=root)
    proj = replace(proj, results_root=root)
    cfg = _report_config(configs)
    for cid in (SMOKE, PAIRED):
        analysis.extract_cell(reg, cid, proj, cfg)
        analysis.analyze_cell(reg, cid, proj, cfg)
    analysis.analyze_cross(reg, cfg, proj)
    rep = report(reg, cfg, proj)
    out = tmp_path_factory.mktemp("generated")
    write(rep, out)
    return reg, proj, cfg, rep, out


def _stats(proj, cid):
    return json.loads((proj.cell(cid).analysis() / "stats.json").read_text())


def test_the_stages_write_what_the_analysis_reads(run):
    reg, proj, cfg, rep, out = run
    paths = proj.cell(SMOKE)
    result = json.loads(paths.eval_file(1, "base+mediq-abstain").read_text())["results"][0]
    assert "pair_id" in result and result["protocol_metadata"]["protocol"] == "mediq-abstain"
    scores = json.loads((paths.shared("filter") / "scores.json").read_text())
    assert scores["columns"] == ["prompt_id", "score"] and scores["rows"]
    header = paths.table(1).read_text().split("\n", 1)[0].split(",")
    assert header[:5] == ["cell", "seed", "condition", "prompt_id", "pair_id"]
    assert any(c.startswith("leak_ask-first__") for c in header) and any(c.startswith("leak_bodhi__") for c in header)


def test_the_judge_classifier_answers_for_every_reply(run):
    """R17's stage on the stub engine: one answer per stored reply of every condition, the
    template's hash recorded, and the gate marked as not passed."""
    reg, proj, cfg, rep, out = run
    judge = reg.cells[PAIRED].grading.judges[0]
    for seed in reg.cells[PAIRED].seeds:
        data = json.loads((proj.cell(PAIRED).stage(seed, "classify") / f"classify__{judge}.json").read_text())
        assert data["gate"].startswith("not passed") and len(data["template_sha256"]) == 64
        for c in data["configs"]:
            n = len(json.loads(proj.cell(PAIRED).eval_file(seed, c["name"]).read_text())["results"])
            assert len(c["examples"]) == c["rates"]["n_parsed"] == n > 0
            assert c["rates"]["asks_user_rate"] == 0.0 and c["rates"]["urgent_care_no_share"] == 1.0


def test_an_eval_only_cell_reports_the_filter_of_its_adapters(run):
    """An adapters_from cell trained nothing itself; its filter section is the source
    cell's, and says so (GEN-12 of the 2026-10-11 audit)."""
    reg, proj, cfg, rep, out = run
    filt = _stats(proj, PAIRED)["filter"]
    assert filt["from_cell"] == SMOKE
    assert filt["graded_scores"]["n"] == _stats(proj, SMOKE)["filter"]["graded_scores"]["n"]


def test_the_analysis_makes_every_statistic_the_records_ask_for(run):
    reg, proj, cfg, rep, out = run
    s = _stats(proj, SMOKE)
    # The declared contrast of the record (H6), after the rule's.
    assert s["contrasts_order"][-1] == "lora+red-flags-first - lora+bodhi"
    assert "lora+red-flags-first - lora+bodhi" in s["leakage"]["leak_bodhi__analysis-format"]["contrasts"]
    assert s["filter"]["graded_scores"]["n"] == len(json.loads(
        (proj.cell(SMOKE).shared("filter") / "scores.json").read_text())["rows"])
    paired = _stats(proj, PAIRED)["paired_probe"]["active_inquiry"]["incomplete"]
    assert set(paired["conditions"]) == set(reg.cells[PAIRED].eval.conditions)
    assert sum(paired["conditions"]["base"]["n_pairs"].values()) > 0
    assert "did" in paired["contrasts"]["lora - base"]


def test_a_new_cell_reaches_table_one_its_macros_and_the_counts(run):
    reg, proj, cfg, rep, out = run
    t = reg.cells[SMOKE].report.token
    table = (out / "tab_cross_cell.tex").read_text()
    for cid in (SMOKE, PAIRED):
        bench, model = reg.cells[cid].report.label.rsplit(", ", 1) if ", " in reg.cells[cid].report.label \
            else (reg.cells[cid].report.label, "")
        assert bench.replace("_", "\\_") in table, cid
    macros = {name for name, _, _ in rep.macros}
    assert {f"{t}ActiveInquiryLoRAvsBaseDiff", f"{t}RubricScoreLoRAvsBaseBootCI", "GroupAllCells",
            "GroupAllAskRose"} <= macros
    for kind in ("conditions", "dimensions", "contrasts", "counts", "worst_case", "bootstrap", "gap"):
        assert (out / f"tab_{kind}_{t}.tex").is_file(), kind
    assert (out / "tab_cell_detail_1.tex").is_file() and (out / f"tab_cell_{t}.tex").is_file()
    assert (out / "numbers.tex").read_text().count("\\newcommand") == len(rep.macros)
