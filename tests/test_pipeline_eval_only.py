"""Eval-only cells (adapters_from) on the stub engine: the MediQ-style paired probe and
clinician questions with no rubric, after the smoke cell's traces and filter ran in the
same results root. They check the adapter leakage gate, pair draws and rubric-less rows."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("bodhi")

from bodhi_lora.paths import project  # noqa: E402
from bodhi_lora.pipeline.runner import RunError, run_cell  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from tests.fixtures.pipeline.registry import fixture_configs  # noqa: E402

SMOKE = "tiny-rubric__tiny-random-llama__bodhi"
PAIRED = "tiny-paired__tiny-random-llama__bodhi"
QUESTIONS = "tiny-questions__tiny-random-llama__bodhi"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    configs = fixture_configs(tmp_path_factory.mktemp("eval_only"))
    reg, proj = load_registry(configs), project(configs)
    return reg, proj, configs


def test_an_eval_only_cell_needs_its_adapter_cell_first(env):
    reg, proj, configs = env
    with pytest.raises(RunError, match="kept traces are what the adapters trained on"):
        run_cell(reg, proj, QUESTIONS, mode="stub", configs_dir=configs, results_root=proj.results_root / "empty")


def test_paired_and_question_cells_run_after_the_adapter_cell(env):
    reg, proj, configs = env
    root = proj.results_root
    run_cell(reg, proj, SMOKE, mode="stub", stages=("traces", "filter"), configs_dir=configs, results_root=root)
    for cell_id in (PAIRED, QUESTIONS):
        run_cell(reg, proj, cell_id, mode="stub", stages=("eval", "grade"), configs_dir=configs,
                 results_root=root)
        gate = json.loads((root / cell_id / "shared" / "inputs" / "leakage.json").read_text())
        assert gate["passed"] and gate["n_trained_prompts"] == 12

    paired = reg.cells[PAIRED]
    ids = json.loads((root / PAIRED / "shared" / "inputs" / "eval_ids_seed_1.json").read_text())
    assert ids["unit"] == "pair" and len(ids["prompt_ids"]) == 2 * paired.eval.ids.n
    lora = json.loads((root / PAIRED / "seed_1" / "eval" / "lora.json").read_text())
    assert {r["prompt_id"].rsplit("-", 1)[1] for r in lora["results"]} == {"complete", "incomplete"}

    q = json.loads((root / QUESTIONS / "seed_1" / "eval" / "base.json").read_text())
    assert all(r["score"] is None and r["criteria_results"] == [] for r in q["results"])
    assert q["mean"] is None and q["grader_rubric_items_total"] == 0
    judged = json.loads((root / QUESTIONS / "seed_1" / "grade" / "judge__medgemma-27b-text.json").read_text())
    ex = judged["configs"][0]["examples"][0]
    assert ex["judge_score"] is None and ex["judge_criteria_met"] == [] and ex["harm"] is not None


def test_a_traces_from_cell_reuses_the_source_grades_and_a_sampled_cell_records_its_decoding(env):
    reg, proj, configs = env
    root = proj.results_root
    unfiltered = "tiny-rubric__tiny-random-llama__bodhi__unfiltered"
    if not (root / SMOKE / "shared" / "filter" / "graded.jsonl").is_file():
        run_cell(reg, proj, SMOKE, mode="stub", stages=("traces", "filter"), configs_dir=configs,
                 results_root=root)
    run_cell(reg, proj, unfiltered, mode="stub", stages=("traces", "filter"), configs_dir=configs,
             results_root=root)
    m = json.loads((root / unfiltered / "shared" / "filter" / "manifest.json").read_text())
    assert m["grades_from"] == SMOKE and m["counts"]["n_grades_reused"] == m["counts"]["n_graded"] == 12
    assert m["tau"]["rule"] == "none" and m["counts"]["n_kept"] == 12

    sampled = "tiny-rubric__tiny-random-llama__bodhi__traces-s1"
    run_cell(reg, proj, sampled, mode="stub", stages=("traces",), configs_dir=configs, results_root=root)
    t = json.loads((root / sampled / "shared" / "traces" / "manifest.json").read_text())
    assert t["config"]["traces"]["decoding"] == "sample" and t["config"]["traces"]["seed"] == 1
