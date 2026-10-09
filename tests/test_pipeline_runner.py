"""The runner (bodhi_lora/pipeline/runner.py) on the smoke cell, and its refusals.

The stub run trains the tiny random model for real on the CPU and runs every other
stage with the stub engine, so every stage's code and every manifest is exercised.
"""
import json
import re
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("bodhi")

from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.pipeline.runner import RunError, run_cell  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from tests.fixtures.pipeline.registry import fixture_configs  # noqa: E402

SMOKE = "tiny-rubric__tiny-random-llama__bodhi"


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


@pytest.fixture(scope="module")
def smoke_run(reg, tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=root)
    return root / SMOKE


def _manifest(path):
    return json.loads(path.read_text())


def test_stub_run_writes_every_stage_and_manifest(reg, smoke_run):
    cell = reg.cells[SMOKE]
    assert _manifest(smoke_run / "manifest.json")["status"] == "finished"
    assert _manifest(smoke_run / "shared" / "traces" / "manifest.json")["counts"]["n_ok"] == 12
    probe = json.loads((smoke_run / "shared" / "probe" / "probe.json").read_text())
    assert probe["benchmark"]["n"] == probe["control"]["n"] == cell.probe.n_prompts
    assert probe["test_used"] == "fisher_exact"
    assert json.loads((smoke_run / "shared" / "inputs" / "leakage.json").read_text())["passed"]
    for seed in cell.seeds:
        sd = smoke_run / f"seed_{seed}"
        for stage in ("traces", "filter", "train", "eval", "grade"):
            assert _manifest(sd / stage / "manifest.json")["status"] == "complete", (seed, stage)
        if seed in cell.bench.seeds:
            bench = json.loads((sd / "bench" / "bench.json").read_text())
            assert set(bench["conditions"]) == set(cell.eval.conditions)
            assert bench["max_new_tokens"] == cell.eval.max_new_tokens
        else:
            assert not (sd / "bench").exists()
        assert (sd / "train" / "adapter_config.json").is_file()
        assert (sd / "train" / "adapter_model.safetensors").is_file()
        assert _manifest(sd / "train" / "manifest.json")["counts"]["global_step"] == 2
        for condition in cell.eval.conditions:
            out = json.loads((sd / "eval" / f"{condition}.json").read_text())
            assert out["config"] == condition and out["n_examples"] == cell.eval.ids.n
            r = out["results"][0]
            assert r["condition"] == condition and r["calls"] >= 1 and "finish_length" in r
        grades = json.loads((sd / "grade" / "epistemic__llama-3.1-8b.json").read_text())
        assert [c["name"] for c in grades["configs"]] == cell.eval.conditions


def test_per_seed_filter_splits_differ_by_seed(smoke_run):
    a = (smoke_run / "seed_1" / "filter" / "val.jsonl").read_text()
    b = (smoke_run / "seed_2" / "filter" / "val.jsonl").read_text()
    assert a and b and a != b


def test_a_rerun_resumes_and_refuses_changed_inputs(reg, smoke_run, capsys):
    root = smoke_run.parent
    run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=root)
    out = capsys.readouterr().out
    assert out.count("complete, skipping") >= 12
    pools = smoke_run / "shared" / "inputs" / "pools.json"
    data = json.loads(pools.read_text())
    data["eval"] = data["eval"][1:]
    pools.write_text(json.dumps(data, indent=2) + "\n")
    with pytest.raises(RunError, match="differs"):
        run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=root)


def test_dry_run_prints_gpu_commands_and_does_not_train(reg, tmp_path, capsys):
    run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="dry_run", results_root=tmp_path, seeds=[1])
    out = capsys.readouterr().out
    assert "DRY-RUN [gpu 0]: CUDA_VISIBLE_DEVICES=0 BODHI_VLLM_PORT=8000" in out
    # shlex quotes the interpreter path when the checkout path has a space.
    assert re.search(r"\.venv/bin/python'? -m bodhi_lora\.pipeline\.train --cell", out)
    assert re.search(r"\.venv-infer/bin/python'? -m bodhi_lora\.pipeline\.eval", out)
    # The plan comes before any stub step runs.
    assert out.index("DRY-RUN") < out.index("prepare:")
    assert not (tmp_path / SMOKE / "seed_1" / "train" / "adapter_config.json").exists()
    assert (tmp_path / SMOKE / "seed_1" / "eval" / "lora.json").is_file()


def test_stub_and_dry_runs_never_write_under_the_results_root(reg, tmp_path, capsys):
    """Their manifests say complete; under the results root a GPU run would skip those stages."""
    proj = replace(project(CONFIGS_DIR), results_root=tmp_path / "results",
                   scratch_root=tmp_path / "scratch")
    run_cell(reg, proj, SMOKE, mode="dry_run", seeds=[1], gpus=[0, 3])
    out = capsys.readouterr().out
    assert not (tmp_path / "results").exists()
    assert (tmp_path / "scratch" / "dry_run" / SMOKE / "seed_1" / "eval" / "base.json").is_file()
    # The printed GPU commands are those of a GPU run: no scratch folder in them.
    assert "--results-root" not in "".join(line for line in out.splitlines() if "DRY-RUN [gpu" in line)


def test_judges_limits_the_judge_steps(reg, tmp_path):
    with pytest.raises(RunError, match="not judges"):
        run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=tmp_path, judges=["phi-9"])


RECORDED = "tiny-rubric__tiny-random-llama__bodhi__2026-01"


def test_a_recorded_cell_only_gets_judge_regrades_that_read_its_stored_replies(tmp_path, capsys):
    configs = fixture_configs(tmp_path)
    reg, proj = load_registry(configs), project(configs)
    cell = reg.cells[RECORDED]
    stored = proj.cell(RECORDED)
    rows = [json.loads(line) for line in (CONFIGS_DIR.parent / "tests/fixtures/pipeline/tiny_rubric.jsonl")
            .read_text().splitlines()]
    for seed in cell.seeds:
        for condition in cell.eval.conditions:
            path = stored.eval_file(seed, condition)
            path.parent.mkdir(parents=True, exist_ok=True)
            results = [{"prompt_id": r["prompt_id"], "response": f"stored reply {seed} {condition}",
                        "score": 0.5, "criteria_results": [
                            {"criterion": c["criterion"], "points": c["points"], "tags": c["tags"],
                             "criteria_met": c["points"] > 0} for c in r["rubrics"]]}
                       for r in rows[:4]]
            path.write_text(json.dumps({"config": condition, "results": results}))
    before = sorted(str(p) for p in stored.root.rglob("*"))
    with pytest.raises(RunError, match="records a past run"):
        run_cell(reg, proj, RECORDED, mode="dry_run", configs_dir=configs)
    run_cell(reg, proj, RECORDED, mode="dry_run", stages=("probe", "grade"), configs_dir=configs)
    out = capsys.readouterr().out
    assert "--judge medgemma-27b-text" in out and "bodhi_lora.pipeline.probe" in out
    assert sorted(str(p) for p in stored.root.rglob("*")) == before
    scratch = proj.scratch_root / "dry_run" / RECORDED
    judged = json.loads((scratch / "seed_2" / "grade" / "judge__medgemma-27b-text.json").read_text())
    assert [c["name"] for c in judged["configs"]] == cell.eval.conditions
    first = judged["configs"][0]["examples"][0]
    assert first["original_score"] == 0.5
    # The judge grades the criteria stored in the eval file and keeps the evaluator's verdicts.
    assert first["evaluator_criteria_met"] == [p > 0 for p in first["criteria_points"]]
    assert len(first["judge_criteria_met"]) == len(first["criteria_points"])
    assert first["harm"]["harm_extent"] == "none" and judged["configs"][0]["harm"]["n_scored"] == 4
    assert not (scratch / "seed_2" / "grade" / "epistemic__llama-3.1-8b.json").exists()
    assert json.loads((scratch / "shared" / "probe" / "probe.json").read_text())["benchmark"]["n"] == 4


def test_recorded_and_external_cells_are_refused(reg, tmp_path):
    proj = project(CONFIGS_DIR)
    with pytest.raises(RunError, match="records a past run"):
        run_cell(reg, proj, "healthbench__mistral-small-24b__bodhi__2026-07", mode="dry_run")
    with pytest.raises(RunError, match="trained outside this package"):
        run_cell(reg, proj, "healthbench__medgemma-27b-text__bodhi__2026-05", mode="dry_run",
                 results_root=tmp_path, stages=("train",))
    with pytest.raises(RunError, match="--gpus"):
        run_cell(reg, proj, SMOKE, mode="gpu", results_root=tmp_path)
    missing = replace(proj, python_train=tmp_path / "no-python")
    with pytest.raises(RunError, match="is missing"):
        run_cell(reg, missing, SMOKE, mode="gpu", gpus=[0], results_root=tmp_path)
    with pytest.raises(RunError, match="not seeds"):
        run_cell(reg, proj, SMOKE, mode="stub", seeds=[99], results_root=tmp_path)
