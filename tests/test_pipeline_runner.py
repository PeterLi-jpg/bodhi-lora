"""The runner (bodhi_lora/pipeline/runner.py) on the smoke cell, and its refusals.

The stub run trains the tiny random model for real on the CPU and runs every other
stage with the stub engine, so every stage's code and every manifest is exercised.
"""
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

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
    # The real training path ran the reply-loss check (train.py check_replies_in_loss).
    train_logs = sorted((smoke_run / "logs").glob("train*.log"))
    assert len(train_logs) == len(cell.seeds)
    assert all("Reply tokens in the loss: checked on" in f.read_text() for f in train_logs)


def test_runner_manifests_record_the_stage_env(smoke_run):
    """Roll-up and pointer manifests record the env that ran the stage, and the cell
    manifest records both envs. Schema 2 stamped the runner's own env on them, which on a
    two-env install named vLLM 0.9.2 for stages that ran under 0.10.2."""
    cell = _manifest(smoke_run / "manifest.json")
    assert cell["schema"] == 3 and set(cell["packages"]) == {"infer", "train"}
    sd = smoke_run / "seed_1"
    assert _manifest(sd / "traces" / "manifest.json")["packages"] == \
        _manifest(smoke_run / "shared" / "traces" / "manifest.json")["packages"]
    part = _manifest(next((sd / "eval").glob("*.manifest.json")))
    assert _manifest(sd / "eval" / "manifest.json")["packages"] == part["packages"]


def test_in_a_gpu_run_the_runner_asks_each_stage_interpreter(tmp_path):
    from types import SimpleNamespace
    from bodhi_lora.pipeline.runner import Runner
    fakes = {}
    for env, version in (("infer", "0.10.2"), ("train", "0.9.2")):
        fake = tmp_path / f"python-{env}"
        fake.write_text(f"#!/bin/sh\necho '{{\"python\": \"3.12.3\", \"vllm\": \"{version}\"}}'\n")
        fake.chmod(0o755)
        fakes[env] = fake
    proj = SimpleNamespace(python_infer=fakes["infer"], python_train=fakes["train"])
    runner = Runner(SimpleNamespace(project=proj), "gpu", [0])
    assert runner.packages("infer")["vllm"] == "0.10.2"
    assert runner.packages("train")["vllm"] == "0.9.2"
    # the cpu filter split runs inside the runner, so its env is the runner's
    from bodhi_lora import manifest as mf
    assert runner.packages("cpu") == mf.package_versions()


def test_per_seed_filter_splits_differ_by_seed(smoke_run):
    a = (smoke_run / "seed_1" / "filter" / "val.jsonl").read_text()
    b = (smoke_run / "seed_2" / "filter" / "val.jsonl").read_text()
    assert a and b and a != b


def test_probe_text_stays_out_of_git(reg, smoke_run):
    """Open decision O8: probe.json holds the rates and the test, and git tracks it; the
    probed text and the continuations go to samples.jsonl, which git ignores under
    results/. The bench output (times and token counts) is tracked."""
    cell = reg.cells[SMOKE]
    folder = smoke_run / "shared" / "probe"
    probe = json.loads((folder / "probe.json").read_text())
    assert all("samples" not in probe[name] for name in ("benchmark", "control"))
    lines = [json.loads(line) for line in (folder / probe["samples_file"]).read_text().splitlines()]
    assert len(lines) == 2 * cell.probe.n_prompts
    assert {"set", "leak_prefix", "ground_truth_continuation", "model_output"} <= set(lines[0])
    for name in ("benchmark", "control"):
        assert sum(x["exact_match"] for x in lines if x["set"] == name) == probe[name]["n_exact_match"]
    outputs = _manifest(folder / "manifest.json")["outputs"]
    assert sorted(Path(o["path"]).name for o in outputs) == ["probe.json", "samples.jsonl"]
    # The same files under the results root: the text is ignored, the summaries are tracked.
    root = Path(__file__).resolve().parent.parent
    for rel, ignored in (("shared/probe/samples.jsonl", True), ("shared/probe/probe.json", False),
                         ("shared/probe/manifest.json", False), ("seed_1/bench/bench.json", False)):
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", f"results/{SMOKE}/{rel}"], cwd=root)
        assert (r.returncode == 0) == ignored, rel


def test_a_rerun_resumes_and_refuses_changed_inputs(reg, smoke_run, capsys):
    root = smoke_run.parent
    run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=root)
    out = capsys.readouterr().out
    # Every step skips: probe, traces and filter; filter split, train and grade per seed;
    # the one eval step of both seeds; the bench of seed 1.
    assert out.count("complete, skipping") == 11
    pools = smoke_run / "shared" / "inputs" / "pools.json"
    data = json.loads(pools.read_text())
    data["eval"] = data["eval"][1:]
    pools.write_text(json.dumps(data, indent=2) + "\n")
    with pytest.raises(RunError, match="differs"):
        run_cell(reg, project(CONFIGS_DIR), SMOKE, mode="stub", results_root=root)


def test_the_frozen_record_is_compared_value_by_value(reg, tmp_path):
    """A key the schema gained after a cell first ran takes its default on both sides, so
    a resume does not stop on it; a changed value, or a key the schema refuses, still does."""
    from bodhi_lora.pipeline.runner import same_record

    cell = reg.cells[SMOKE]
    path = tmp_path / "cell.yaml"

    def frozen(change):
        data = cell.model_dump(mode="json")
        change(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return same_record(path, cell)

    assert frozen(lambda d: None)
    assert frozen(lambda d: d["eval"]["serving"].pop("enforce_eager"))
    assert frozen(lambda d: d["eval"].pop("shared_server"))
    assert not frozen(lambda d: d["eval"].update(shared_server=False))
    assert not frozen(lambda d: d["eval"].update(max_new_tokens=32))
    assert not frozen(lambda d: d["eval"]["serving"].update(enforce_eager=True))
    assert not frozen(lambda d: d.update(unknown_key=1))


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
    # Naming the results root, or a folder in it, does not get around the rule.
    for root in (tmp_path / "results", tmp_path / "results" / "sub"):
        with pytest.raises(RunError, match="may not write under the results root"):
            run_cell(reg, proj, SMOKE, mode="stub", seeds=[1], results_root=root)
    assert not (tmp_path / "results").exists()


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
    assert first["evaluator_criteria_parse_failed"] == [False] * len(first["criteria_points"])
    assert first["harm"]["harm_extent"] == "none" and judged["configs"][0]["harm"]["n_scored"] == 4
    assert not (scratch / "seed_2" / "grade" / "epistemic__llama-3.1-8b.json").exists()
    assert json.loads((scratch / "shared" / "probe" / "probe.json").read_text())["benchmark"]["n"] == 4
    manifest = json.loads((scratch / "seed_2" / "grade" / "judge__medgemma-27b-text.manifest.json").read_text())
    assert manifest["engines"]["judge"]["max_model_len"] == 2048   # grading.judge_serving


def test_a_recorded_cell_dry_run_prints_its_plan_before_it_fetches(tmp_path, capsys, monkeypatch):
    """Offline, the probe's control benchmark cannot be fetched; the GPU commands of the
    judge re-grade and the probe must be printed before that stops the run."""
    from bodhi_lora.pipeline import runner

    def offline(ctx, stages, fetch=True):
        raise ConnectionError("offline")

    monkeypatch.setattr(runner, "fetch_benchmarks", offline)
    configs = fixture_configs(tmp_path)
    with pytest.raises(ConnectionError, match="offline"):
        run_cell(load_registry(configs), project(configs), RECORDED, mode="dry_run",
                 stages=("probe", "grade"), configs_dir=configs)
    out = capsys.readouterr().out
    assert "--judge medgemma-27b-text" in out and "bodhi_lora.pipeline.probe" in out


def test_a_request_too_long_for_the_context_is_listed_not_retried(tmp_path, monkeypatch):
    """A too-long request fails the same way on every rerun, so it must not leave the
    stage incomplete; any other failed request must."""
    from bodhi_lora.engine import ContextLengthError
    from bodhi_lora.pipeline import grade
    from bodhi_lora.pipeline.context import StageContext

    configs = fixture_configs(tmp_path)
    reg, proj = load_registry(configs), project(configs)
    cell = reg.cells[RECORDED]
    rows = [json.loads(line) for line in (CONFIGS_DIR.parent / "tests/fixtures/pipeline/tiny_rubric.jsonl")
            .read_text().splitlines()][:4]
    for condition in cell.eval.conditions:
        path = proj.cell(RECORDED).eval_file(1, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": condition, "results": [
            {"prompt_id": r["prompt_id"], "response": f"reply to {r['prompt_id']}", "score": 0.5}
            for r in rows]}))
    first = rows[0]["prompt_id"]
    real = grade.grade_dimensions

    def raising(exc):
        def grade_dimensions(grader, text, response):
            if response == f"reply to {first}":
                raise exc
            return real(grader, text, response)
        return grade_dimensions

    ctx = StageContext(reg=reg, project=proj, cell=cell, paths=proj.cell(RECORDED), stub=True)
    monkeypatch.setattr(grade, "grade_dimensions",
                        raising(ContextLengthError("This model's maximum context length is 2048 tokens.")))
    grade.run(ctx, 1, "medgemma-27b-text")
    out = proj.cell(RECORDED).judge_file(1, "medgemma-27b-text")
    for c in json.loads(out.read_text())["configs"]:
        assert (c["n_too_long"], c["too_long_ids"], c["n_grade_errors"], len(c["examples"])) == (1, [first], 0, 3)
    manifest = out.parent / "judge__medgemma-27b-text.manifest.json"
    assert json.loads(manifest.read_text())["status"] == "complete"

    monkeypatch.setattr(grade, "grade_dimensions", raising(RuntimeError("read timed out")))
    with pytest.raises(SystemExit, match="run the stage again"):
        grade.run(ctx, 1, "medgemma-27b-text")
    assert json.loads(manifest.read_text())["status"] == "incomplete"


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


def test_a_dry_run_never_downloads_and_names_the_fetch_command(tmp_path, capsys, monkeypatch):
    """A dry run writes only scratch files: a benchmark that is not local stops it after
    the plan, with the fetch command, and no download starts (G14 of the 2026-10-11 audit)."""
    import urllib.request

    def no_download(*args, **kwargs):
        raise AssertionError("a dry run tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    configs = fixture_configs(tmp_path)          # its data root is empty
    cell = "healthbench__mistral-small-24b__bodhi"
    with pytest.raises(RunError, match="python -m bodhi_lora fetch healthbench"):
        run_cell(load_registry(configs), project(configs), cell, mode="dry_run", seeds=[42],
                 configs_dir=configs)
    assert "DRY-RUN [gpu" in capsys.readouterr().out
