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
    assert json.loads((smoke_run / "shared" / "inputs" / "leakage.json").read_text())["passed"]
    for seed in cell.seeds:
        sd = smoke_run / f"seed_{seed}"
        for stage in ("traces", "filter", "train", "eval", "grade"):
            assert _manifest(sd / stage / "manifest.json")["status"] == "complete", (seed, stage)
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
    assert not (tmp_path / SMOKE / "seed_1" / "train" / "adapter_config.json").exists()
    assert (tmp_path / SMOKE / "seed_1" / "eval" / "lora.json").is_file()


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
