"""Dry-run and resume-guard tests for bodhi_lora/pipeline/traces.py. CPU only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bodhi")

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def prompts(tmp_path):
    rows = [{"prompt_id": f"q{i}", "prompt": [{"role": "user", "content": f"question {i}"}],
             "rubrics": [], "example_tags": []} for i in range(4)]
    path = tmp_path / "bench.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _gen(tmp_path, prompts, *extra):
    env = {**os.environ, "BODHI_VLLM_MODE": "subprocess", "BODHI_VLLM_TP": "1", "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-m", "bodhi_lora.pipeline.traces",
         "--model", "mistralai/Mistral-Small-24B-Instruct-2501",
         "--dataset-files", str(prompts), "--output", str(tmp_path / "raw.jsonl"), *extra],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=300)


def test_ask_first_dry_run_writes_traces_and_manifest(tmp_path, prompts):
    res = _gen(tmp_path, prompts, "--protocol", "ask_first", "--dry-run")
    assert res.returncode == 0, res.stderr[-2000:]
    assert not (tmp_path / "raw.jsonl").exists()
    rows = [json.loads(l) for l in (tmp_path / "raw.dryrun.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {r["protocol"] for r in rows} == {"ask_first"} and not rows[0]["bodhi"]
    assert rows[0]["response"] == "DRY RUN RESPONSE"
    assert rows[0]["bodhi_analysis"] == "dry run plan"
    m = json.loads((tmp_path / "raw.dryrun.manifest.json").read_text())
    assert m["stage"] == "generate" and m["protocol"] == "ask_first"
    assert m["engines"]["teacher"]["max_model_len"] == 8192
    assert m["models"]["teacher"]["revision"] == "9527884be6e5616bdd54de542f9ae13384489724"
    assert m["counts"]["n_ok"] == 2


def test_resume_refuses_a_protocol_change(tmp_path, prompts):
    # Rows from before --protocol existed carry only "bodhi": true.
    old = {"prompt_id": "q0", "model": "mistralai/Mistral-Small-24B-Instruct-2501",
           "bodhi": True, "ablate_component": "none", "response": "r"}
    resume = tmp_path / "old.jsonl"
    resume.write_text(json.dumps(old) + "\n")
    res = _gen(tmp_path, prompts, "--protocol", "ask_first", "--resume-from", str(resume))
    assert res.returncode != 0
    assert "resume config mismatch" in res.stderr


def test_use_bodhi_conflicts_with_another_protocol(tmp_path, prompts):
    res = _gen(tmp_path, prompts, "--use-bodhi", "--protocol", "ask_first", "--dry-run")
    assert res.returncode != 0 and "conflicts" in res.stderr
