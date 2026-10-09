"""Dry-run tests for eval_healthbench.py and eval_epistemic.py.

The dry run stubs both vLLM engines, so these tests exercise argument
handling, the eval conditions, the eval-setting defaults, the result
metadata and the manifest on a CPU, with no model download.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tqdm")

REPO = Path(__file__).resolve().parent.parent
RUBRIC = [{"criterion": "Answers the question.", "points": 5, "tags": ["axis:completeness"]},
          {"criterion": "Gives dangerous advice.", "points": -5, "tags": ["axis:accuracy"]}]


@pytest.fixture
def bench(tmp_path):
    rows = [{"prompt_id": f"bench-{i}", "prompt": [{"role": "user", "content": f"question {i}"}],
             "rubrics": RUBRIC, "example_tags": ["theme:test"]} for i in range(5)]
    path = tmp_path / "bench.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"prompt_ids": ["bench-0", "bench-1", "bench-2"]}))
    return path, ids


def _eval(tmp_path, bench, *extra):
    path, ids = bench
    out = tmp_path / "eval" / "result.json"
    res = subprocess.run(
        [sys.executable, "-m", "bodhi_lora.pipeline.eval",
         "--model", "mistralai/Mistral-Small-24B-Instruct-2501",
         "--benchmark-jsonl", str(path), "--sample-ids", str(ids),
         "--output", str(out), "--dry-run", *extra],
        capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    return out


def test_defaults_equal_the_submitted_eval(tmp_path, bench):
    out = _eval(tmp_path, bench)
    assert not out.exists(), "a dry run must never write --output"
    result = json.loads(out.with_name("result.dryrun.json").read_text())
    md = result["run_metadata"]
    assert result["config"] == "base_no_wrapper"
    assert (md["max_model_len"], md["max_num_seqs"], md["concurrency"],
            md["request_timeout_s"]) == (4096, 32, 32, 900)
    assert md["grader_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert md["grader_revision"] == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert md["eval_ids_n"] == 3
    assert result["n_examples"] == 2

    manifest = json.loads(out.with_name("result.dryrun.manifest.json").read_text())
    assert manifest["stage"] == "eval"
    assert manifest["engines"]["grader"]["max_model_len"] == 4096
    assert manifest["models"]["policy"]["family"] == "mistral"
    assert manifest["eval_ids"]["n"] == 3
    assert manifest["counts"]["n_graded"] == 2


@pytest.mark.parametrize("flags,tag", [
    (["--use-bodhi"], "base_bodhi"),
    (["--wrapper", "ask_first"], "base_ask_first"),
    (["--baseline", "clarify"], "base_clarify"),
    (["--baseline", "verbalized_confidence"], "base_verbalized_confidence"),
])
def test_conditions_write_their_tag(tmp_path, bench, flags, tag):
    if "--use-bodhi" in flags:
        pytest.importorskip("bodhi")
    out = _eval(tmp_path, bench, *flags)
    result = json.loads(out.with_name("result.dryrun.json").read_text())
    assert result["config"] == tag
    if flags[0] == "--baseline":
        # Mistral-Small accepts a system message, so the baseline uses one.
        first_call = result["dry_run_policy_calls"][0]
        assert first_call[0]["role"] == "system"
        assert result["instruction_placement"] == "system"


def test_quantized_base_and_lora_reach_the_engine_settings(tmp_path, bench):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    out = _eval(tmp_path, bench, "--lora-path", str(adapter), "--base-quantization", "bitsandbytes")
    result = json.loads(out.with_name("result.dryrun.json").read_text())
    assert result["config"] == "lora_no_wrapper"
    assert result["run_metadata"]["base_quantization"] == "bitsandbytes"
    manifest = json.loads(out.with_name("result.dryrun.manifest.json").read_text())
    assert manifest["engines"]["policy"]["quantization"] == "bitsandbytes"
    assert manifest["adapter"]["r"] == 16


def test_wrapper_and_baseline_conflict_is_refused(tmp_path, bench):
    path, ids = bench
    res = subprocess.run(
        [sys.executable, "-m", "bodhi_lora.pipeline.eval", "--model", "m",
         "--benchmark-jsonl", str(path), "--sample-ids", str(ids),
         "--output", str(tmp_path / "o.json"), "--dry-run",
         "--use-bodhi", "--baseline", "clarify"],
        capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300)
    assert res.returncode != 0 and "--baseline" in res.stderr


def test_epistemic_dry_run_uses_the_cell_prompts(tmp_path, bench):
    path, _ = bench
    responses = tmp_path / "base_no_wrapper.json"
    responses.write_text(json.dumps({"config": "base_no_wrapper", "results": [
        {"prompt_id": "bench-0", "response": "Rest."},
        {"prompt_id": "bench-1", "response": "Drink water."}]}))
    out = tmp_path / "epistemic_scores.json"
    res = subprocess.run(
        [sys.executable, "-m", "bodhi_lora.grading.epistemic",
         "--response-files", str(responses), "--benchmark-jsonl", str(path),
         "--healthbench-data", str(tmp_path / "absent.jsonl"),
         "--output", str(out), "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    result = json.loads((tmp_path / "epistemic_scores.dryrun.json").read_text())
    assert result["configs"][0]["aggregates"]["n_scored"] == 2
    assert result["run_metadata"]["max_model_len"] == 4096
    manifest = json.loads((tmp_path / "epistemic_scores.dryrun.manifest.json").read_text())
    assert manifest["models"]["grader"]["revision_source"] == "registry"
    assert not (tmp_path / "absent.jsonl").exists(), "must not download HealthBench"
