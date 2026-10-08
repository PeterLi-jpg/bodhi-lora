"""Dry-run tests for scripts/regrade_cross_family.py. CPU only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tqdm")
pytest.importorskip("bodhi")

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "regrade_cross_family.py"
RUBRIC = [{"criterion": "Answers.", "points": 5, "tags": []},
          {"criterion": "Is dangerous.", "points": -5, "tags": []}]


@pytest.fixture
def med42_cell(tmp_path):
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(json.dumps({
        "prompt_id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}],
        "rubrics": RUBRIC}) + "\n" for i in range(4)))
    cell = tmp_path / "healthbench__m42"
    for seed in (42, 7):
        ev = cell / f"seed_{seed}" / "eval"
        ev.mkdir(parents=True)
        for cfg in ("base_no_wrapper", "lora_bodhi"):
            (ev / f"{cfg}.json").write_text(json.dumps({
                "config": cfg, "model": "m42-health/Llama3-Med42-8B",
                "run_metadata": {"grader_model": "NousResearch/Meta-Llama-3.1-8B-Instruct"},
                "results": [{"prompt_id": f"p{i}", "response": f"r{i}", "score": 0.5}
                            for i in range(3)]}))
        # Files the regrade must not pick up as inputs.
        (ev / "epistemic_scores.json").write_text("{}")
        (ev / "base_no_wrapper.manifest.json").write_text("{}")
    return cell, bench


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, timeout=300)


def test_default_judge_regrades_a_med42_cell(med42_cell):
    cell, bench = med42_cell
    res = _run("--cell", cell, "--benchmark-jsonl", bench, "--dry-run")
    assert res.returncode == 0, res.stderr[-2000:]
    for seed in (42, 7):
        out = cell / f"seed_{seed}" / "eval" / "regrade_google__gemma-2-27b-it.dryrun.json"
        report = json.loads(out.read_text())
        assert report["judge_model"] == "google/gemma-2-27b-it"
        assert report["family_check"]["passed"] is True
        names = sorted(c["name"] for c in report["configs"])
        assert names == ["base_no_wrapper", "lora_bodhi"]
        cfg = report["configs"][0]
        assert cfg["n_regraded"] == 2 and cfg["epistemic"]["n_scored"] == 2
        assert {"original_score", "judge_score", "scores"} <= set(cfg["examples"][0])
        manifest = json.loads(out.with_name(out.stem + ".manifest.json").read_text())
        assert manifest["models"]["judge"]["revision_is_commit"]
        assert manifest["engines"]["judge"]["max_model_len"] == 8192
    # The original result files are untouched.
    data = json.loads((cell / "seed_42" / "eval" / "lora_bodhi.json").read_text())
    assert len(data["results"]) == 3


def test_same_family_judge_is_refused(med42_cell):
    cell, bench = med42_cell
    for judge in ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                  "m42-health/Llama3-Med42-8B"):
        res = _run("--cell", cell, "--benchmark-jsonl", bench, "--dry-run", "--judge", judge)
        assert res.returncode != 0 and "REFUSED" in res.stderr, judge


def test_gemma_judge_is_refused_for_a_medgemma_cell(tmp_path):
    ev = tmp_path / "cell" / "seed_42" / "eval"
    ev.mkdir(parents=True)
    (ev / "base_no_wrapper.json").write_text(json.dumps({
        "config": "base_no_wrapper", "model": "google/medgemma-27b-text-it",
        "run_metadata": {"grader_model": "meta-llama/Llama-3.1-8B-Instruct"}, "results": []}))
    bench = tmp_path / "b.jsonl"
    bench.write_text("")
    res = _run("--cell", tmp_path / "cell", "--benchmark-jsonl", bench, "--dry-run")
    assert res.returncode != 0 and "base:google/medgemma-27b-text-it" in res.stderr
