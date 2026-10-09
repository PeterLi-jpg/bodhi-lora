"""Tests for the tau rule and the filter manifest in bodhi_lora/pipeline/filter.py.

The end-to-end tests resume from a fully graded file, so the grader engine
is never started and the run needs no GPU.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _legacy_tau(scores):
    """The inline rule from the posted rebuttal/launch/run_cell.sh (22nd pct).

    Copied verbatim from 61dada9:rebuttal/launch/run_cell.sh:175-186; the
    shell then passed the printed string to --min-score.
    """
    sc = sorted(s for s in scores if isinstance(s, (int, float)))
    return float(f"{sc[max(0, int(0.22 * len(sc)) - 1)] - 1e-9:.12f}" if sc else "0.0")


def _percentile_tau():
    pytest.importorskip("transformers")
    pytest.importorskip("tqdm")
    pytest.importorskip("bodhi")
    sys.path.insert(0, str(REPO))
    from bodhi_lora.pipeline.filter import percentile_tau
    return percentile_tau


def test_percentile_tau_reproduces_the_posted_rule():
    percentile_tau = _percentile_tau()
    rng = random.Random(0)
    for n in list(range(1, 60)) + list(range(100, 5000, 37)):
        # Mass points like the synthesized-rubric scores (k/21) plus noise.
        scores = [rng.choice([8 / 21, 0.5, 13 / 21]) if rng.random() < 0.6 else rng.random()
                  for _ in range(n)]
        assert percentile_tau(scores, 22) == _legacy_tau(scores), n


def test_percentile_tau_keeps_the_tied_spike():
    percentile_tau = _percentile_tau()
    scores = [8 / 21] * 80 + [0.9] * 20
    tau = percentile_tau(scores, 22)
    assert tau < 8 / 21
    assert sum(s >= tau for s in scores) == 100


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def graded_cell(tmp_path):
    rng = random.Random(1)
    rubric = [{"criterion": "c", "points": 5, "tags": []}]
    raw, bench, graded = [], [], []
    for i in range(150):
        pid = f"p{i}"
        msgs = [{"role": "user", "content": f"q{i}"}]
        raw.append({"prompt_id": pid, "messages": msgs, "response": f"r{i}"})
        bench.append({"prompt_id": pid, "prompt": msgs, "rubrics": rubric})
        score = rng.random()
        graded.append({**raw[-1], "grade": {
            "overall_score": score, "normalized_score": score,
            "parse_failures": 0, "criteria_results": [{"criterion": "c"}]}})
    _write_jsonl(tmp_path / "raw.jsonl", raw)
    _write_jsonl(tmp_path / "bench.jsonl", bench)
    _write_jsonl(tmp_path / "graded.jsonl", graded)
    (tmp_path / "exclude.json").write_text(json.dumps(["p0", "p1", "p2"]))
    return tmp_path, [g["grade"]["overall_score"] for g in graded]


def _filter(tmp, *extra):
    pytest.importorskip("transformers")
    pytest.importorskip("bodhi")
    env = {**os.environ, "BODHI_VLLM_MODE": "subprocess", "BODHI_VLLM_ACCEL": "gpu",
           "BODHI_VLLM_TP": "1", "PYTHONPATH": str(REPO)}
    out = tmp / "sft"
    res = subprocess.run(
        [sys.executable, "-m", "bodhi_lora.pipeline.filter",
         "--input", str(tmp / "raw.jsonl"), "--healthbench-data", str(tmp / "bench.jsonl"),
         "--resume-from", str(tmp / "graded.jsonl"), "--output-dir", str(out), *extra],
        capture_output=True, text=True, env=env, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:] + res.stdout[-2000:]
    assert "not starting the grader engine" in res.stdout
    return json.loads((out / "manifest.json").read_text()), res.stdout


def test_fixed_tau_is_the_default_and_is_recorded(graded_cell):
    tmp, scores = graded_cell
    m, out = _filter(tmp)
    assert m["filter"]["tau"] == 0.4 and m["filter"]["tau_rule"] == "fixed"
    assert "TAU: 0.4 (rule: fixed)" in out
    c = m["counts"]
    assert c["n_generated"] == 150 and c["n_graded"] == 150
    assert c["n_kept"] == sum(s >= 0.4 for s in scores)
    assert c["n_train"] + c["n_val"] == c["n_kept"]
    assert m["engines"]["grader"]["started"] is False
    assert m["models"]["grader"]["repo_id"] == "Qwen/Qwen2.5-14B-Instruct"
    assert {o["path"].rsplit("/", 1)[-1] for o in m["outputs"]} == {"train.jsonl", "val.jsonl"}


def test_percentile_rule_only_behind_the_flag(graded_cell):
    tmp, scores = graded_cell
    m, out = _filter(tmp, "--tau-percentile", "22")
    assert m["filter"]["tau_rule"] == "percentile"
    assert m["filter"]["tau"] == _legacy_tau(scores)
    assert m["counts"]["n_kept"] == sum(s >= m["filter"]["tau"] for s in scores)
    assert "= percentile 22 of 150 graded scores (rule: percentile" in out


def test_excluded_ids_cannot_return_through_the_resume_file(graded_cell):
    tmp, _ = graded_cell
    m, _ = _filter(tmp, "--exclude-ids", str(tmp / "exclude.json"), "--min-score", "0")
    assert m["counts"]["n_excluded"] == 3
    assert m["counts"]["n_resume_rows_dropped"] == 3
    kept_ids = set()
    for name in ("train.jsonl", "val.jsonl"):
        kept_ids |= {json.loads(l)["prompt_id"] for l in (tmp / "sft" / name).read_text().splitlines()}
    assert not kept_ids & {"p0", "p1", "p2"}
    assert len(kept_ids) == 147
