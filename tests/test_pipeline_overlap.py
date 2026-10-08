"""Tests for the leakage gate in scripts/check_dataset_overlap.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_dataset_overlap.py"


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _bench_row(i, text=None):
    return {"prompt_id": f"b{i}", "prompt": [{"role": "user", "content": text or f"Question number {i}"}]}


def _train_row(pid, text):
    return {"prompt_id": pid, "messages": [{"role": "user", "content": text}], "response": "r"}


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture
def cell(tmp_path):
    bench = _jsonl(tmp_path / "bench.jsonl", [_bench_row(i) for i in range(20)])
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"prompt_ids": [f"b{i}" for i in range(10)]}))
    seed_ids = tmp_path / "seed_42.json"
    seed_ids.write_text(json.dumps({"prompt_ids": ["b1", "b2"]}))
    return tmp_path, bench, pool, seed_ids


def _bench_check(tmp, bench, pool, train_rows, *extra):
    train = _jsonl(tmp / "train.jsonl", train_rows)
    out = tmp / "overlap.json"
    res = _run("--benchmark-jsonl", bench, "--holdout-ids", pool, "--train-jsonl", train,
               "--json-out", out, *extra)
    return res, json.loads(out.read_text())


def test_clean_cell_passes(cell):
    tmp, bench, pool, seed_ids = cell
    res, report = _bench_check(tmp, bench, pool,
                               [_train_row(f"b{i}", f"Question number {i}") for i in range(10, 20)],
                               "--eval-ids-files", seed_ids)
    assert res.returncode == 0, res.stderr
    assert report["passed"] and report["id_leak"] == 0 and report["text_duplicates"] == 0
    assert report["eval_files"][0]["outside_pool"] == 0


def test_id_leak_fails_and_is_recorded(cell):
    tmp, bench, pool, _ = cell
    res, report = _bench_check(tmp, bench, pool, [_train_row("b3", "Question number 3")])
    assert res.returncode != 0 and report["passed"] is False and report["id_leak"] == 1


def test_same_text_under_another_id_fails(cell):
    tmp, bench, pool, _ = cell
    res, report = _bench_check(tmp, bench, pool, [_train_row("other-id", "  question   NUMBER 4 ")])
    assert res.returncode != 0 and report["text_duplicates"] == 1
    assert report["text_duplicates_first"] == ["other-id"]


def test_eval_file_outside_the_pool_fails(cell):
    tmp, bench, pool, _ = cell
    bad = tmp / "seed_7.json"
    bad.write_text(json.dumps(["b1", "b15"]))
    res, report = _bench_check(tmp, bench, pool, [_train_row("b12", "Question number 12")],
                               "--eval-ids-files", bad)
    assert res.returncode != 0 and report["eval_files"][0]["outside_pool"] == 1


def test_healthbench_mode_catches_a_text_duplicate(tmp_path):
    hard = [_bench_row(i) for i in range(5)]
    full = hard + [_bench_row(i) for i in range(5, 12)]
    _jsonl(tmp_path / "hard.jsonl", hard)
    _jsonl(tmp_path / "full.jsonl", full)
    (tmp_path / "fixed.json").write_text(json.dumps(["b0", "b1"]))
    train = _jsonl(tmp_path / "train.jsonl",
                   [_train_row("b7", "Question number 7"), _train_row("copy", "QUESTION number 2")])
    out = tmp_path / "overlap.json"
    res = _run("--healthbench", tmp_path / "full.jsonl", "--healthbench-hard", tmp_path / "hard.jsonl",
               "--eval-ids", tmp_path / "fixed.json", "--train-jsonl", train, "--json-out", out)
    report = json.loads(out.read_text())
    assert res.returncode != 0
    assert report["hard_id_leak"] == 0 and report["text_duplicates"] == 1
