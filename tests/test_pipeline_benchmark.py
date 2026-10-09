"""Tests for dataset pinning, seeded selection and holdouts. No download."""

import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bodhi_lora.data import loaders as bb  # noqa: E402


def fake_chatdoctor(n=50):
    rows = [{"input": f"patient question {i}", "output": f"doctor reply {i}"} for i in range(n)]
    rows.insert(5, {"input": "", "output": "no question"})       # unusable
    rows.insert(9, {"input": "patient question 3", "output": "dup"})  # duplicate
    return rows


def legacy_head(ds, benchmark, max_rows):
    """The loop of build_benchmark_jsonl.py at 61dada9 (first --max rows)."""
    build = bb.BUILDERS[benchmark]
    seen, out = set(), []
    for ex in ds:
        built = build(ex)
        if built is None:
            continue
        question, user, rubrics, tags = built
        pid = bb.stable_id(benchmark, question)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if max_rows and len(out) >= max_rows:
            break
    return out


def test_head_selection_reproduces_the_posted_builder():
    ds = fake_chatdoctor()
    rows, counts = bb.build_rows(ds, "chatdoctor", 20, select="head")
    assert [r["prompt_id"] for r in rows] == legacy_head(ds, "chatdoctor", 20)
    assert counts["n_written"] == 20


def test_seeded_selection_is_a_deterministic_sample_of_the_whole_dataset():
    ds = fake_chatdoctor()
    a, counts = bb.build_rows(ds, "chatdoctor", 20, select="seeded", select_seed=0)
    b, _ = bb.build_rows(list(reversed(ds)), "chatdoctor", 20, select="seeded", select_seed=0)
    assert [r["prompt_id"] for r in a] == [r["prompt_id"] for r in b]
    assert counts == {"n_source_rows": 52, "n_unusable": 1, "n_duplicates": 1, "n_written": 20}
    head = set(legacy_head(ds, "chatdoctor", 20))
    assert {r["prompt_id"] for r in a} != head, "seeded pool should not be the head"
    c, _ = bb.build_rows(ds, "chatdoctor", 20, select="seeded", select_seed=1)
    assert [r["prompt_id"] for r in a] != [r["prompt_id"] for r in c]


def test_make_holdout_methods():
    ids = [f"id{i:03d}" for i in range(100)]
    assert bb.make_holdout(ids, 10, "head") == ids[:10]
    seeded = bb.make_holdout(ids, 10, "seeded", seed=0)
    assert seeded == sorted(seeded) and set(seeded) <= set(ids) and len(seeded) == 10
    assert seeded == bb.make_holdout(ids, 10, "seeded", seed=0)
    with pytest.raises(SystemExit):
        bb.make_holdout(ids, 101, "head")


def _run_main(monkeypatch, argv, ds):
    calls = {}

    def load_dataset(ds_id, split, revision=None):
        calls.update(ds_id=ds_id, split=split, revision=revision)
        return ds

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(sys, "argv", ["build_benchmark_jsonl.py", *argv])
    runpy.run_module("bodhi_lora.data.loaders", run_name="__main__")
    return calls


def test_main_pins_revision_and_writes_holdout_and_manifest(tmp_path, monkeypatch):
    out = tmp_path / "chatdoctor_s0.jsonl"
    calls = _run_main(monkeypatch, ["--benchmark", "chatdoctor", "--out", str(out),
                                    "--max", "30", "--holdout-n", "10"], fake_chatdoctor())
    assert calls["revision"] == bb.DEFAULT_REVISIONS["lavita/ChatDoctor-HealthCareMagic-100k"]
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 30
    holdout = json.loads((tmp_path / "chatdoctor_s0_holdout_ids.json").read_text())
    assert holdout["method"] == "seeded" and len(holdout["prompt_ids"]) == 10
    assert set(holdout["prompt_ids"]) <= {r["prompt_id"] for r in rows}
    m = json.loads((tmp_path / "chatdoctor_s0.manifest.json").read_text())
    assert m["dataset"]["revision_source"] == "pinned"
    assert m["holdout_ids"]["n"] == 10
    assert m["counts"]["n_written"] == 30
    assert m["outputs"][0]["sha256"] and m["outputs"][0]["n_lines"] == 30


def test_holdout_only_reproduces_the_posted_first_n_rule(tmp_path, monkeypatch):
    out = tmp_path / "chatdoctor.jsonl"
    _run_main(monkeypatch, ["--benchmark", "chatdoctor", "--out", str(out), "--max", "30",
                            "--select", "head"], fake_chatdoctor())
    ids_out = tmp_path / "cell" / "eval_ids.json"
    _run_main(monkeypatch, ["--benchmark", "chatdoctor", "--out", str(out), "--holdout-only",
                            "--holdout-n", "8", "--holdout-method", "head",
                            "--holdout-out", str(ids_out)], [])
    # The inline rule of run_cell.sh at 61dada9: first N prompt_ids of the file.
    first_n = [json.loads(l)["prompt_id"] for l in out.read_text().splitlines() if l.strip()][:8]
    assert json.loads(ids_out.read_text())["prompt_ids"] == first_n


def test_bootstrap_draw_from_a_pool(tmp_path):
    pool = tmp_path / "pool.json"
    ids = [f"bench-{i:03d}" for i in range(50)]
    pool.write_text(json.dumps({"prompt_ids": ids}))
    outs = []
    for name in ("a.json", "b.json"):
        res = subprocess.run(
            [sys.executable, "-m", "bodhi_lora.data.splits",
             "--pool-ids", str(pool), "--seed", "42", "--size", "20",
             "--output", str(tmp_path / name)], capture_output=True, text=True, cwd=REPO)
        assert res.returncode == 0, res.stderr
        outs.append(json.loads((tmp_path / name).read_text()))
    assert outs[0]["prompt_ids"] == outs[1]["prompt_ids"]
    assert set(outs[0]["prompt_ids"]) <= set(ids) and len(outs[0]["prompt_ids"]) == 20
    assert "held-out pool" in outs[0]["description"]
