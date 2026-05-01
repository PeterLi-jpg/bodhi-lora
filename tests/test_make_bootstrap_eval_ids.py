"""Tests for scripts/make_bootstrap_eval_ids.py (audit finding B9).

The script generates a deterministic per-seed 200-prompt subset of the
HealthBench Hard pool for the multi-seed bootstrap eval protocol (issue #60).
Determinism is what the preflight leakage gate in check_dataset_overlap.py
relies on: same seed -> same draw, every run, every machine. These tests pin
that contract plus the small handful of CLI behaviours the four launchers
(scripts/run_multi_seed.sh, tpu/launch_5seeds*.sh, slurm/eval_lora.sh) depend on.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_module():
    # Re-import each time so a previous test's argparse state doesn't leak.
    sys.modules.pop("scripts.make_bootstrap_eval_ids", None)
    return importlib.import_module("scripts.make_bootstrap_eval_ids")


def _write_pool(path: Path, n: int) -> None:
    """Write n synthetic HealthBench Hard rows; only prompt_id matters."""
    lines = [json.dumps({"prompt_id": f"hb_hard_{i:04d}"}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")


def _run(monkeypatch, *, pool, output, seed, size=None, force=False):
    argv = [
        "make_bootstrap_eval_ids.py",
        "--healthbench-jsonl", str(pool),
        "--seed", str(seed),
        "--output", str(output),
    ]
    if size is not None:
        argv += ["--size", str(size)]
    if force:
        argv += ["--force"]
    monkeypatch.setattr(sys, "argv", argv)
    mod = _import_module()
    mod.main()


def test_determinism_same_seed(monkeypatch, tmp_path):
    """Same seed + same pool -> bit-identical prompt_ids list."""
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"

    _run(monkeypatch, pool=pool, output=out_a, seed=42, size=10)
    _run(monkeypatch, pool=pool, output=out_b, seed=42, size=10)

    a = json.loads(out_a.read_text())
    b = json.loads(out_b.read_text())
    assert a["prompt_ids"] == b["prompt_ids"]


def test_different_seeds_differ(monkeypatch, tmp_path):
    """Different seeds should produce different draws (with high probability
    on a 10-of-50 sample; the choice of seeds 42 vs 7 is fixed so this is
    deterministic, not flaky).
    """
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out_a = tmp_path / "seed42.json"
    out_b = tmp_path / "seed7.json"

    _run(monkeypatch, pool=pool, output=out_a, seed=42, size=10)
    _run(monkeypatch, pool=pool, output=out_b, seed=7, size=10)

    a = json.loads(out_a.read_text())
    b = json.loads(out_b.read_text())
    assert a["prompt_ids"] != b["prompt_ids"]


def test_size_flag(monkeypatch, tmp_path):
    """--size flag controls draw size; size field in JSON matches."""
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=17)

    data = json.loads(out.read_text())
    assert len(data["prompt_ids"]) == 17
    assert data["size"] == 17


def test_output_schema(monkeypatch, tmp_path):
    """All keys the launchers / downstream consumers depend on must be present."""
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=10)

    data = json.loads(out.read_text())
    expected = {"description", "seed", "size", "total_pool", "source", "prompt_ids"}
    assert expected <= set(data.keys())
    assert data["seed"] == 42
    assert data["total_pool"] == 50
    assert data["source"] == str(pool)


def test_skips_when_file_exists_without_force(monkeypatch, tmp_path, capsys):
    """Idempotency: re-running without --force must not rewrite the file.

    Launchers call this script every run; if it weren't idempotent it could
    silently regenerate (and thus change) the eval set.
    """
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=10)
    original = out.read_text()

    # Run again with a DIFFERENT seed; if --force were implied, the contents
    # would change. Without --force we expect the file untouched.
    _run(monkeypatch, pool=pool, output=out, seed=7, size=10)
    after = out.read_text()

    assert original == after
    out_text = capsys.readouterr().out
    assert "skipping" in out_text.lower()


def test_force_overwrites(monkeypatch, tmp_path):
    """--force regenerates: a different seed yields different content."""
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=10)
    first = json.loads(out.read_text())

    _run(monkeypatch, pool=pool, output=out, seed=7, size=10, force=True)
    second = json.loads(out.read_text())

    assert first["seed"] == 42
    assert second["seed"] == 7
    assert first["prompt_ids"] != second["prompt_ids"]


def test_errors_when_size_exceeds_pool(monkeypatch, tmp_path):
    """Asking for more prompts than exist must abort hard, not silently
    truncate. The script raises SystemExit with a descriptive message.
    """
    pool = tmp_path / "tiny.jsonl"
    _write_pool(pool, 5)
    out = tmp_path / "out.json"

    with pytest.raises(SystemExit):
        _run(monkeypatch, pool=pool, output=out, seed=42, size=10)


def test_prompt_ids_are_sorted(monkeypatch, tmp_path):
    """The script sorts the drawn IDs so the file is order-stable across
    Python versions / hash seeds. Pin that.
    """
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=20)

    data = json.loads(out.read_text())
    assert data["prompt_ids"] == sorted(data["prompt_ids"])


def test_creates_parent_dirs(monkeypatch, tmp_path):
    """Output dir is created on demand (launchers don't pre-mkdir)."""
    pool = tmp_path / "hard.jsonl"
    _write_pool(pool, 50)
    out = tmp_path / "nested" / "subdir" / "out.json"

    _run(monkeypatch, pool=pool, output=out, seed=42, size=10)

    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["prompt_ids"]) == 10
