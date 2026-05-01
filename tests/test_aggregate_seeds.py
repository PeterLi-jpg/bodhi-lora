"""Tests for scripts/aggregate_seeds.py.

These exercise the unit that combines per-seed eval JSONs into an
across-seed summary. The interesting bits (per the audit and PR #150's
contract) are:

  * std is None when n_seeds < 2 (not 0.0)
  * ci_low/ci_high present-but-None when n_seeds < 5; ci_method strings
    explain why
  * NaN/Inf are filtered out and counted as n_excluded_nan, never silently
    averaged in
  * malformed per-seed JSONs raise SystemExit with the file path

Most tests target ``_aggregate_metric_across_seeds`` directly because that
is where all of the policy lives. Two end-to-end tests run ``main()`` over
synthetic seed dirs to make sure the full pipeline (including the
malformed-JSON guard) wires up correctly.
"""

import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_aggregate_seeds():
    sys.modules.pop("scripts.aggregate_seeds", None)
    return importlib.import_module("scripts.aggregate_seeds")


# ── unit tests for _aggregate_metric_across_seeds ────────────────────────


def test_aggregates_clean_seeds():
    """3 seeds, no NaN: mean/std come out as expected; CI is skipped."""
    ag = _import_aggregate_seeds()
    values = [0.40, 0.50, 0.60]
    out = ag._aggregate_metric_across_seeds(values)

    assert out["n_seeds"] == 3
    # mean 0.5 by construction; std uses sample stdev.
    assert out["mean"] == pytest.approx(0.5)
    assert out["std"] == pytest.approx(statistics.stdev(values))
    assert out["min"] == pytest.approx(0.4)
    assert out["max"] == pytest.approx(0.6)
    assert out["values"] == [0.4, 0.5, 0.6]
    assert out["n_excluded_nan"] == 0
    # 3 < 5: CI is skipped but still reported as None.
    assert out["ci_low"] is None
    assert out["ci_high"] is None
    assert out["ci_method"] == "skipped (n<5)"
    assert "note" not in out


def test_excludes_nan_scores():
    """NaN/Inf are filtered before any stat; n_excluded_nan reports the count."""
    ag = _import_aggregate_seeds()
    values = [0.4, float("nan"), 0.6]
    out = ag._aggregate_metric_across_seeds(values)

    # NaN filtered out, only 2 finite seeds remain.
    assert out["n_seeds"] == 2
    assert out["n_excluded_nan"] == 1
    assert out["mean"] == pytest.approx(0.5)
    # std is the stdev of the two finite values, not over [0.4, NaN, 0.6].
    assert out["std"] == pytest.approx(statistics.stdev([0.4, 0.6]))
    # NaN should never leak into the values list.
    for v in out["values"]:
        assert math.isfinite(v)

    # Same shape with +Inf to make sure it is also filtered.
    out_inf = ag._aggregate_metric_across_seeds([0.4, float("inf"), 0.6])
    assert out_inf["n_excluded_nan"] == 1
    assert out_inf["n_seeds"] == 2


def test_handles_single_seed():
    """N=1 means stdev is undefined; aggregator must say so explicitly."""
    ag = _import_aggregate_seeds()
    out = ag._aggregate_metric_across_seeds([0.55])

    assert out["n_seeds"] == 1
    assert out["mean"] == pytest.approx(0.55)
    # The contract: std=None and a note that explains why, NOT 0.0.
    assert out["std"] is None
    assert out["note"] == "stdev undefined for n_seeds<2"
    # CI fields present-but-None.
    assert out["ci_low"] is None
    assert out["ci_high"] is None
    assert out["ci_method"] == "skipped (n<5)"


def test_handles_few_seeds_for_ci():
    """N=3 has well-defined std but CI must still be skipped (need n>=5)."""
    ag = _import_aggregate_seeds()
    out = ag._aggregate_metric_across_seeds([0.3, 0.5, 0.7])

    assert out["n_seeds"] == 3
    assert out["std"] is not None  # stdev IS defined for n>=2
    assert out["ci_low"] is None
    assert out["ci_high"] is None
    assert out["ci_method"] == "skipped (n<5)"


def test_aggregates_full_5_seeds_with_ci():
    """N=5 triggers percentile-95 CI; check the bounds match np.percentile."""
    ag = _import_aggregate_seeds()
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = ag._aggregate_metric_across_seeds(values)

    assert out["n_seeds"] == 5
    assert out["ci_method"] == "percentile-95 (n>=5)"
    # 95% CI = 2.5th / 97.5th percentiles.
    assert out["ci_low"] == pytest.approx(float(np.percentile(values, 2.5)))
    assert out["ci_high"] == pytest.approx(float(np.percentile(values, 97.5)))
    assert out["n_excluded_nan"] == 0


def test_handles_all_none_or_all_nan():
    """If every value is None/NaN we report n_seeds=0, not crash."""
    ag = _import_aggregate_seeds()

    out_none = ag._aggregate_metric_across_seeds([None, None])
    assert out_none == {"n_seeds": 0}

    out_nan = ag._aggregate_metric_across_seeds([float("nan"), float("nan")])
    # Both NaN: aggregator returns n_seeds=0 but still flags n_excluded_nan
    # so a downstream check can tell "no data" from "all data was corrupt."
    assert out_nan["n_seeds"] == 0
    assert out_nan["n_excluded_nan"] == 2


# ── end-to-end: malformed-JSON guard via _load_eval_json ────────────────


def test_load_eval_json_raises_systemexit_with_path(tmp_path):
    """Truncated/corrupt seed JSONs must surface the file path so the
    cluster log reader knows which seed is broken."""
    ag = _import_aggregate_seeds()
    bad = tmp_path / "broken.json"
    bad.write_text("{not valid json")

    with pytest.raises(SystemExit) as exc:
        ag._load_eval_json(bad)
    msg = str(exc.value)
    assert "aggregate_seeds: malformed JSON" in msg
    assert str(bad) in msg


# ── end-to-end via main(): full pipeline with synthetic seeds ────────────


def _write_healthbench_jsonl(path, n_prompts=6):
    """Minimal HealthBench fixture with a couple of theme tags + rubrics."""
    rows = []
    for i in range(n_prompts):
        rows.append({
            "prompt_id": f"p{i}",
            "example_tags": ["theme:hedging" if i % 2 == 0 else "theme:context_seeking"],
            "rubrics": [{"points": (i % 3) + 1}, {"points": 1}],
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_seed_dir(seed_dir, score_per_config):
    """Write the four CONFIG_NAMES JSONs into seed_dir.

    score_per_config maps config name -> per-prompt score (constant for
    simplicity). The eval JSON shape mirrors what eval_healthbench.py
    emits: {mean, results: [{prompt_id, score}, ...]}.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    config_names = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")
    for cfg in config_names:
        score = score_per_config[cfg]
        results = [{"prompt_id": f"p{i}", "score": score} for i in range(6)]
        payload = {"mean": score, "results": results}
        (seed_dir / f"{cfg}.json").write_text(json.dumps(payload))


def test_main_aggregates_across_three_seeds(tmp_path, monkeypatch, capsys):
    """End-to-end: 3 seeds with deterministic scores -> summary JSON has the
    right shape, std is reported (n>=2), CI is skipped (n<5)."""
    ag = _import_aggregate_seeds()

    hb_path = tmp_path / "healthbench.jsonl"
    _write_healthbench_jsonl(hb_path)

    # 3 seeds; same scores so mean=value, std=0 (well-defined since n>=2).
    seeds = {42: 0.4, 7: 0.5, 13: 0.6}
    eval_root = tmp_path / "eval"
    for seed, base_score in seeds.items():
        _write_seed_dir(
            eval_root / f"seed_{seed}",
            score_per_config={
                "base_no_wrapper": base_score,
                "base_bodhi": base_score + 0.05,
                "lora_no_wrapper": base_score + 0.10,
                "lora_bodhi": base_score + 0.15,
            },
        )

    out_path = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "aggregate_seeds.py",
        "--seed-dirs", str(eval_root / "seed_*"),
        "--healthbench", str(hb_path),
        "--output", str(out_path),
    ])

    ag.main()

    summary = json.loads(out_path.read_text())
    assert summary["n_seeds"] == 3
    assert sorted(summary["seeds"]) == [7, 13, 42]
    cfg = summary["configs"]["base_no_wrapper"]["overall_mean"]
    assert cfg["n_seeds"] == 3
    assert cfg["mean"] == pytest.approx(0.5)
    assert cfg["std"] is not None
    # n=3 < 5 -> CI skipped per the contract.
    assert cfg["ci_low"] is None
    assert cfg["ci_method"] == "skipped (n<5)"


def test_main_aborts_on_malformed_seed_json(tmp_path, monkeypatch):
    """If one seed's JSON is corrupt, main() must SystemExit with the path."""
    ag = _import_aggregate_seeds()

    hb_path = tmp_path / "healthbench.jsonl"
    _write_healthbench_jsonl(hb_path)

    eval_root = tmp_path / "eval"
    # First seed clean.
    _write_seed_dir(
        eval_root / "seed_1",
        {"base_no_wrapper": 0.3, "base_bodhi": 0.4,
         "lora_no_wrapper": 0.5, "lora_bodhi": 0.6},
    )
    # Second seed: every config file looks valid except base_no_wrapper.
    _write_seed_dir(
        eval_root / "seed_2",
        {"base_no_wrapper": 0.3, "base_bodhi": 0.4,
         "lora_no_wrapper": 0.5, "lora_bodhi": 0.6},
    )
    bad_path = eval_root / "seed_2" / "base_no_wrapper.json"
    bad_path.write_text("{this is not json")

    monkeypatch.setattr(sys, "argv", [
        "aggregate_seeds.py",
        "--seed-dirs", str(eval_root / "seed_*"),
        "--healthbench", str(hb_path),
        "--output", str(tmp_path / "summary.json"),
    ])

    with pytest.raises(SystemExit) as exc:
        ag.main()
    assert "malformed JSON" in str(exc.value)
    assert str(bad_path) in str(exc.value)
