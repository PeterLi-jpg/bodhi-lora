"""Dedicated tests for scripts/eval_ushape.py (audit: zero-coverage critical script).

eval_ushape.py is the post-hoc stratification step that turns per-example
healthbench scores into the U-shape / tier / theme tables that go straight
into the paper. A regression here silently rewrites headline numbers, so
every helper that aggregates scores needs at least one synthetic-data
sanity check.

A small handful of cutoff and summarize cases live in test_pure_logic.py
already; this file fills in the per-tier and per-theme aggregators, the
overall summarizer, the bootstrap CI plumbing, the empty-input edge case,
and the argparse contract.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── compute_tertile_cutoffs: tertile contract on a 9-point dataset ─────────

def test_compute_tertile_cutoffs_nine_known_scores():
    """With 9 evenly-spaced pos_points the q1/q2 cutoffs land where
    statistics.quantiles(n=3) puts them. Pinning these means a future
    refactor (e.g. to numpy.percentile) can't silently shift tier
    boundaries by an off-by-one."""
    from scripts.eval_ushape import compute_tertile_cutoffs

    meta = {f"p{i}": {"pos_points": float(i)} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    expected = statistics.quantiles([float(i) for i in range(1, 10)], n=3)
    assert q1 == pytest.approx(expected[0])
    assert q2 == pytest.approx(expected[1])
    assert q1 < q2


def test_compute_tertile_cutoffs_raises_on_too_few_examples():
    from scripts.eval_ushape import compute_tertile_cutoffs

    with pytest.raises(ValueError, match="at least 3"):
        compute_tertile_cutoffs({"only": {"pos_points": 1}})


# ── aggregate_by_tier: synthetic results with known tier means ─────────────

def test_aggregate_by_tier_means_match_synthetic_assignment():
    """Build a synthetic eval where we know which prompt is in which tier,
    then confirm aggregate_by_tier returns the expected tier means."""
    from scripts.eval_ushape import aggregate_by_tier, compute_tertile_cutoffs

    # 9 prompts: pos_points 1..9 give clean easy/medium/hard splits at q1=3.5, q2=6.5.
    meta = {f"p{i}": {"pos_points": float(i), "themes": []} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    # Score map: easy prompts (1..3) all 0.9, medium (4..6) all 0.5, hard (7..9) all 0.1.
    # Hard tier therefore must come out at fail_rate=1.0 (all below 0.4).
    results = []
    for i in range(1, 10):
        if i <= 3:
            score = 0.9
        elif i <= 6:
            score = 0.5
        else:
            score = 0.1
        results.append({"prompt_id": f"p{i}", "score": score})

    out = aggregate_by_tier(results, meta, q1, q2, fail_threshold=0.4)

    assert out["easy"]["mean"] == pytest.approx(0.9)
    assert out["medium"]["mean"] == pytest.approx(0.5)
    assert out["hard"]["mean"] == pytest.approx(0.1)
    assert out["easy"]["fail_rate"] == 0.0
    assert out["medium"]["fail_rate"] == 0.0
    assert out["hard"]["fail_rate"] == 1.0
    assert out["_missing_prompt_ids"] == 0


def test_aggregate_by_tier_counts_missing_prompt_ids():
    """Results referencing prompts that aren't in the metadata are counted
    in _missing_prompt_ids rather than silently dropped."""
    from scripts.eval_ushape import aggregate_by_tier, compute_tertile_cutoffs

    meta = {f"p{i}": {"pos_points": float(i), "themes": []} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    results = [
        {"prompt_id": "p1", "score": 0.9},
        {"prompt_id": "ghost_a", "score": 0.0},
        {"prompt_id": "ghost_b", "score": 0.0},
    ]
    out = aggregate_by_tier(results, meta, q1, q2, fail_threshold=0.4)

    assert out["_missing_prompt_ids"] == 2
    # easy still has the one valid score.
    assert out["easy"]["n"] == 1


def test_aggregate_by_tier_fills_missing_tiers_with_n_zero():
    """Even when no eval rows hit a tier (e.g. holdout only contains easy
    prompts) we still emit all three tier entries with n=0 so downstream
    plotting never KeyErrors."""
    from scripts.eval_ushape import aggregate_by_tier, compute_tertile_cutoffs

    meta = {f"p{i}": {"pos_points": float(i), "themes": []} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    # Only easy-tier prompts in the eval.
    results = [{"prompt_id": "p1", "score": 0.9}, {"prompt_id": "p2", "score": 0.8}]
    out = aggregate_by_tier(results, meta, q1, q2, fail_threshold=0.4)

    assert out["easy"]["n"] == 2
    assert out["medium"] == {"n": 0}
    assert out["hard"] == {"n": 0}


# ── aggregate_by_theme: confirm theme means match synthetic assignment ─────

def test_aggregate_by_theme_means_match_synthetic_assignment():
    from scripts.eval_ushape import aggregate_by_theme

    meta = {
        "p1": {"pos_points": 1.0, "themes": ["emergency_referrals"]},
        "p2": {"pos_points": 2.0, "themes": ["emergency_referrals", "hedging"]},
        "p3": {"pos_points": 3.0, "themes": ["hedging"]},
    }
    results = [
        {"prompt_id": "p1", "score": 0.2},
        {"prompt_id": "p2", "score": 0.8},
        {"prompt_id": "p3", "score": 0.6},
    ]
    out = aggregate_by_theme(results, meta, fail_threshold=0.4)

    # emergency_referrals = mean(0.2, 0.8) = 0.5
    assert out["emergency_referrals"]["mean"] == pytest.approx(0.5)
    # hedging = mean(0.8, 0.6) = 0.7
    assert out["hedging"]["mean"] == pytest.approx(0.7)
    # fail_rate for emergency_referrals: only 0.2 < 0.4 -> 0.5.
    assert out["emergency_referrals"]["fail_rate"] == pytest.approx(0.5)


# ── summarize_overall: top-level non-stratified stats ──────────────────────

def test_summarize_overall_collapses_results_to_score_list():
    from scripts.eval_ushape import summarize_overall

    results = [
        {"prompt_id": "p1", "score": 0.1},
        {"prompt_id": "p2", "score": 0.5},
        {"prompt_id": "p3", "score": 0.9},
    ]
    overall = summarize_overall(results, fail_threshold=0.4)

    assert overall["n"] == 3
    assert overall["mean"] == pytest.approx(0.5)
    assert overall["fail_rate"] == pytest.approx(1 / 3)


def test_summarize_overall_empty_results_returns_n_zero():
    """Edge case the aggregator hits when an eval JSON is empty
    (e.g. a smoke run that errored before a single example completed).
    Must return {n: 0}, not crash."""
    from scripts.eval_ushape import summarize_overall

    assert summarize_overall([], fail_threshold=0.4) == {"n": 0}


def test_aggregate_by_tier_empty_results_does_not_crash():
    """Empty eval list -> all tiers empty, no missing prompts, no exception."""
    from scripts.eval_ushape import aggregate_by_tier, compute_tertile_cutoffs

    meta = {f"p{i}": {"pos_points": float(i), "themes": []} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    out = aggregate_by_tier([], meta, q1, q2, fail_threshold=0.4)

    assert out["easy"] == {"n": 0}
    assert out["medium"] == {"n": 0}
    assert out["hard"] == {"n": 0}
    assert out["_missing_prompt_ids"] == 0


def test_aggregate_by_theme_empty_results_returns_empty_dict():
    """No eval rows -> no themes to aggregate. Must not crash."""
    from scripts.eval_ushape import aggregate_by_theme

    meta = {"p1": {"pos_points": 1.0, "themes": ["emergency_referrals"]}}
    out = aggregate_by_theme([], meta, fail_threshold=0.4)
    assert out == {}


# ── bootstrap plumbing: per-tier CIs round-trip through the aggregator ─────

def test_aggregate_by_tier_attaches_bootstrap_cis():
    """When bootstrap > 0, every non-empty tier must carry mean_ci and
    fail_rate_ci. Pinning this catches a refactor that forgets to forward
    rng/bootstrap to the inner summarize() call."""
    from scripts.eval_ushape import aggregate_by_tier, compute_tertile_cutoffs

    meta = {f"p{i}": {"pos_points": float(i), "themes": []} for i in range(1, 10)}
    q1, q2 = compute_tertile_cutoffs(meta)

    results = [{"prompt_id": f"p{i}", "score": 0.5} for i in range(1, 10)]
    rng = np.random.default_rng(7)

    out = aggregate_by_tier(
        results, meta, q1, q2, fail_threshold=0.4,
        bootstrap=100, rng=rng,
    )
    for tier in ("easy", "medium", "hard"):
        assert "mean_ci" in out[tier]
        assert "fail_rate_ci" in out[tier]
        lo, hi = out[tier]["mean_ci"]
        assert lo <= out[tier]["mean"] <= hi


# ── argparse contract: pin the default of --tertile-on-holdout-only ────────
#
# This flag controls whether tier cutoffs are computed on the holdout subset
# or on full HealthBench. When the default flips, every multi-seed table that
# was published before silently means something different — so the default
# is part of the methodology contract and we pin it.

def test_tertile_on_holdout_only_default():
    """Pin the current default of the --tertile-on-holdout-only flag.

    The script defines this with action='store_true', so the default is
    False. PR #150 (#N3 fix) discusses flipping this to True; this test
    documents what the flag is set to today and will need to be flipped
    in lockstep when that change lands. Either way the test is the
    single tripwire for an unintentional change.
    """
    import argparse
    import importlib

    # Re-import to start with a clean argparse state.
    sys.modules.pop("scripts.eval_ushape", None)
    eu = importlib.import_module("scripts.eval_ushape")

    # Rebuild the parser by replicating the flag definition. We can't call
    # main() with no args because it has required arguments, but the
    # argparse contract for store_true is well-defined.
    parser = argparse.ArgumentParser()
    parser.add_argument("--tertile-on-holdout-only", action="store_true")
    args = parser.parse_args([])
    flag_default = args.tertile_on_holdout_only

    # The module-level definition must agree with the local replica:
    # both should be store_true (default False).
    src = Path(eu.__file__).read_text()
    assert "--tertile-on-holdout-only" in src
    assert "action=\"store_true\"" in src.split("--tertile-on-holdout-only", 1)[1].split(")", 1)[0]
    assert flag_default is False


# ── end-to-end: script writes the JSON we expect (uses synthetic files) ────
#
# This is the integration spine: we feed the script the same JSON shapes
# eval_healthbench.py produces and confirm the output document has the
# fields plotting code reads. No real HealthBench files used.

def test_main_end_to_end_with_synthetic_files(monkeypatch, tmp_path):
    import importlib

    # Tiny HealthBench Hard fixture: 6 examples is enough for cutoffs
    # (statistics.quantiles requires >=3 samples; we use 6 for stable splits).
    hb_path = tmp_path / "hb.jsonl"
    rows = []
    for i in range(1, 7):
        rows.append({
            "prompt_id": f"hb_{i}",
            # Mimic the real schema: example_tags carry theme:..., rubrics
            # carry per-criterion points (positive sum is what tertiles use).
            "example_tags": [f"theme:t{i % 2}"],
            "rubrics": [{"points": float(i)}, {"points": -1.0}],
        })
    hb_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Two eval files, both in eval_healthbench.py output shape.
    eval_a = tmp_path / "eval_a.json"
    eval_a.write_text(json.dumps({
        "config": "baseline",
        "results": [{"prompt_id": f"hb_{i}", "score": 0.5} for i in range(1, 7)],
    }))
    eval_b = tmp_path / "eval_b.json"
    eval_b.write_text(json.dumps({
        "config": "lora",
        "results": [{"prompt_id": f"hb_{i}", "score": 0.8} for i in range(1, 7)],
    }))

    out_path = tmp_path / "summary.json"

    monkeypatch.setattr(sys, "argv", [
        "eval_ushape.py",
        "--eval-jsons", str(eval_a), str(eval_b),
        "--healthbench", str(hb_path),
        "--output", str(out_path),
        "--fail-threshold", "0.4",
    ])

    sys.modules.pop("scripts.eval_ushape", None)
    eu = importlib.import_module("scripts.eval_ushape")
    eu.main()

    written = json.loads(out_path.read_text())
    assert "thresholds" in written
    assert set(written["configs"].keys()) == {"baseline", "lora"}
    base = written["configs"]["baseline"]
    assert base["n_examples"] == 6
    assert base["overall"]["mean"] == pytest.approx(0.5)
    assert "by_tier" in base and "by_theme" in base
    # The fail_threshold rate at score=0.5 for baseline is 0 (none below 0.4).
    assert base["overall"]["fail_rate"] == 0.0
