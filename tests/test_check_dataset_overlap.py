"""Tests for check_dataset_overlap.py — pure Python, no GPU required."""

import io
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_dataset_overlap import (
    SAFETY_THEMES,
    _tag_counts_from_jsonl,
    load_eval_ids,
    load_eval_tag_scores,
    per_theme_score_report,
    stratify_eval_holdout,
    tag_overlap_report,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj))


def _capture(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), capture all print() output, return as string."""
    buf = io.StringIO()
    with patch("builtins.print", lambda *a, **k: buf.write(" ".join(str(x) for x in a) + "\n")):
        fn(*args, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# load_eval_ids
# ---------------------------------------------------------------------------

def test_load_eval_ids_list_schema(tmp_path):
    p = tmp_path / "ids.json"
    _write_json(p, ["id1", "id2", "id3"])
    assert load_eval_ids(str(p)) == {"id1", "id2", "id3"}


def test_load_eval_ids_dict_schema(tmp_path):
    p = tmp_path / "ids.json"
    _write_json(p, {"prompt_ids": ["a", "b"], "total_samples": 2})
    assert load_eval_ids(str(p)) == {"a", "b"}


# ---------------------------------------------------------------------------
# _tag_counts_from_jsonl
# ---------------------------------------------------------------------------

def test_tag_counts_correct(tmp_path):
    p = tmp_path / "train.jsonl"
    _write_jsonl(str(p), [
        {"tags": ["theme:hedging", "axis:safety"]},
        {"tags": ["theme:hedging"]},
        {"tags": ["theme:medication_safety"]},
    ])
    counts, total = _tag_counts_from_jsonl(str(p), "tags")
    assert total == 3
    assert counts["theme:hedging"] == 2
    assert counts["theme:medication_safety"] == 1
    assert counts["axis:safety"] == 1


def test_tag_counts_warns_when_many_missing(tmp_path):
    p = tmp_path / "train.jsonl"
    # 9 records missing the field, 1 present → 90% missing → should warn
    _write_jsonl(str(p), [{"no_tags": True}] * 9 + [{"tags": ["t1"]}])
    out = _capture(_tag_counts_from_jsonl, str(p), "tags")
    assert "WARNING" in out


def test_tag_counts_no_warning_below_threshold(tmp_path):
    p = tmp_path / "train.jsonl"
    _write_jsonl(str(p), [{"tags": ["t1"]}] * 10)
    out = _capture(_tag_counts_from_jsonl, str(p), "tags")
    assert "WARNING" not in out


# ---------------------------------------------------------------------------
# load_eval_tag_scores
# ---------------------------------------------------------------------------

def test_load_eval_tag_scores_aggregates(tmp_path):
    p = tmp_path / "eval.json"
    _write_json(str(p), {
        "results": [
            {"prompt_id": "p1", "score": 0.8,
             "tag_scores": {"theme:hedging": 0.9, "axis:safety": 0.7}},
            {"prompt_id": "p2", "score": 0.6,
             "tag_scores": {"theme:hedging": 0.5}},
        ]
    })
    ts = load_eval_tag_scores(str(p))
    assert pytest.approx(sorted(ts["theme:hedging"])) == [0.5, 0.9]
    assert len(ts["axis:safety"]) == 1
    assert pytest.approx(ts["axis:safety"][0]) == 0.7


def test_load_eval_tag_scores_skips_nan(tmp_path):
    p = tmp_path / "eval.json"
    _write_json(str(p), {
        "results": [
            {"prompt_id": "p1", "score": 0.5,
             "tag_scores": {"theme:hedging": float("nan"), "t2": 0.6}},
        ]
    })
    ts = load_eval_tag_scores(str(p))
    assert "theme:hedging" not in ts
    assert "t2" in ts


def test_load_eval_tag_scores_empty_results(tmp_path):
    p = tmp_path / "eval.json"
    _write_json(str(p), {"results": []})
    assert load_eval_tag_scores(str(p)) == {}


# ---------------------------------------------------------------------------
# tag_overlap_report
# ---------------------------------------------------------------------------

def test_tag_overlap_no_skew(tmp_path):
    train_p = tmp_path / "train.jsonl"
    hard_p = tmp_path / "hard.jsonl"
    _write_jsonl(str(train_p), [{"tags": ["theme:hedging"]}] * 10)
    _write_jsonl(str(hard_p), [
        {"prompt_id": f"p{i}", "example_tags": ["theme:hedging"]} for i in range(5)
    ])
    eval_ids = {f"p{i}" for i in range(5)}

    out = _capture(tag_overlap_report, str(train_p), str(hard_p), eval_ids)
    assert "tag overlap OK" in out
    assert "WARN" not in out


def test_tag_overlap_skew_detected(tmp_path):
    # theme:hedging is 100% of train but only 10% of eval
    train_p = tmp_path / "train.jsonl"
    hard_p = tmp_path / "hard.jsonl"
    _write_jsonl(str(train_p), [{"tags": ["theme:hedging"]}] * 100)
    hard_records = (
        [{"prompt_id": "e0", "example_tags": ["theme:hedging"]}]
        + [{"prompt_id": f"e{i}", "example_tags": ["theme:other"]} for i in range(1, 10)]
    )
    _write_jsonl(str(hard_p), hard_records)
    eval_ids = {f"e{i}" for i in range(10)}

    out = _capture(tag_overlap_report, str(train_p), str(hard_p), eval_ids)
    assert "WARN" in out
    assert "theme:hedging" in out


def test_tag_overlap_safety_theme_marked(tmp_path):
    train_p = tmp_path / "train.jsonl"
    hard_p = tmp_path / "hard.jsonl"
    _write_jsonl(str(train_p), [{"tags": ["theme:emergency_referrals"]}] * 50)
    hard_records = (
        [{"prompt_id": "e0", "example_tags": ["theme:emergency_referrals"]}]
        + [{"prompt_id": f"e{i}", "example_tags": ["theme:other"]} for i in range(1, 10)]
    )
    _write_jsonl(str(hard_p), hard_records)
    eval_ids = {f"e{i}" for i in range(10)}

    out = _capture(tag_overlap_report, str(train_p), str(hard_p), eval_ids,
                   safety_themes=SAFETY_THEMES)
    assert "theme:emergency_referrals" in out
    assert "[SAFETY]" in out


def test_tag_overlap_missing_train_raises(tmp_path):
    hard_p = tmp_path / "hard.jsonl"
    _write_jsonl(str(hard_p), [])
    with pytest.raises(SystemExit):
        tag_overlap_report(str(tmp_path / "nope.jsonl"), str(hard_p), set())


def test_tag_overlap_missing_hard_raises(tmp_path):
    train_p = tmp_path / "train.jsonl"
    _write_jsonl(str(train_p), [{"tags": ["t1"]}])
    with pytest.raises(SystemExit):
        tag_overlap_report(str(train_p), str(tmp_path / "nope.jsonl"), {"e0"})


def test_tag_overlap_with_eval_results_shows_per_theme_scores(tmp_path):
    train_p = tmp_path / "train.jsonl"
    hard_p = tmp_path / "hard.jsonl"
    eval_p = tmp_path / "eval.json"

    _write_jsonl(str(train_p), [{"tags": ["theme:hedging"]}] * 100)
    _write_jsonl(str(hard_p), [
        {"prompt_id": "e0", "example_tags": ["theme:hedging"]},
        *[{"prompt_id": f"e{i}", "example_tags": ["theme:other"]} for i in range(1, 10)],
    ])
    _write_json(str(eval_p), {
        "results": [
            {"prompt_id": "e0", "score": 0.9,
             "tag_scores": {"theme:hedging": 0.95}},
            *[{"prompt_id": f"e{i}", "score": 0.5,
               "tag_scores": {"theme:other": 0.5}} for i in range(1, 10)],
        ]
    })
    eval_ids = {f"e{i}" for i in range(10)}

    out = _capture(tag_overlap_report, str(train_p), str(hard_p), eval_ids,
                   eval_results_path=str(eval_p))
    assert "PER-THEME SCORES" in out
    assert "MEMORIZATION RISK" in out


# ---------------------------------------------------------------------------
# per_theme_score_report
# ---------------------------------------------------------------------------

def test_per_theme_score_memorization_risk(tmp_path):
    tag_score_map = {"theme:hedging": [0.9, 0.85, 0.95]}
    out = _capture(
        per_theme_score_report,
        tag_score_map,
        ["theme:hedging"],        # flagged
        SAFETY_THEMES,
        tag_ratios={"theme:hedging": 4.0},  # 4x over-represented
    )
    assert "MEMORIZATION RISK" in out
    assert "theme:hedging" in out
    assert "[SAFETY]" in out


def test_per_theme_score_no_risk_when_low_score():
    # Flagged tag but low eval score → no memorization risk
    tag_score_map = {"theme:hedging": [0.2, 0.3, 0.1]}
    out = _capture(
        per_theme_score_report,
        tag_score_map,
        ["theme:hedging"],
        SAFETY_THEMES,
        tag_ratios={"theme:hedging": 5.0},
    )
    assert "MEMORIZATION RISK" not in out


def test_per_theme_score_empty_map():
    out = _capture(per_theme_score_report, {}, [], SAFETY_THEMES)
    assert "no tag scores found" in out


def test_per_theme_score_no_risk_column_when_no_ratios():
    tag_score_map = {"theme:other": [0.6, 0.7]}
    out = _capture(per_theme_score_report, tag_score_map, [], SAFETY_THEMES)
    assert "risk" not in out.lower() or "risk" not in out.split("\n")[1]


def test_per_theme_score_inf_ratio(tmp_path):
    # Tag in train only (inf ratio) with high eval score → MEMORIZATION RISK
    tag_score_map = {"theme:medication_safety": [0.9, 0.8]}
    out = _capture(
        per_theme_score_report,
        tag_score_map,
        ["theme:medication_safety"],
        SAFETY_THEMES,
        tag_ratios={"theme:medication_safety": float("inf")},
    )
    assert "MEMORIZATION RISK" in out
    assert "inf" in out


# ---------------------------------------------------------------------------
# stratify_eval_holdout
# ---------------------------------------------------------------------------

def _make_hard(tmp_path, theme_counts):
    """Write a Hard JSONL with the given {theme: count} distribution."""
    records = []
    idx = 0
    for theme, count in theme_counts.items():
        for _ in range(count):
            records.append({"prompt_id": f"p{idx}", "example_tags": [theme]})
            idx += 1
    path = str(tmp_path / "hard.jsonl")
    _write_jsonl(path, records)
    return path


def test_stratify_exact_n(tmp_path):
    path = _make_hard(tmp_path, {
        "theme:hedging": 100,
        "theme:medication_safety": 100,
        "theme:other": 100,
    })
    result = stratify_eval_holdout(path, n=60, seed=42)
    assert len(result) == 60


def test_stratify_proportional(tmp_path):
    # 300 hedging : 100 medication_safety → 3:1 → expect 60:20 for n=80
    path = _make_hard(tmp_path, {
        "theme:hedging": 300,
        "theme:medication_safety": 100,
    })
    result = stratify_eval_holdout(path, n=80, seed=42)
    assert len(result) == 80

    theme_counts = {"theme:hedging": 0, "theme:medication_safety": 0}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["prompt_id"] in result:
                t = obj["example_tags"][0]
                theme_counts[t] = theme_counts.get(t, 0) + 1

    # LRM gives exactly 60 and 20 for these clean proportions
    assert theme_counts["theme:hedging"] == 60
    assert theme_counts["theme:medication_safety"] == 20


def test_stratify_excludes_ids(tmp_path):
    path = _make_hard(tmp_path, {"theme:hedging": 100})
    exclude = {f"p{i}" for i in range(50)}
    result = stratify_eval_holdout(path, n=10, seed=42, exclude_ids=exclude)
    assert len(result) == 10
    assert len(result & exclude) == 0


def test_stratify_insufficient_raises(tmp_path):
    path = _make_hard(tmp_path, {"theme:hedging": 5})
    with pytest.raises(SystemExit):
        stratify_eval_holdout(path, n=200, seed=42)


def test_stratify_untagged_bucket(tmp_path):
    records = [{"prompt_id": f"p{i}", "example_tags": []} for i in range(50)]
    path = str(tmp_path / "hard.jsonl")
    _write_jsonl(path, records)
    result = stratify_eval_holdout(path, n=10, seed=42)
    assert len(result) == 10


def test_stratify_deterministic(tmp_path):
    path = _make_hard(tmp_path, {
        "theme:hedging": 100,
        "theme:other": 100,
    })
    assert stratify_eval_holdout(path, n=20, seed=7) == stratify_eval_holdout(path, n=20, seed=7)


def test_stratify_different_seeds_differ(tmp_path):
    path = _make_hard(tmp_path, {"theme:hedging": 200})
    r1 = stratify_eval_holdout(path, n=10, seed=1)
    r2 = stratify_eval_holdout(path, n=10, seed=99)
    assert r1 != r2


def test_stratify_largest_remainder_sums_exactly(tmp_path):
    # Three themes with non-integer proportional shares → LRM must sum to n
    path = _make_hard(tmp_path, {
        "theme:hedging": 33,
        "theme:medication_safety": 33,
        "theme:other": 34,
    })
    for n in (10, 17, 33, 50, 99):
        result = stratify_eval_holdout(path, n=n, seed=0)
        assert len(result) == n
