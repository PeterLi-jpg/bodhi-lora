"""Tests for scripts/rubric_diff.py — pure Python, no heavy deps required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rubric_diff import axis_tags, index_results, compare


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_criteria_result(criterion, criteria_met, tags=None):
    return {
        "criterion": criterion,
        "criteria_met": criteria_met,
        "tags": tags or [],
    }


def _make_result(prompt_id, criteria_list, response="some response"):
    return {
        "prompt_id": prompt_id,
        "response": response,
        "criteria_results": criteria_list,
    }


def _make_payload(results):
    return {"results": results}


# ---------------------------------------------------------------------------
# axis_tags
# ---------------------------------------------------------------------------

def test_axis_tags_returns_axes():
    item = {"tags": ["axis:safety", "axis:accuracy", "theme:oncology"]}
    assert axis_tags(item) == ["safety", "accuracy"]


def test_axis_tags_ignores_non_axis_tags():
    item = {"tags": ["theme:cardiology", "tier:hard"]}
    assert axis_tags(item) == ["unlabeled"]


def test_axis_tags_empty_tags():
    item = {"tags": []}
    assert axis_tags(item) == ["unlabeled"]


def test_axis_tags_missing_tags_key():
    item = {}
    assert axis_tags(item) == ["unlabeled"]


# ---------------------------------------------------------------------------
# index_results
# ---------------------------------------------------------------------------

def test_index_results_basic():
    payload = _make_payload([
        _make_result("p1", [
            _make_criteria_result("is_safe", True),
            _make_criteria_result("is_accurate", False),
        ]),
    ])
    indexed = index_results(payload)
    assert "p1" in indexed
    assert indexed["p1"]["criteria"]["is_safe"]["criteria_met"] is True
    assert indexed["p1"]["criteria"]["is_accurate"]["criteria_met"] is False


def test_index_results_empty_payload():
    assert index_results({"results": []}) == {}


def test_index_results_missing_results_key():
    assert index_results({}) == {}


def test_index_results_preserves_response():
    payload = _make_payload([
        _make_result("p1", [], response="the model said this"),
    ])
    assert index_results(payload)["p1"]["response"] == "the model said this"


# ---------------------------------------------------------------------------
# compare — basic flips
# ---------------------------------------------------------------------------

def test_compare_detects_improvement():
    base = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", False)])
    ])
    candidate = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", True)])
    ])
    result = compare(base, candidate)
    assert result["improved"]["is_safe"] == 1
    assert "is_safe" not in result["regressed"]


def test_compare_detects_regression():
    base = _make_payload([
        _make_result("p1", [_make_criteria_result("is_accurate", True)])
    ])
    candidate = _make_payload([
        _make_result("p1", [_make_criteria_result("is_accurate", False)])
    ])
    result = compare(base, candidate)
    assert result["regressed"]["is_accurate"] == 1
    assert "is_accurate" not in result["improved"]


def test_compare_ignores_unchanged_criteria():
    base = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", True)])
    ])
    candidate = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", True)])
    ])
    result = compare(base, candidate)
    assert len(result["improved"]) == 0
    assert len(result["regressed"]) == 0


def test_compare_skips_missing_prompts():
    base = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", True)]),
        _make_result("p2", [_make_criteria_result("is_safe", False)]),
    ])
    candidate = _make_payload([
        # p2 missing from candidate
        _make_result("p1", [_make_criteria_result("is_safe", True)]),
    ])
    result = compare(base, candidate)
    # p2 absent in candidate — should not show up as a flip
    assert len(result["improved"]) == 0
    assert len(result["regressed"]) == 0


def test_compare_skips_missing_criterion_in_candidate():
    base = _make_payload([
        _make_result("p1", [
            _make_criteria_result("is_safe", False),
            _make_criteria_result("is_verbose", True),
        ])
    ])
    candidate = _make_payload([
        _make_result("p1", [
            _make_criteria_result("is_safe", True),
            # is_verbose missing from candidate
        ])
    ])
    result = compare(base, candidate)
    assert result["improved"]["is_safe"] == 1
    assert "is_verbose" not in result["regressed"]


# ---------------------------------------------------------------------------
# compare — axis summary
# ---------------------------------------------------------------------------

def test_compare_axis_summary():
    base = _make_payload([
        _make_result("p1", [
            _make_criteria_result("is_safe", False, tags=["axis:safety"]),
        ])
    ])
    candidate = _make_payload([
        _make_result("p1", [
            _make_criteria_result("is_safe", True, tags=["axis:safety"]),
        ])
    ])
    result = compare(base, candidate)
    assert result["axis_summary"]["safety"]["improved"] == 1
    assert result["axis_summary"]["safety"]["regressed"] == 0


def test_compare_axis_summary_unlabeled_fallback():
    base = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", False)])
    ])
    candidate = _make_payload([
        _make_result("p1", [_make_criteria_result("is_safe", True)])
    ])
    result = compare(base, candidate)
    assert "unlabeled" in result["axis_summary"]


# ---------------------------------------------------------------------------
# compare — example collection (capped at 5)
# ---------------------------------------------------------------------------

def test_compare_examples_capped_at_five():
    prompts = [f"p{i}" for i in range(10)]
    base = _make_payload([
        _make_result(pid, [_make_criteria_result("is_safe", False)]) for pid in prompts
    ])
    candidate = _make_payload([
        _make_result(pid, [_make_criteria_result("is_safe", True)]) for pid in prompts
    ])
    result = compare(base, candidate)
    assert len(result["examples"]["is_safe"]["improved"]) == 5


def test_compare_empty_payloads():
    result = compare(_make_payload([]), _make_payload([]))
    assert len(result["improved"]) == 0
    assert len(result["regressed"]) == 0
    assert result["axis_summary"] == {}
