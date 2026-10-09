"""Extract reads each eval file by the score formula the file says it stored, and puts
the normalized score in the table for every file."""
import pytest

pytest.importorskip("pandas")

from bodhi_lora.analysis.extract import STORED_SCORE, raw_rubric_score, rubric_score  # noqa: E402

CRITERIA = [{"points": 8, "criteria_met": True}, {"points": 4, "criteria_met": False},
            {"points": -6, "criteria_met": True}]


def test_stored_score_formulas():
    # raw: earned over positive points; normalized: (earned - neg) / (pos - neg).
    assert raw_rubric_score(CRITERIA) == pytest.approx((8 - 6) / 12)
    assert rubric_score(CRITERIA) == pytest.approx((8 - 6 + 6) / (12 + 6))
    assert STORED_SCORE["normalized_score"](CRITERIA) == rubric_score(CRITERIA)
    # A file without a score_metric field stored the raw score.
    assert STORED_SCORE[None](CRITERIA) == raw_rubric_score(CRITERIA)


def test_raw_score_can_be_negative_and_has_no_positive_points_case():
    assert raw_rubric_score([{"points": 2, "criteria_met": False}, {"points": -5, "criteria_met": True}]) < 0
    assert raw_rubric_score([{"points": -5, "criteria_met": True}]) == 0.0
