"""The seven-dimension grader's parsing and aggregation (bodhi_lora/grading/epistemic.py)."""

import hashlib
import json

from bodhi_lora.grading import epistemic as ep

# The evaluator template of every recorded cell, 02ac9a3:scripts/eval_epistemic.py.
FROZEN_TEMPLATE_SHA256 = "aa6e2f507de69fd5cad113ebb1f94cd605f129e62f4b3ab727bc73c7e440d313"


def test_the_grader_template_is_the_one_that_graded_the_recorded_cells():
    """Every reported dimension comes from this text, so it does not change in place. A new
    template, for example one that does not name the protocol under test, is a new grader
    version with its own hash (docs/ANALYSIS_PLAN.md section 12)."""
    assert hashlib.sha256(ep.EPISTEMIC_GRADER_TEMPLATE.encode("utf-8")).hexdigest() == FROZEN_TEMPLATE_SHA256
    assert ep.TEMPLATE_SHA256 == FROZEN_TEMPLATE_SHA256


def test_messages_to_text_handles_strings_and_lists():
    # Already a string — returned as-is.
    assert ep.messages_to_text("hi") == "hi"

    # Standard chat list.
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
    ]
    text = ep.messages_to_text(msgs)
    assert "[user] first" in text
    assert "[assistant] ok" in text

    # Multimodal-style content (list of {type, text}).
    msgs_mm = [{"role": "user", "content": [
        {"type": "text", "text": "part-a"},
        {"type": "text", "text": "part-b"},
    ]}]
    assert "part-a" in ep.messages_to_text(msgs_mm)
    assert "part-b" in ep.messages_to_text(msgs_mm)


def test_validate_grade_dict_required_fields():
    good = {
        "uncertainty_acknowledgment": {"score": 2, "reason": "x"},
        "context_seeking": {"score": 1, "n_questions": 0, "active_inquiry": False, "reason": "x"},
        "red_flag_identification": {"score": 0, "reason": "x"},
        "scope_bounding": {"score": 1, "reason": "x"},
        "specificity": {"score": 2, "has_concrete_numbers": True, "reason": "x"},
        "hedging": {"score": 1, "is_blanket_disclaimer": False, "appropriate": True, "reason": "x"},
    }
    assert ep._validate_grade_dict(good) is True

    # Missing the entire context_seeking block.
    bad_missing = dict(good)
    bad_missing.pop("context_seeking")
    assert ep._validate_grade_dict(bad_missing) is False

    # Present but missing a required nested field (n_questions).
    bad_partial = json.loads(json.dumps(good))
    bad_partial["context_seeking"].pop("n_questions")
    assert ep._validate_grade_dict(bad_partial) is False

    # Wrong type at a top-level key.
    bad_type = json.loads(json.dumps(good))
    bad_type["specificity"] = "strong"
    assert ep._validate_grade_dict(bad_type) is False


def test_flatten_grade_pulls_expected_fields():
    parsed = {
        "uncertainty_acknowledgment": {"score": 2, "reason": "x"},
        "context_seeking": {"score": 1, "n_questions": 3, "active_inquiry": True, "reason": "x"},
        "red_flag_identification": {"score": 2, "reason": "x"},
        "scope_bounding": {"score": 1, "reason": "x"},
        "specificity": {"score": 2, "has_concrete_numbers": True, "reason": "x"},
        "hedging": {"score": 0, "is_blanket_disclaimer": True, "appropriate": False, "reason": "x"},
    }
    flat = ep.flatten_grade(parsed)
    assert flat["uncertainty_acknowledgment"] == 2
    assert flat["n_questions"] == 3
    assert flat["active_inquiry"] is True
    assert flat["red_flag_identification"] == 2
    assert flat["specificity"] == 2
    assert flat["has_concrete_numbers"] is True
    assert flat["is_blanket_disclaimer"] is True
    assert flat["appropriate_hedging"] is False


def test_aggregate_handles_empty_and_parse_failures():
    assert ep.aggregate([]) == {"n": 0}

    # All parse-failed — no scored examples.
    only_failed = [{"prompt_id": "a", "response": "r", "parse_failure": True, "scores": None}]
    agg_failed = ep.aggregate(only_failed)
    assert agg_failed["n"] == 1
    assert agg_failed["n_scored"] == 0
    assert agg_failed["n_parse_failures"] == 1


def test_aggregate_computes_means_and_rates():
    def make(uncertainty, n_q, active, red, scope, has_num, blanket, appropriate):
        return {
            "prompt_id": "p", "response": "r", "parse_failure": False,
            "scores": {
                "uncertainty_acknowledgment": uncertainty,
                "context_seeking": 1,
                "n_questions": n_q,
                "active_inquiry": active,
                "red_flag_identification": red,
                "scope_bounding": scope,
                "specificity": 2 if has_num else 0,
                "has_concrete_numbers": has_num,
                "hedging": 1,
                "is_blanket_disclaimer": blanket,
                "appropriate_hedging": appropriate,
            },
        }

    graded = [
        make(2, 3, True,  2, 2, True,  False, True),
        make(1, 1, False, 0, 0, False, True,  False),
    ]
    agg = ep.aggregate(graded)

    assert agg["n"] == 2
    assert agg["n_scored"] == 2
    assert agg["n_parse_failures"] == 0
    assert agg["uncertainty_acknowledgment_mean"] == 1.5
    assert agg["questions_asked_mean"] == 2.0
    assert agg["active_inquiry_rate"] == 0.5
    # red_flag_rate is the share with red_flag_identification >= 2.
    assert agg["red_flag_rate"] == 0.5
    assert agg["specificity_rate"] == 0.5
    assert agg["blanket_disclaimer_rate"] == 0.5
    # scope_bounded_rate is the share with scope_bounding >= 2.
    assert agg["scope_bounded_rate"] == 0.5
    assert agg["appropriate_hedging_rate"] == 0.5
