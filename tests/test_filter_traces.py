"""Tests for parse_json_response in scripts/filter_traces.py — pure Python, no GPU."""

import sys
from pathlib import Path

import pytest

# mock heavy deps before import
from unittest.mock import MagicMock
for mod in ["torch", "transformers", "tqdm", "peft"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.filter_traces import parse_json_response


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------

def test_parse_plain_json():
    text = '{"criteria_met": true, "explanation": "good"}'
    result = parse_json_response(text)
    assert result["criteria_met"] is True
    assert result["explanation"] == "good"


def test_parse_json_with_code_fence():
    text = '```json\n{"criteria_met": false, "explanation": "bad"}\n```'
    result = parse_json_response(text)
    assert result["criteria_met"] is False


def test_parse_json_with_plain_code_fence():
    text = '```\n{"criteria_met": true, "explanation": "ok"}\n```'
    result = parse_json_response(text)
    assert result["criteria_met"] is True


def test_parse_json_strips_whitespace():
    text = '   \n{"criteria_met": true, "explanation": "padded"}\n   '
    result = parse_json_response(text)
    assert result["criteria_met"] is True


def test_parse_json_returns_falsy_on_invalid():
    # function returns None or {} when no valid JSON is found — both are falsy
    result = parse_json_response("this is not json at all")
    assert not result


def test_parse_json_returns_falsy_on_empty():
    result = parse_json_response("")
    assert not result


def test_parse_json_handles_nested_object():
    text = '{"criteria_met": true, "explanation": "ok", "details": {"score": 0.9}}'
    result = parse_json_response(text)
    assert result["details"]["score"] == 0.9
