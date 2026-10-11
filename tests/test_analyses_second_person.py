"""Tests for rebuttal/analyses/second_person_questions.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The script reads the 2026-05 cell's raw eval and grade JSON, which git ignores since
# 2026-10-11 (open decision O2), so a checkout without them skips the golden values.
RAW_PRESENT = any((ROOT / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05").glob("seed_*/eval/base.json"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "second_person_questions", ROOT / "rebuttal" / "analyses" / "second_person_questions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rule_on_examples():
    sp = _load()
    assert sp.has_second_person_question("How old are you?")
    assert sp.has_second_person_question("**Are you able to see a doctor this week?**")
    assert sp.has_second_person_question("Rest well.\nDo you have a fever?")
    # A question with no second person, or second person with no question, does not count.
    assert not sp.has_second_person_question("Is this an emergency? It depends.")
    assert not sp.has_second_person_question("You should rest and drink fluids.")
    # "you" inside another word is not a pronoun.
    assert not sp.has_second_person_question("Is the youth league safe?")
    assert not sp.has_second_person_question("")


@pytest.mark.skipif(not RAW_PRESENT, reason="the 2026-05 raw eval files are not in this checkout")
def test_golden_values_on_committed_results(tmp_path, monkeypatch):
    """Pins what the stated rule gives. The posted 18.4% / 39.6% (PROV-7) do not
    reproduce under it: 17.9% / 36.7%, a gap of 18.8pp, not 21.2pp."""
    sp = _load()
    out = tmp_path / "second_person_questions.json"
    monkeypatch.setattr(sys, "argv", ["second_person_questions.py", "--out", str(out)])
    sp.main()
    d = json.loads(out.read_text())
    c = d["conditions"]
    assert (c["base_no_wrapper"]["count"], c["base_no_wrapper"]["n"]) == (179, 1000)
    assert (c["lora_no_wrapper"]["count"], c["lora_no_wrapper"]["n"]) == (367, 1000)
    assert round(100 * d["lora_minus_base"]["diff"], 1) == 18.8
    assert d["lora_minus_base"]["p"] < 0.001
