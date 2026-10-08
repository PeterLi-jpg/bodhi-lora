"""Tests for rebuttal/analyses/second_person_questions.py."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
