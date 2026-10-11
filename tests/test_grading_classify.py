"""The judge classifier (bodhi_lora/grading/classify.py): its questions are the analysis
plan's, word for word; parse and retries; the aggregate; and the hand-label gate."""
import json
import re

import pytest

from bodhi_lora.engine import StubEngine
from bodhi_lora.grading.classify import (CLASSIFY_PROMPT_MARKER, GATE_KAPPA, QUESTION_ASKS, QUESTION_URGENT,
                                         STUB_CLASSIFY_REPLY, aggregate, binary, classify, cohen_kappa, gate)
from bodhi_lora.grading.rubric import Grader
from bodhi_lora.paths import REPO_ROOT


def _classify(reply, conversation="[user] my chest hurts", response="How long has it hurt?"):
    eng = StubEngine("judge", reply=reply)
    return classify(Grader(eng), conversation, response), eng


def test_the_questions_are_the_text_the_analysis_plan_fixed():
    """The hand labels and the judge must answer the same questions (ANALYSIS_PLAN.md
    section 12, 2026-10-10, item 6). The plan quotes them with double quotes."""
    plan = " ".join((REPO_ROOT / "docs" / "ANALYSIS_PLAN.md").read_text().split())
    for q in (QUESTION_ASKS, QUESTION_URGENT):
        assert f'"{q}"' in plan, q[:40]


def test_a_valid_answer_parses_and_the_prompt_carries_both_texts():
    (answer, raw), eng = _classify('```json\n{"asks_user": true, "urgent_care": "if", "reason": "asks"}\n```')
    assert answer == {"asks_user": True, "urgent_care": "if", "reason": "asks"}
    prompt = eng.calls[0][-1]["content"]
    assert CLASSIFY_PROMPT_MARKER in prompt and "my chest hurts" in prompt and "How long has it hurt?" in prompt


def test_an_invalid_answer_is_retried_three_times_then_missing():
    (answer, raw), eng = _classify('{"asks_user": "yes", "urgent_care": "later"}')
    assert answer is None and len(eng.calls) == 3 and "later" in raw


def test_the_template_holds_no_word_list():
    """A semantic rubric, not a keyword rule (docs/CONVENTIONS.md rule 15): the module
    compiles no regular expression."""
    src = (REPO_ROOT / "bodhi_lora" / "grading" / "classify.py").read_text()
    assert not re.search(r"\bre\.(compile|search|match|findall|sub)\b", src)


def test_aggregate_counts_rates_and_missing_answers():
    a = json.loads(STUB_CLASSIFY_REPLY)
    b = {"asks_user": True, "urgent_care": "now", "reason": ""}
    agg = aggregate([a, b, None])
    assert (agg["n"], agg["n_parsed"], agg["n_missing"]) == (3, 2, 1)
    assert agg["asks_user_rate"] == 0.5 and agg["urgent_care_any_rate"] == 0.5


def test_binary_reads_urgent_care_as_now_or_if_against_no():
    assert [binary("urgent_care", v) for v in ("now", "if", "no", None)] == [True, True, False, None]
    assert binary("asks_user", False) is False
    with pytest.raises(ValueError):
        binary("other", True)


def test_the_gate_passes_full_agreement_and_fails_chance_agreement():
    rows = [{"prompt_id": f"p{i}", "reference": i % 2 == 0, "judge": i % 2 == 0} for i in range(60)]
    res = gate(rows, "asks_user")
    assert res["passed"] and res["kappa"] == 1.0 and res["n"] == 60
    # The judge says yes to everything: kappa 0, the gate fails.
    rows = [dict(r, judge=True) for r in rows]
    res = gate(rows, "asks_user")
    assert not res["passed"] and res["kappa"] == pytest.approx(0.0)


def test_the_gate_needs_the_lower_end_too_and_counts_missing_rows():
    # 90% agreement on balanced labels: kappa 0.8, at the threshold, but with 30 prompts the
    # lower end of the prompt bootstrap falls below 0.70, so the gate fails.
    rows = [{"prompt_id": f"p{i}", "reference": i < 15, "judge": (i < 15) != (i in (0, 1, 15))} for i in range(30)]
    rows.append({"prompt_id": "p99", "reference": None, "judge": True})
    res = gate(rows, "asks_user")
    assert res["n_missing"] == 1 and res["kappa"] == pytest.approx(cohen_kappa(
        [r["reference"] for r in rows[:30]], [r["judge"] for r in rows[:30]]))
    assert res["kappa"] >= GATE_KAPPA - 0.01 and res["ci95"][0] < 0.70 and not res["passed"]
