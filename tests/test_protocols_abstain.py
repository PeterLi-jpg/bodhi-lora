"""The MediQ-style abstention protocol (bodhi_lora/protocols/abstain.py) on the stub engine."""
import pytest

pytest.importorskip("bodhi")

from bodhi_lora.engine import StubEngine  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.protocols import Decoding, build  # noqa: E402
from bodhi_lora.protocols.abstain import parse_rating  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

REG = load_registry(CONFIGS_DIR)
MESSAGES = [{"role": "user", "content": "I feel dizzy when I stand up. Why?"}]


@pytest.mark.parametrize("text, score", [
    ("REASON: age and drugs are unknown.\nDECISION: Somewhat Unconfident", 2),
    ("REASON: clear case.\nDECISION: Very Confident.", 5),
    ("DECISION: neither confident nor unconfident", 3),
    ("DECISION: Neither Confident or Unconfident", 3),
    ("DECISION: very unconfident", 1),
    ("I would say somewhat confident", 4),
    ("no label here", 0),
])
def test_parse_rating_uses_the_mediq_labels(text, score):
    assert parse_rating(text)[0] == score


def test_the_decision_line_wins_over_labels_in_the_reason():
    score, reason = parse_rating("REASON: I am not very confident about the dose.\nDECISION: Somewhat Unconfident")
    assert score == 2 and reason.startswith("I am not very confident")


def _run(rating):
    proto = build(REG.protocols["mediq-abstain"], REG.models["mistral-small-24b"])

    def reply(messages):
        content = messages[-1]["content"]
        if "How confident are you" in content:
            return rating
        if "Ask the person one short" in content:
            return "How long have you felt dizzy? "
        return "Stand up slowly and drink water."

    eng = StubEngine("m", reply=reply)
    return proto.respond(eng, MESSAGES, Decoding(max_new_tokens=64)), eng


def test_below_the_threshold_it_asks_one_question():
    out, eng = _run("REASON: no history.\nDECISION: Somewhat Unconfident")
    assert out.response == "How long have you felt dizzy?" and out.calls == 2
    assert out.metadata["decision"] == "ask" and out.metadata["confidence_score"] == 2
    assert "dizzy when I stand up" in eng.calls[1][-1]["content"]


def test_at_the_threshold_it_answers_the_conversation_as_given():
    out, eng = _run("REASON: common case.\nDECISION: Somewhat Confident")
    assert out.response == "Stand up slowly and drink water." and out.calls == 2
    assert out.metadata["decision"] == "answer" and eng.calls[1] == MESSAGES
    assert out.analysis.startswith("REASON")


def test_an_unreadable_rating_asks():
    out, _ = _run("I cannot say.")
    assert out.metadata["confidence_score"] == 0 and out.metadata["decision"] == "ask"
