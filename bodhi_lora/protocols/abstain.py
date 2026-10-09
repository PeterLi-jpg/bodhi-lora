"""Confidence check, then ask or answer (impl key abstain): a MediQ-style abstention baseline.

MediQ (Li et al., NeurIPS 2024, github.com/stellalisy/mediQ) lets an expert model
abstain from answering and ask a follow-up question when it is not confident. Its
"Scale" variant with a rationale (expert_system["scale_RG"] in src/prompts.py) asks for
a one-sentence reason and a rating on a five-label scale; src/expert_basics.py
(parse_likert_scale) maps the labels to 5 to 1, and src/expert_functions.py abstains
when the score is below SCALE_THRESHOLD = 4.0, so it asks unless the model is
"somewhat" or "very" confident. A rating that matches no label scores 0, so it asks.

Here, as an inference-time baseline on the unchanged base (EXPERIMENTS.md E12), with
two calls like BODHI and no self-consistency:
  pass 1  the record's params.rating_prompt with the conversation: how confident is the
          model that it has the information to answer safely? A reason and a decision.
  pass 2  below params.threshold: the record's params.question_prompt, and the reply is
          the one most important question. Otherwise the conversation as given, and the
          reply is the answer.
MediQ asks about picking the correct option of a multiple-choice question; the prompts
here ask about a safe answer to an open question, because the benchmarks are open-ended.
The labels, their scores and the default threshold are MediQ's.
"""
import re

from pydantic import BaseModel, ConfigDict, Field

from bodhi_lora.protocols.ask_first import render_conversation
from bodhi_lora.protocols.base import CallLog, Decoding, Output, Protocol
from bodhi_lora.registry.schema import ModelConfig, ProtocolConfig

# MediQ's labels and scores, in the order parse_likert_scale checks them: when a reply
# names several labels, the first one in this order wins, as in MediQ.
SCALE = (
    ("very confident", 5),
    ("somewhat confident", 4),
    ("neither confident nor unconfident", 3),
    ("neither confident or unconfident", 3),
    ("somewhat unconfident", 2),
    ("very unconfident", 1),
)
_DECISION_RE = re.compile(r"DECISION\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_PUNCT_RE = re.compile(r"[.,;:]")


class AbstainParams(BaseModel):
    """params of an abstain protocol record."""

    model_config = ConfigDict(extra="forbid")
    # Ask when the rating scores below this (MediQ's SCALE_THRESHOLD).
    threshold: float = Field(default=4.0, gt=0, le=5)
    # Both prompts take {conversation}; the rating prompt ends with a DECISION: line.
    rating_prompt: str
    question_prompt: str


def parse_rating(text: str) -> tuple[int, str]:
    """(score, reason) of a pass-1 reply: the label after DECISION:, else anywhere."""
    reason = ""
    m = re.search(r"REASON\s*:\s*(.*?)(?:\n|DECISION\s*:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()
    decision = _DECISION_RE.search(text)
    part = decision.group(1) if decision else text
    part = _PUNCT_RE.sub("", part.lower())
    for label, score in SCALE:
        if label in part:
            return score, reason
    return 0, reason


class Abstain(Protocol):
    """Rate the information, then ask the most important question or answer."""

    kind = "wrapper"

    def __init__(self, protocol: ProtocolConfig, model: ModelConfig):
        self.id = protocol.id
        self.params = AbstainParams.model_validate(protocol.params)

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        log = CallLog(engine, decoding)
        conversation = render_conversation(messages)
        rating = log.chat([{"role": "user", "content":
                            self.params.rating_prompt.format(conversation=conversation)}])
        score, reason = parse_rating(rating)
        asks = score < self.params.threshold
        if asks:
            reply = log.chat([{"role": "user", "content":
                               self.params.question_prompt.format(conversation=conversation)}]).strip()
        else:
            reply = log.chat(messages)
        return log.output(reply, rating, {"protocol": self.id, "confidence_score": score,
                                          "reason": reason, "decision": "ask" if asks else "answer",
                                          "threshold": self.params.threshold})
