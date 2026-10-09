"""Tests for the prompt baselines and the ask-first protocol. CPU only."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

from bodhi_lora.protocols.ask_first import AskFirstProtocol, parse_reply, render_conversation  # noqa: E402
from bodhi_lora.protocols.instruction import BASELINE_PROMPTS, apply_instructions  # noqa: E402

CONVO = [
    {"role": "user", "content": "I have a headache."},
    {"role": "assistant", "content": "How long has it lasted?"},
    {"role": "user", "content": "Two days."},
]


def test_system_placement_adds_or_merges_a_system_message():
    out = apply_instructions(CONVO, "INSTR", "system")
    assert out[0] == {"role": "system", "content": "INSTR"}
    assert out[1:] == CONVO
    merged = apply_instructions([{"role": "system", "content": "old"}] + CONVO, "INSTR", "system")
    assert merged[0]["content"] == "INSTR\n\nold"
    assert len(merged) == len(CONVO) + 1


def test_first_user_placement_keeps_roles_alternating():
    out = apply_instructions(CONVO, "INSTR", "first_user")
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"] == "INSTR\n\nI have a headache."
    assert out[2]["content"] == "Two days."
    # The input list is not changed in place.
    assert CONVO[0]["content"] == "I have a headache."


def test_unknown_placement_is_refused():
    with pytest.raises(ValueError):
        apply_instructions(CONVO, "x", "middle")


def test_both_baselines_are_defined():
    assert set(BASELINE_PROMPTS) == {"clarify", "verbalized_confidence"}
    assert "Confidence:" in BASELINE_PROMPTS["verbalized_confidence"]


def test_parse_reply_with_and_without_marker():
    text = "<plan>1. age\n2. none\n3. rest</plan>\nREPLY:\nHow old are you? Rest for now."
    reply, plan, found = parse_reply(text)
    assert found and reply == "How old are you? Rest for now."
    assert plan.startswith("1. age")
    reply, plan, found = parse_reply("<plan>p</plan>\nJust rest.")
    assert not found and reply == "Just rest." and plan == "p"
    reply, _, found = parse_reply("**REPLY:** Drink water.")
    assert found and reply == "Drink water."


def test_ask_first_makes_one_call_with_the_whole_conversation():
    seen = []

    def chat(msgs):
        seen.append(msgs)
        return "<plan>plan</plan>\nREPLY:\nWhere is the pain?"

    resp = AskFirstProtocol(chat).complete(CONVO)
    assert len(seen) == 1 and len(seen[0]) == 1 and seen[0][0]["role"] == "user"
    assert render_conversation(CONVO) in seen[0][0]["content"]
    assert resp.content == "Where is the pain?"
    assert resp.analysis == "plan"
    assert resp.metadata["protocol"] == "ask_first"
    assert resp.metadata["reply_marker_found"] is True
