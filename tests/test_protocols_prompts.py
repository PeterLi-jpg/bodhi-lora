"""Prompt protocols and the ask-first protocol, through the stub engine. CPU only."""

import hashlib

import pytest

pytest.importorskip("pydantic")
# bodhi_lora.protocols imports the BODHI protocol, which needs bodhi-llm from the train
# extra; the CPU install (.[dev]) skips these tests, as the other protocol tests do.
pytest.importorskip("bodhi")

from bodhi_lora.engine import StubEngine  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.protocols import Decoding, build, decoding_for  # noqa: E402
from bodhi_lora.protocols.ask_first import parse_reply, render_conversation  # noqa: E402
from bodhi_lora.protocols.instruction import apply_instructions  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

REG = load_registry(CONFIGS_DIR)

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


# sha256 of the two baseline instructions as scripts/baselines/prompts.py held them at
# 02ac9a3; the protocol records must keep the text byte for byte.
BASELINE_SHA256 = {
    "clarify": "3671672511dc433afaf9514fc743b9d11a706c5667a1d2dc61b191218b955ec5",
    "verbalized-confidence": "9ffa2664c1c913456244464812ea2dff14c31df0731389df0fc3f95f7f317f43",
}


def test_baseline_records_keep_the_instruction_text():
    for pid, digest in BASELINE_SHA256.items():
        text = REG.protocols[pid].instructions
        assert hashlib.sha256(text.encode()).hexdigest() == digest, pid
    assert "Confidence:" in REG.protocols["verbalized-confidence"].instructions


def test_prompt_protocol_places_instructions_where_the_template_accepts_them():
    clarify = REG.protocols["clarify"]
    eng = StubEngine("m", reply="ok")
    out = build(clarify, REG.models["mistral-small-24b"]).respond(eng, CONVO, Decoding(max_new_tokens=8))
    assert eng.calls[-1][0]["role"] == "system" and out.metadata["instruction_placement"] == "system"
    out = build(clarify, REG.models["biomistral-7b"]).respond(eng, CONVO, Decoding(max_new_tokens=8))
    assert eng.calls[-1][0]["role"] == "user" and eng.calls[-1][0]["content"].startswith(clarify.instructions)
    assert (out.calls, out.response, out.metadata["instruction_placement"]) == (1, "ok", "first_user")


def test_identity_and_wrappers_get_their_token_limits():
    model = REG.models["mistral-small-24b"]
    limits = {name: decoding_for(build(REG.protocols[name] if name else None, model),
                                 max_new_tokens=1024, max_new_tokens_per_pass=2048).max_new_tokens
              for name in (None, "clarify", "bodhi", "ask-first")}
    assert limits == {None: 1024, "clarify": 1024, "bodhi": 2048, "ask-first": 2048}


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
    eng = StubEngine("m", reply="<plan>plan</plan>\nREPLY:\nWhere is the pain?")
    proto = build(REG.protocols["ask-first"], REG.models["mistral-small-24b"])
    out = proto.respond(eng, CONVO, Decoding(max_new_tokens=64))
    assert len(eng.calls) == 1 and len(eng.calls[0]) == 1 and eng.calls[0][0]["role"] == "user"
    assert render_conversation(CONVO) in eng.calls[0][0]["content"]
    assert (out.response, out.analysis, out.calls) == ("Where is the pain?", "plan", 1)
    assert out.metadata["protocol"] == "ask-first" and out.metadata["reply_marker_found"] is True
    # The tokens count the whole generation, plan included.
    assert out.completion_tokens == 6 and not out.finish_length


def test_ask_first_format_fields_are_the_ones_its_template_asks_for():
    """The leak rule of the ask-first record may only count these (tests/test_no_lexicon.py)."""
    from bodhi_lora.protocols.ask_first import ASK_FIRST_TEMPLATE, FORMAT_FIELDS

    proto = build(REG.protocols["ask-first"], REG.models["mistral-small-24b"])
    assert proto.format_fields() == FORMAT_FIELDS == ("<plan>", "</plan>", "REPLY:")
    assert all(f in ASK_FIRST_TEMPLATE for f in FORMAT_FIELDS)
    assert set(REG.protocols["ask-first"].leak_rules) == {"analysis-format"}
