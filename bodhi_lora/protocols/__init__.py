"""Protocols: how a reply is made at inference, from a protocol record.

    proto = build(reg.protocols["bodhi"], reg.models[cell.model])
    out = proto.respond(engine, messages, decoding_for(proto, ...))

IMPLS maps a record's impl key to its class:
  bodhi_llm    BODHI through the bodhi-llm package, with its component ablations and
               section order (red flags first)
  ask_first    a three-question plan, then a reply that asks first
  instruction  one instruction block added to the conversation (the prompt baselines)
  abstain      a confidence rating, then one question or the answer (MediQ-style)
build(None, model) is the identity protocol of the plain conditions "base" and "lora".
"""
from typing import Optional

from bodhi_lora.protocols.abstain import Abstain
from bodhi_lora.protocols.ask_first import AskFirst
from bodhi_lora.protocols.base import CallLog, Decoding, Identity, Output, Protocol
from bodhi_lora.protocols.bodhi_llm import BodhiLLM
from bodhi_lora.protocols.instruction import Instruction
from bodhi_lora.registry.schema import PROTOCOL_IMPLS, ModelConfig, ProtocolConfig

IMPLS = {"bodhi_llm": BodhiLLM, "ask_first": AskFirst, "instruction": Instruction, "abstain": Abstain}
# The registry checks records against its copy of these keys and kinds, without importing
# this package; a new implementation goes in both.
if {k: c.kind for k, c in IMPLS.items()} != PROTOCOL_IMPLS:
    raise ImportError("bodhi_lora/protocols IMPLS and registry/schema.py PROTOCOL_IMPLS disagree")

__all__ = ["CallLog", "Decoding", "IMPLS", "Identity", "Output", "Protocol", "build", "decoding_for"]


def build(protocol: Optional[ProtocolConfig], model: ModelConfig) -> Protocol:
    if protocol is None:
        return Identity()
    if protocol.impl not in IMPLS:
        raise ValueError(f"protocol {protocol.id}: no implementation {protocol.impl!r}; "
                         f"implementations are {sorted(IMPLS)}")
    return IMPLS[protocol.impl](protocol, model)


def decoding_for(proto: Protocol, *, max_new_tokens: int, max_new_tokens_per_pass: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: Optional[int] = None) -> Decoding:
    """A wrapper's calls get the per-pass limit; a single-call protocol gets the reply limit."""
    limit = max_new_tokens_per_pass if proto.kind == "wrapper" else max_new_tokens
    return Decoding(max_new_tokens=limit, temperature=temperature, top_p=top_p, seed=seed)
