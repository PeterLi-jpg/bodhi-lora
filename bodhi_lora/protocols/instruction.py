"""Prompt protocols (impl key instruction): one instruction block, one call.

A prompt protocol adds the record's instructions to the conversation, here as the
inference-time baselines on the unchanged base (conditions such as base+clarify).
Nothing is trained. They run on the same eval prompts and with the same graders as
every other condition, so a baseline row compares with the base and LoRA rows of the
same cell prompt by prompt. The instruction texts live in the protocol records; the two
baselines so far:

clarify
    Tells the model to ask short clarifying questions when facts that would
    change the advice are missing. This is the cheapest way to get context
    seeking without training.

verbalized-confidence
    Tells the model to state its confidence in words and as a number, and to
    name what would change its answer. This follows the verbalized-confidence
    setup of Lin et al. (2022), "Teaching Models to Express Their Uncertainty
    in Words", and Tian et al. (2023), "Just Ask for Calibration". Those papers
    ask for the probability that a short answer is correct. The questions here
    are open-ended and have no single correct answer, so the prompt asks for
    confidence in the main advice.
"""

from bodhi_lora.protocols.base import CallLog, Decoding, Output, Protocol
from bodhi_lora.registry.schema import ModelConfig, ProtocolConfig

PLACEMENTS = ("system", "first_user")


def apply_instructions(messages, instructions, placement="system"):
    """Return a new message list with the instructions added.

    placement="system" puts the instructions in a system message at the
    start. If the conversation already has a system message, the
    instructions go before its text.

    placement="first_user" puts the instructions at the top of the first user
    turn, for chat templates that raise on a system message (system_role: false
    in the model record). Some templates do the same merge themselves.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement={placement!r}; use one of {PLACEMENTS}")
    msgs = [dict(m) for m in messages]
    if placement == "system":
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{instructions}\n\n{msgs[0]['content']}"
            return msgs
        return [{"role": "system", "content": instructions}] + msgs
    for m in msgs:
        if m.get("role") == "user":
            m["content"] = f"{instructions}\n\n{m['content']}"
            return msgs
    raise ValueError("first_user placement needs at least one user message")


class Instruction(Protocol):
    """The record's instructions, placed where the model's chat template accepts them."""

    kind = "prompt"

    def __init__(self, protocol: ProtocolConfig, model: ModelConfig):
        self.id = protocol.id
        self.instructions = protocol.instructions
        self.placement = "system" if model.chat_template.system_role else "first_user"

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        log = CallLog(engine, decoding, logprobs=True)
        text = log.chat(apply_instructions(messages, self.instructions, self.placement))
        return log.output(text, metadata={"instruction_placement": self.placement},
                          token_logprobs=log.last.token_logprobs)
