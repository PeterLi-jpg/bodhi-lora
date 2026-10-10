"""Ask-first (impl key ask_first): a second teaching protocol, distinct from BODHI.

A cell uses it in place of BODHI by naming the ask-first protocol record.

How it differs from BODHI (bodhi-llm 0.1.4, bodhi/core.py):
  - One model call, not two. BODHI makes one call for an analysis and a
    second call that writes the reply from that analysis.
  - A short plan with three questions, not BODHI's numbered analysis
    sections (task type, audience, key uncertainties, questions to ask,
    safe recommendations and so on).
  - A fixed order in the reply: questions about missing facts first, then
    the advice that is safe now, then warning signs.

Only the text after "REPLY:" becomes the response (the training target),
in the same way that only BODHI's pass 2 output does. The plan is kept as
`analysis` for audit. Like BODHI, the whole conversation goes into one user
message, so every chat template accepts it.
"""

import re

from bodhi_lora.protocols.base import CallLog, Decoding, Output, Protocol
from bodhi_lora.registry.schema import ModelConfig, ProtocolConfig

ASK_FIRST_TEMPLATE = """\
Below is a conversation with a person who asks a health question. Write the \
next assistant reply.

First, plan inside <plan> and </plan>. In the plan, answer three questions in \
a few lines:
1. Which missing facts would change the advice? List at most three.
2. Which warning signs, if present, need urgent care?
3. What advice is safe to give now?

Then write "REPLY:" on its own line, and after it write the reply to the \
person. In the reply:
- If a missing fact would change the advice, ask about it first, in short, \
direct questions.
- Then give the advice that is safe now, and say what would change it.
- Name the warning signs that need urgent care, if any apply.
- Do not use a general disclaimer in place of advice.
Do not mention the plan in the reply.

Conversation:
{conversation}
"""

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
_REPLY_RE = re.compile(r"^\s*\**REPLY:\**\s*", re.MULTILINE)
# The fixed format the template asks for: the plan tags and the reply line. A reply that
# repeats them has copied the internal format (the leak rule of the ask-first record).
FORMAT_FIELDS = ("<plan>", "</plan>", "REPLY:")


def render_conversation(messages) -> str:
    """'user: ...' / 'assistant: ...' lines, one block per turn."""
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{m.get('role', 'user')}: {content}")
    return "\n\n".join(parts)


def parse_reply(text):
    """Split a raw generation into (reply, plan, marker_found).

    The reply is the text after the first "REPLY:" line. Without the marker,
    the reply is the text with the plan block removed, and marker_found is
    False so the trace can be audited or dropped.
    """
    plan_match = _PLAN_RE.search(text)
    plan = plan_match.group(1).strip() if plan_match else ""
    reply_match = _REPLY_RE.search(text)
    if reply_match:
        return text[reply_match.end():].strip(), plan, True
    return _PLAN_RE.sub("", text).strip(), plan, False


def build_prompt(messages) -> list:
    """The one user message of the call: the template with the whole conversation."""
    return [{"role": "user", "content": ASK_FIRST_TEMPLATE.format(conversation=render_conversation(messages))}]


class AskFirst(Protocol):
    """One call; the reply is the text after the REPLY: line."""

    kind = "wrapper"

    def __init__(self, protocol: ProtocolConfig, model: ModelConfig):
        self.id = protocol.id

    def format_fields(self) -> tuple[str, ...]:
        """The plan tags and the REPLY: line of the template's fixed format."""
        return FORMAT_FIELDS

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        log = CallLog(engine, decoding)
        reply, plan, found = parse_reply(log.chat(build_prompt(messages)))
        return log.output(reply, plan, {"protocol": self.id, "mode": "single_pass",
                                        "reply_marker_found": found, "format_parsed": found})
