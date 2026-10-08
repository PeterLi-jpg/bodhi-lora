"""Ask-first: a second chain-of-thought teaching protocol, distinct from BODHI.

Reviewer HomM (W1) asked whether the result depends on the one CoT protocol
the paper used. This module gives a second protocol. A cell uses it in place
of BODHI with one setting: generate_traces.py --protocol ask_first, or
PROTOCOL=ask_first for rebuttal/launch/run_cell.sh.

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
import time
from dataclasses import dataclass, field

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


@dataclass
class ProtocolResponse:
    """Same fields as bodhi.BODHIResponse, so callers treat both alike."""
    content: str
    analysis: str
    metadata: dict = field(default_factory=dict)


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


class AskFirstProtocol:
    """Single-pass ask-first wrapper with the BODHI .complete() interface."""

    name = "ask_first"

    def __init__(self, chat_function):
        self.chat = chat_function

    def build_prompt(self, messages) -> list:
        return [{"role": "user",
                 "content": ASK_FIRST_TEMPLATE.format(conversation=render_conversation(messages))}]

    def complete(self, messages) -> ProtocolResponse:
        start = time.time()
        raw = self.chat(self.build_prompt(messages))
        reply, plan, found = parse_reply(raw)
        return ProtocolResponse(
            content=reply,
            analysis=plan,
            metadata={
                "protocol": self.name,
                "mode": "single_pass",
                "reply_marker_found": found,
                "total_time": time.time() - start,
            },
        )
