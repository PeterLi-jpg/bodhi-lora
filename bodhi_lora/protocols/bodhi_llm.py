"""BODHI as a protocol (impl key bodhi_llm), with its component ablations and section order.

BODHI (the bodhi-llm package) makes two calls per reply: pass 1 writes an analysis of
the case, pass 2 writes the reply from it. Only the pass-2 text is the reply (and the
training target of a trace); the analysis is kept for audit.

Three record params change the protocol, each from data in the protocol record:
  ablate             removes one component from pass 1 (EXPERIMENTS.md E26), below.
  first_section      moves one numbered pass-1 section to the top and renumbers the
                     list, e.g. "RED FLAGS" (the red-flags-first variant, E17).
  reply_instruction  a line added to the pass-2 prompt, before its last line, for the
                     conversation and emergency replies. Documentation tasks
                     (TECHNICAL and HYBRID in bodhi-llm's routing) keep their format.

The record's params.ablate removes one component from pass 1 (EXPERIMENTS.md E26).
BODHI bundles several distinct things into pass 1:

  - domain framing       (medical vs general)
  - calibration prompt   ("4. KEY UNCERTAINTIES")
  - multi-turn           ("5. QUESTIONS TO ASK")
  - abstention           ("7. SAFE RECOMMENDATIONS")

This module produces ``BODHIConfig`` instances with one of those components
disabled, by overriding ``analysis_template`` so it skips the corresponding
numbered section of the medical prompt.

Pass 2 (response_template) is intentionally left unchanged: the response
prompt branches on what Pass 1 produced (task_type, audience), so removing a
section from Pass 1 already cascades into Pass 2. Modifying Pass 1 alone is
sufficient to ablate the analysis-driven components.

Fail-fast on prompt drift: if bodhi-llm changes the prompt format (renames a
header, renumbers, etc.) the regex below will silently mismatch and we'd ship
an ablation that doesn't actually ablate. Each strip path verifies the target
text was both present before and absent after.
"""

import re
from typing import Optional, Set

from bodhi import BODHI, BODHIConfig
from bodhi.prompts import _detect_task_type, render_analysis_prompt
from pydantic import BaseModel, ConfigDict

from bodhi_lora.protocols.base import CallLog, Decoding, Output, Protocol
from bodhi_lora.registry.schema import ModelConfig, ProtocolConfig


# Sentinel used during template construction so the literal "{input}" inside
# the rendered prompt isn't formatted twice when BODHI later calls
# ``analysis_template.format(input=case_text)``. We render with this sentinel,
# strip the target section, then swap the sentinel back to "{input}".
_PLACEHOLDER = "__BODHI_INPUT_PLACEHOLDER__"

ALLOWED_COMPONENTS: Set[str] = {
    "none",
    "no_calibration",
    "no_questions",
    "no_abstention",
    "no_domain_framing",
}

# Map ablation tag -> (section header substring used for the invariant check,
# regex pattern that strips that section from the rendered medical template).
# Patterns assume ``re.MULTILINE | re.DOTALL`` and match from the numbered
# header up through (but not including) the start of the next numbered header.
# For section 7 (the last one), match through the closing humility line.
_SECTION_PATTERNS = {
    "no_calibration": (
        "KEY UNCERTAINTIES",
        re.compile(r"^\s*4\.\s*KEY UNCERTAINTIES:.*?(?=^\s*5\.)", re.MULTILINE | re.DOTALL),
    ),
    "no_questions": (
        "QUESTIONS TO ASK",
        re.compile(r"^\s*5\.\s*QUESTIONS TO ASK:.*?(?=^\s*6\.)", re.MULTILINE | re.DOTALL),
    ),
    "no_abstention": (
        "SAFE RECOMMENDATIONS",
        re.compile(
            r"^\s*7\.\s*SAFE RECOMMENDATIONS:.*?(?=Be genuinely curious and humble)",
            re.MULTILINE | re.DOTALL,
        ),
    ),
}


def make_ablated_bodhi_config(component: str) -> BODHIConfig:
    """Return a ``BODHIConfig`` with the named component disabled.

    Args:
        component: One of ``"none"``, ``"no_calibration"``, ``"no_questions"``,
            ``"no_abstention"``, ``"no_domain_framing"``.

    Returns:
        A ``BODHIConfig``. For ``"none"`` and ``"no_domain_framing"`` no
        template overrides are needed (the domain field alone fully specifies
        them). For the three section-strip ablations, ``analysis_template``
        is set; ``response_template`` remains ``None``.

    Raises:
        ValueError: on unknown component name.
        RuntimeError: if bodhi-llm's prompt format has drifted from what we
            expect (regex no-match), so the ablation would silently be a
            no-op.
    """
    if component not in ALLOWED_COMPONENTS:
        raise ValueError(
            f"unknown component {component!r}; "
            f"allowed: {sorted(ALLOWED_COMPONENTS)}"
        )

    if component == "none":
        return BODHIConfig(domain="medical")

    if component == "no_domain_framing":
        # Swap the medical-domain prompt for the general-domain one.
        # No regex strip needed: bodhi.prompts.render_analysis_prompt picks
        # the general template when domain != "medical".
        return BODHIConfig(domain="general")

    # Section-strip path: render once with a sentinel for {input}, strip the
    # target section, swap the sentinel back to {input} so BODHI's later
    # .format(input=case_text) call works.
    rendered = render_analysis_prompt(_PLACEHOLDER, domain="medical")
    section_marker, pattern = _SECTION_PATTERNS[component]

    if section_marker not in rendered:
        raise RuntimeError(
            f"bodhi-llm prompt format changed: section "
            f"{section_marker!r} not found in rendered medical template "
            f"(update bodhi_lora/protocols/bodhi_llm.py)"
        )

    stripped = pattern.sub("", rendered)

    if section_marker in stripped:
        raise RuntimeError(
            f"bodhi-llm prompt format changed: section "
            f"{section_marker!r} still present after strip "
            f"(update bodhi_lora/protocols/bodhi_llm.py)"
        )

    template = stripped.replace(_PLACEHOLDER, "{input}")
    return BODHIConfig(analysis_template=template, domain="medical")


# A numbered pass-1 section: a paragraph that starts "<n>. <HEADER>:".
_SECTION_RE = re.compile(r"^\s*(\d+)\.\s*([A-Z][A-Z' ]*?)\s*:")


def move_section_first(template: str, header: str) -> str:
    """The analysis template with the numbered section `header` first, renumbered 1, 2, ...

    The template is split into paragraphs at blank lines; the numbered paragraphs are the
    sections, and the paragraphs before and after them stay where they are. Raises
    RuntimeError when the section is missing or not unique, so a prompt change in
    bodhi-llm fails here instead of silently leaving the order as it was.
    """
    paragraphs = template.split("\n\n")
    numbered = [i for i, par in enumerate(paragraphs) if _SECTION_RE.match(par)]
    hits = [i for i in numbered if _SECTION_RE.match(paragraphs[i]).group(2) == header]
    if len(hits) != 1 or numbered != list(range(numbered[0], numbered[-1] + 1)):
        raise RuntimeError(f"bodhi-llm prompt format changed: section {header!r} found {len(hits)} "
                           "times, or the sections are not one block (update "
                           "bodhi_lora/protocols/bodhi_llm.py)")
    order = hits + [i for i in numbered if i != hits[0]]
    sections = [_SECTION_RE.sub(lambda m: m.group(0).replace(m.group(1), str(k), 1), paragraphs[i], count=1)
                for k, i in enumerate(order, start=1)]
    out = paragraphs[:numbered[0]] + sections + paragraphs[numbered[-1] + 1:]
    return "\n\n".join(out)


def insert_before_last_line(prompt: str, line: str) -> str:
    """The pass-2 prompt with `line` as a paragraph before its last line ("Write your
    response:" and its kin), so the prompt still ends with bodhi-llm's own cue."""
    head, _, last = prompt.rstrip().rpartition("\n")
    return f"{head.rstrip()}\n\n{line}\n\n{last}"


# Task types whose pass-2 prompt asks for a fixed document format. bodhi-llm's routing
# (bodhi.prompts._detect_task_type) names them; a reply instruction would break the format.
_FORMAT_TASKS = ("TECHNICAL", "HYBRID")


class BodhiParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ablate: str = "none"
    first_section: Optional[str] = None
    reply_instruction: Optional[str] = None


def make_bodhi_config(params: BodhiParams) -> BODHIConfig:
    """The BODHIConfig of a record's params: the ablation, then the section order."""
    config = make_ablated_bodhi_config(params.ablate)
    if params.first_section is None:
        return config
    if config.domain != "medical":
        raise ValueError("first_section needs the medical prompt; the general one has other sections")
    template = config.analysis_template or render_analysis_prompt(_PLACEHOLDER, domain="medical")
    moved = move_section_first(template.replace("{input}", _PLACEHOLDER), params.first_section)
    return BODHIConfig(analysis_template=moved.replace(_PLACEHOLDER, "{input}"), domain="medical")


class _InstructedBODHI(BODHI):
    """BODHI whose pass-2 prompt carries one more line for conversation replies.

    It overrides _get_response_prompt, the method bodhi-llm 0.1.4 (bodhi/core.py) calls
    for pass 2, and keeps bodhi-llm's routing by task type and audience.
    """

    def __init__(self, *args, reply_instruction: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.reply_instruction = reply_instruction

    def _get_response_prompt(self, case_text: str, analysis: str) -> str:
        prompt = super()._get_response_prompt(case_text, analysis)
        if _detect_task_type(analysis) in _FORMAT_TASKS:
            return prompt
        return insert_before_last_line(prompt, self.reply_instruction)


class BodhiLLM(Protocol):
    """Two calls per reply through bodhi-llm; pass 2's text is the reply."""

    kind = "wrapper"

    def __init__(self, protocol: ProtocolConfig, model: ModelConfig):
        self.id = protocol.id
        self.params = BodhiParams.model_validate(protocol.params)
        self.ablate = self.params.ablate
        # Built once, so a prompt drift in bodhi-llm fails before any call.
        self.config = make_bodhi_config(self.params)
        if self.params.reply_instruction and not hasattr(BODHI, "_get_response_prompt"):
            raise RuntimeError("bodhi-llm has no BODHI._get_response_prompt; the reply "
                               "instruction cannot reach pass 2 (update bodhi_lora/protocols/bodhi_llm.py)")

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        log = CallLog(engine, decoding)
        # A BODHI object per reply: it holds the call function, and replies run in
        # parallel threads, so each needs its own call log.
        if self.params.reply_instruction:
            bodhi = _InstructedBODHI(chat_function=log.chat, config=self.config,
                                     reply_instruction=self.params.reply_instruction)
        else:
            bodhi = BODHI(chat_function=log.chat, config=self.config)
        resp = bodhi.complete(messages)
        metadata = dict(resp.metadata or {})
        metadata["ablate_component"] = self.ablate
        if self.params.first_section:
            metadata["first_section"] = self.params.first_section
        return log.output(resp.content, resp.analysis, metadata)
