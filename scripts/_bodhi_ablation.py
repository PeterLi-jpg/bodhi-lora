"""BODHI wrapper component ablations for paper figure 3 (issue #69).

The paper claims the LoRA internalizes the BODHI wrapper's epistemic virtues.
A reviewer will reasonably ask which component of the wrapper actually drives
the behavioral shift, since BODHI bundles several distinct things into Pass 1:

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
from typing import Set

from bodhi import BODHIConfig
from bodhi.prompts import render_analysis_prompt


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
            f"(re-vendor or update _bodhi_ablation.py)"
        )

    stripped = pattern.sub("", rendered)

    if section_marker in stripped:
        raise RuntimeError(
            f"bodhi-llm prompt format changed: section "
            f"{section_marker!r} still present after strip "
            f"(re-vendor or update _bodhi_ablation.py)"
        )

    template = stripped.replace(_PLACEHOLDER, "{input}")
    return BODHIConfig(analysis_template=template, domain="medical")
