"""The BODHI protocol (bodhi_lora/protocols/bodhi_llm.py) and its component ablations.

bodhi-llm is a small pure-Python package, so these run on the CPU with the stub engine.
The ablation tests check that each strip removes its section and nothing else, so a
prompt change in bodhi-llm fails here instead of silently not ablating (issue #69).
"""

import pytest

pytest.importorskip("bodhi")

from bodhi_lora.engine import StubEngine  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.protocols import Decoding, build  # noqa: E402
from bodhi_lora.protocols.bodhi_llm import make_ablated_bodhi_config  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

REG = load_registry(CONFIGS_DIR)


def test_none_returns_medical_default():
    cfg = make_ablated_bodhi_config("none")
    assert cfg.domain == "medical"
    assert cfg.analysis_template is None
    assert cfg.response_template is None


def test_no_calibration_strips_uncertainties_section():
    cfg = make_ablated_bodhi_config("no_calibration")
    assert cfg.analysis_template is not None
    template = cfg.analysis_template
    # Target section gone.
    assert "KEY UNCERTAINTIES" not in template
    # Sibling sections still present (only that one section was removed).
    assert "WHAT I THINK" in template
    assert "QUESTIONS TO ASK" in template
    assert "RED FLAGS" in template
    assert "SAFE RECOMMENDATIONS" in template


def test_no_questions_strips_clarifying_questions():
    cfg = make_ablated_bodhi_config("no_questions")
    template = cfg.analysis_template
    assert "QUESTIONS TO ASK" not in template
    assert "KEY UNCERTAINTIES" in template
    assert "RED FLAGS" in template
    assert "SAFE RECOMMENDATIONS" in template


def test_no_abstention_strips_safe_recommendations():
    cfg = make_ablated_bodhi_config("no_abstention")
    template = cfg.analysis_template
    assert "SAFE RECOMMENDATIONS" not in template
    assert "RED FLAGS" in template
    assert "KEY UNCERTAINTIES" in template
    assert "QUESTIONS TO ASK" in template
    # Closing humility line should survive (not part of section 7).
    assert "Be genuinely curious and humble" in template


def test_no_domain_framing_uses_general():
    cfg = make_ablated_bodhi_config("no_domain_framing")
    assert cfg.domain == "general"
    assert cfg.analysis_template is None
    assert cfg.response_template is None


@pytest.mark.parametrize(
    "component", ["no_calibration", "no_questions", "no_abstention"],
)
def test_template_has_input_placeholder(component):
    cfg = make_ablated_bodhi_config(component)
    # Exactly one literal "{input}" so BODHI's later
    # analysis_template.format(input=case_text) substitutes once.
    assert cfg.analysis_template.count("{input}") == 1


def test_invalid_component_raises():
    with pytest.raises(ValueError) as excinfo:
        make_ablated_bodhi_config("not_a_real_component")
    msg = str(excinfo.value)
    # Error should list the allowed set so the user knows what to pick.
    assert "no_calibration" in msg
    assert "no_questions" in msg
    assert "no_abstention" in msg
    assert "no_domain_framing" in msg
    assert "none" in msg


def test_format_roundtrip_substitutes_input():
    """The ablated template must work as a BODHIConfig.analysis_template:
    BODHI calls ``template.format(input=case_text)`` at runtime.
    """
    cfg = make_ablated_bodhi_config("no_calibration")
    rendered = cfg.analysis_template.format(input="patient with chest pain")
    assert "patient with chest pain" in rendered
    assert "{input}" not in rendered
    assert "KEY UNCERTAINTIES" not in rendered


def test_bodhi_makes_two_calls_and_returns_the_second_as_the_reply():
    replies = iter(["TASK TYPE: advice\nRED FLAGS: none", "Rest and drink water."])
    eng = StubEngine("m", reply=lambda msgs: next(replies))
    proto = build(REG.protocols["bodhi"], REG.models["mistral-small-24b"])
    out = proto.respond(eng, [{"role": "user", "content": "I have a cold."}], Decoding(max_new_tokens=4))
    assert out.calls == 2 and len(eng.calls) == 2
    assert out.response == "Rest and drink water." and out.analysis.startswith("TASK TYPE")
    # The first pass had 6 words against a limit of 4, so a call hit its limit.
    assert out.finish_length and out.completion_tokens == 6 + 4
    assert out.metadata["ablate_component"] == "none" and out.metadata["mode"] == "two_pass"


@pytest.mark.parametrize("pid, key", [("bodhi-no-calibration", "no_calibration"),
                                      ("bodhi-no-questions", "no_questions"),
                                      ("bodhi-no-abstention", "no_abstention"),
                                      ("bodhi-no-domain-framing", "no_domain_framing")])
def test_ablation_records_build_their_ablation(pid, key):
    proto = build(REG.protocols[pid], REG.models["mistral-small-24b"])
    assert proto.ablate == key
    assert proto.config.domain == ("general" if key == "no_domain_framing" else "medical")
