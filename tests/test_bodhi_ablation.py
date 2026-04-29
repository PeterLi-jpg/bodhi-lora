"""Pure-logic tests for the BODHI component-ablation wrapper (issue #69).

bodhi-llm is a small pure-Python package (no torch/vllm deps), so we can
import it directly without the ``import_with_mocks`` shim used elsewhere.

CI (.github/workflows/ci.yml) installs only pyyaml/pytest/numpy, not
bodhi-llm. Skip the whole module if bodhi isn't importable so the test
suite stays green there; locally and on production pods the dependency
is installed and the regex-strip invariants are exercised end-to-end.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("bodhi")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._bodhi_ablation import make_ablated_bodhi_config


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
