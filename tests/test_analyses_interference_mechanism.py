"""Tests for rebuttal/analyses/interference_mechanism.py.

The golden test reruns the script on the committed results_modal/ data and pins the
leakage numbers posted in the rebuttal (54.6% under LoRA+CoT, red-flag 1.33 leaked vs
1.73 clean). Note the denominator: the script counts responses with a parsed epistemic
grade (507/928). Over every LoRA+CoT completion in the HealthBench files the same rule
gives 532/959 = 55.5%, which is the figure the audit recomputed.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "interference_mechanism", ROOT / "rebuttal" / "analyses" / "interference_mechanism.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leak_threshold_counts_distinct_headers():
    im = _load()
    patient = "Call your doctor if the fever lasts more than three days."
    one_header = "RED FLAGS: none noted. Rest and drink fluids."
    two_headers = "1. TASK TYPE: HYBRID\n6. RED FLAGS: None\n7. SAFE RECOMMENDATIONS: rest"
    assert im.count_leak_markers(patient) == 0
    assert not im.is_leaked(one_header)
    assert im.count_leak_markers(two_headers) == 3
    assert im.is_leaked(two_headers)
    # Case-insensitive, as in the committed run.
    assert im.is_leaked("task type: x\nkey uncertainties: y")


def test_golden_values_on_committed_results(tmp_path, monkeypatch):
    im = _load()
    out = tmp_path / "interference_mechanism.json"
    monkeypatch.setattr(sys, "argv", ["interference_mechanism.py", "--out", str(out)])
    im.main()
    d = json.loads(out.read_text())
    lb = d["configs"]["lora_bodhi"]
    assert lb["n"] == 928
    assert round(100 * lb["leaked_rate"], 1) == 54.6
    assert d["decisive"]["n_leaked"] == 507
    assert d["decisive"]["n_clean"] == 421
    assert round(d["decisive"]["lora_bodhi_leaked_red_flag"], 2) == 1.33
    assert round(d["decisive"]["lora_bodhi_clean_red_flag"], 2) == 1.73
    assert round(d["decisive"]["lora_red_flag"], 2) == 1.67
    # "0.2 to 3.0% in every other condition"
    others = [round(100 * d["configs"][c]["leaked_rate"], 1)
              for c in ("base_no_wrapper", "base_bodhi", "lora_no_wrapper")]
    assert others == [0.2, 3.0, 1.8]
    assert [q["n"] for q in d["length_control_clean"]] == [105, 105, 105, 106]
    # The seven Pass-1 headers alone (audit NUM-4): nearly the same split.
    assert round(100 * lb["leaked_rate_pass1_headers"], 1) == 53.2
    p1 = d["decisive_pass1_headers"]
    assert (p1["n_leaked"], p1["n_clean"]) == (494, 434)
    assert round(p1["lora_bodhi_leaked_red_flag"], 2) == 1.32
    assert round(p1["lora_bodhi_clean_red_flag"], 2) == 1.73


def test_pass1_rule_ignores_the_two_extra_markers():
    im = _load()
    soap = "DIFFERENTIAL DIAGNOSIS: flu vs cold. SCOPE OF this advice: general."
    assert im.is_leaked(soap)
    assert not im.is_leaked(soap, patterns=im.PASS1_RE)
