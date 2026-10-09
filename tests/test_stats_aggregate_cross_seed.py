"""Golden test for scripts/aggregate_cross_seed.py on the submitted cell's committed results.

Pins the cross-seed means the paper's Tables 1 and 2 quote, and the pooled n the
CoT conditions actually have (PROV-4: not 985-998).
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import aggregate_cross_seed as acs  # noqa: E402


@pytest.fixture(scope="module")
def agg():
    s4 = acs.aggregate_stage4({s: acs.load_per_seed_stage4(s) for s in acs.SEEDS})
    s5 = acs.aggregate_stage5({s: acs.load_per_seed_stage5(s) for s in acs.SEEDS})
    return s4, s5


def test_healthbench_means(agg):
    s4, _ = agg
    got = {c: round(s4[c]["score_mean_across_seeds"], 3) for c in acs.STAGE4_CONFIGS}
    assert got == {"base_no_wrapper": 0.459, "base_bodhi": 0.463,
                   "lora_no_wrapper": 0.463, "lora_bodhi": 0.455}


def test_headline_epistemic_means(agg):
    _, s5 = agg
    base, lora = s5["base_no_wrapper"], s5["lora_no_wrapper"]
    assert round(100 * base["active_inquiry_rate"]["mean"], 1) == 17.5
    assert round(100 * lora["active_inquiry_rate"]["mean"], 1) == 45.6
    assert round(base["context_seeking_mean"]["mean"], 2) == 1.40
    assert round(lora["context_seeking_mean"]["mean"], 2) == 1.75
    assert round(base["red_flag_identification_mean"]["mean"], 2) == 1.52
    assert round(lora["red_flag_identification_mean"]["mean"], 2) == 1.67
    assert round(base["scope_bounding_mean"]["mean"], 2) == 1.91
    assert round(lora["scope_bounding_mean"]["mean"], 2) == 1.97
    assert round(100 * s5["lora_bodhi"]["active_inquiry_rate"]["mean"], 1) == 80.0


def test_pooled_counts(agg):
    s4, s5 = agg
    assert [s4[c]["n_results_pooled"] for c in acs.STAGE4_CONFIGS] == [1000, 957, 1000, 959]
    assert [s5[c]["n_scored_pooled"] for c in acs.STAGE4_CONFIGS] == [998, 947, 985, 928]
    assert [s5[c]["n_parse_failures_pooled"] for c in acs.STAGE4_CONFIGS] == [2, 9, 14, 14]


def test_committed_aggregate_matches_a_fresh_run(agg):
    """The cell's analysis/cross_seed_aggregate.json must be what the script writes now."""
    s4, s5 = agg
    committed = json.loads((acs.OUT_DIR / "cross_seed_aggregate.json").read_text())
    fresh = json.loads(json.dumps({"seeds": list(acs.SEEDS), "stage4_healthbench": s4,
                                   "stage5_epistemic": s5}))
    # Floats compared to 1e-12: Python versions differ in the last bit of stdev.
    assert _close(committed, fresh)


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return b == pytest.approx(a, rel=1e-12, abs=1e-15)
    return a == b
