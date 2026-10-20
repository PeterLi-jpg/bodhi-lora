"""The decision rules of docs/ANALYSIS_PLAN.md (bodhi_lora/analysis/hypotheses.py) on
small hand-made stats, one case per branch of each rule, and on the committed cells."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("statsmodels")

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.hypotheses import asking, baselines, context_pair, hypotheses_summary  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

ALPHA = 0.05


def seed_test(diff, lo, hi, p, p_holm=None):
    r = {"diff": diff, "ci95": [lo, hi], "p": p, "n_seeds": 5}
    if p_holm is not None:
        r["p_holm_within_contrast"] = p_holm
    return r


UP = seed_test(0.2, 0.1, 0.3, 0.001, 0.004)
FLAT = seed_test(0.01, -0.05, 0.07, 0.6, 0.9)
DOWN = seed_test(-0.2, -0.3, -0.1, 0.001, 0.004)
# Clears 0 before Holm but not after it.
UP_BEFORE_HOLM = seed_test(0.05, 0.001, 0.099, 0.04, 0.12)


def cell(contrasts: dict, **extra) -> dict:
    return {"contrasts": {k: {"metrics": m} for k, m in contrasts.items()}, "subsets": {}, **extra}


@pytest.mark.parametrize("ai, cs, decision", [
    (UP, UP, "supported"),
    (UP, FLAT, "partial"),
    (UP, UP_BEFORE_HOLM, "partial"),
    (FLAT, FLAT, "not supported"),
    (UP, DOWN, "not supported"),
])
def test_asking_follows_h1s_rule(ai, cs, decision):
    stats = cell({"lora - base": {"active_inquiry": ai, "context_seeking": cs}})
    assert asking(stats, ALPHA)["decision"] == decision


def test_asking_reads_a_subset_when_named():
    stats = cell({"lora - base": {"active_inquiry": FLAT, "context_seeking": FLAT}})
    stats["subsets"]["clinician-user=positive"] = {
        "contrasts": {"lora - base": {"metrics": {"active_inquiry": UP, "context_seeking": UP}}}}
    assert asking(stats, ALPHA, subset="clinician-user=positive")["decision"] == "supported"
    assert asking(stats, ALPHA)["decision"] == "not supported"


def test_a_missing_primary_outcome_is_not_decided():
    assert asking(cell({"lora - base": {"active_inquiry": UP}}), ALPHA)["decision"] is None


def protocols():
    return {"clarify": SimpleNamespace(passes=1), "mediq-abstain": SimpleNamespace(passes=2)}


@pytest.mark.parametrize("ai, cs, rubric, decision, narrows", [
    (UP, UP, seed_test(-0.02, -0.03, -0.01, 0.01), "adapter ahead", False),
    (DOWN, FLAT, seed_test(-0.02, -0.03, -0.01, 0.01), "baseline ahead", True),
    (FLAT, FLAT, seed_test(-0.02, -0.03, -0.01, 0.01), "not separated", True),
    (DOWN, FLAT, seed_test(0.02, 0.01, 0.03, 0.01), "baseline ahead", False),
])
def test_baselines_and_the_narrowing_clause(ai, cs, rubric, decision, narrows):
    stats = cell({"lora - base+clarify": {"active_inquiry": ai, "context_seeking": cs, "rubric_score": rubric}})
    r = baselines(stats, ALPHA, protocols(), ["base+clarify"])["baselines"]["base+clarify"]
    assert (r["decision"], r["narrows"]) == (decision, narrows)


def test_a_two_call_baseline_never_narrows_the_claim():
    stats = cell({"lora - base+mediq-abstain": {"active_inquiry": DOWN, "context_seeking": FLAT,
                                               "rubric_score": seed_test(-0.02, -0.03, -0.01, 0.01)}})
    r = baselines(stats, ALPHA, protocols(), ["base+mediq-abstain"])["baselines"]["base+mediq-abstain"]
    assert r["decision"] == "baseline ahead" and r["one_call"] is False and r["narrows"] is False


def summary(rule: str, stats_by_cell: dict, **spec):
    h = SimpleNamespace(id="HX", token="HX", rule=rule, cells=list(stats_by_cell), subset=None,
                        baselines=[], reference=None, **spec)
    cfg = SimpleNamespace(hypotheses=[h], alpha=ALPHA)
    reg = SimpleNamespace(cells={c: SimpleNamespace(protocol="bodhi") for c in stats_by_cell}, protocols={})
    return hypotheses_summary(stats_by_cell, cfg, reg)["HX"]["cells"]


def test_cost_runs_holm_across_the_questions_cells():
    # p = 0.03 alone lowers the score; with a second cell at p = 0.04 in the family, Holm
    # gives both 0.06.
    a = cell({"lora - base": {"rubric_score": seed_test(-0.02, -0.04, -0.002, 0.03)}})
    b = cell({"lora - base": {"rubric_score": seed_test(-0.02, -0.04, -0.001, 0.04)}})
    assert summary("cost", {"a": a})["a"]["decision"] == "lowers the score"
    both = summary("cost", {"a": a, "b": b})
    assert [both[c]["decision"] for c in "ab"] == ["no change detected"] * 2
    assert both["a"]["outcomes"]["rubric_score"]["p_adj"] == pytest.approx(0.06)


def paired(did, complete, lora_delta_ci):
    return {"contrasts": {}, "subsets": {},
            "paired_probe": {"active_inquiry": {"incomplete": {
                "conditions": {"lora": {"delta": {"seed_mean": 0.1, "seed_ci95_t": lora_delta_ci}},
                               "base": {"delta": {"seed_mean": 0.2, "seed_ci95_t": [0.1, 0.3]}}},
                "contrasts": {"lora - base": {"did": did}}}}},
            "discrimination": {"active_inquiry": {"incomplete": {"contrasts": {"lora - base": {
                "negative_rate": {"seed_test": complete}}}}}}}


@pytest.mark.parametrize("did, complete, lora_ci, decision", [
    (seed_test(0.2, 0.1, 0.3, 0.001), FLAT, [0.1, 0.3], "calibration supported"),
    (seed_test(0.2, 0.1, 0.3, 0.001), FLAT, [-0.1, 0.3], "not resolved"),
    (seed_test(-0.2, -0.3, -0.1, 0.001), UP, [-0.1, 0.1], "mimicry indicated"),
    (seed_test(0.01, -0.1, 0.1, 0.8), UP, [-0.1, 0.1], "mimicry indicated"),
    (seed_test(0.01, -0.1, 0.1, 0.8), FLAT, [-0.1, 0.1], "not resolved"),
])
def test_paired_probe_follows_section_5_1(did, complete, lora_ci, decision):
    assert summary("paired_probe", {"a": paired(did, complete, lora_ci)})["a"]["decision"] == decision


def red_flags(drop, primary):
    return cell({"lora+bodhi - lora": {"red_flag": drop}, "lora+red-flags-first - lora+bodhi": {"red_flag": primary}})


@pytest.mark.parametrize("drop, primary, decision", [
    (FLAT, UP, "no drop to recover"),
    (UP, UP, "no drop to recover"),
    (DOWN, UP, "recovered"),
    (DOWN, FLAT, "not recovered"),
])
def test_red_flags_first_needs_a_drop_to_recover(drop, primary, decision):
    assert summary("red_flags_first", {"a": red_flags(drop, primary)})["a"]["decision"] == decision


def test_over_flagging_reads_the_non_emergent_rate():
    stats = {"contrasts": {}, "subsets": {}, "discrimination": {"red_flag_rate": {"emergent-physician": {
        "contrasts": {"lora - base": {"negative_rate": {"seed_test": UP}, "positive_rate": {"seed_test": FLAT}}}}}}}
    r = summary("over_flagging", {"a": stats})["a"]
    assert r["decision"] == "over-flagging found"
    stats["discrimination"]["red_flag_rate"]["emergent-physician"]["contrasts"]["lora - base"]["negative_rate"] = \
        {"seed_test": FLAT}
    assert summary("over_flagging", {"a": stats})["a"]["decision"] == "not detected"


def test_committed_decisions_are_what_the_rules_give():
    """cross_cell/stats.json holds the decisions of the committed cells' stats."""
    reg, cfg, proj = load_registry(CONFIGS_DIR), report_config(), project(CONFIGS_DIR)
    path = proj.results_root / "cross_cell" / "stats.json"
    if not path.is_file():
        pytest.skip("the cross-cell statistics are not in this checkout")
    # A cell that has not run yet has no stats; the rules list it as not measured, in both.
    cells = {c for h in cfg.hypotheses for c in [*h.cells, h.reference] if c}
    all_stats = {c: json.loads(p.read_text()) for c in cells
                 if (p := proj.cell(c).analysis() / "stats.json").is_file()}
    fresh = json.loads(json.dumps(hypotheses_summary(all_stats, cfg, reg)))
    committed = json.loads(path.read_text())["hypotheses"]
    assert {h: {c: r.get("decision") for c, r in v["cells"].items()} for h, v in fresh.items()} == \
        {h: {c: r.get("decision") for c, r in v["cells"].items()} for h, v in committed.items()}


def ctx_cell(drop, primary=None):
    m = {"lora+bodhi - lora": {"red_flag": drop}}
    if primary:
        m["lora+red-flags-first - lora+bodhi"] = {"red_flag": primary}
    return cell(m)


@pytest.mark.parametrize("short, full, decision", [
    (DOWN, FLAT, "context implicated"),
    (DOWN, UP, "context implicated"),
    (FLAT, DOWN, "context not implicated"),
    (UP, FLAT, "context not implicated"),
    (DOWN, DOWN, "not resolved"),
])
def test_the_4096_pair_follows_the_2026_10_19_entry(short, full, decision):
    r = context_pair(ctx_cell(short, FLAT), ctx_cell(full), "bodhi")
    assert r["decision"] == decision
    assert r["outcomes"]["red_flag"]["diff"] == FLAT["diff"]


def test_the_4096_pair_waits_for_its_reference_only_when_there_is_a_drop():
    assert context_pair(ctx_cell(FLAT), None, "bodhi")["decision"] == "context not implicated"
    assert context_pair(ctx_cell(DOWN), None, "bodhi")["decision"] is None
