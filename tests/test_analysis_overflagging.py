"""The over-flagging check (commitment C30) and the rules it rests on: the escalation
mention rule of configs/report.yaml, the red-flag-at-stake label of the HealthBench
record, and the label rates of the discrimination section, on a synthetic table."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.cell import analyze_tables  # noqa: E402
from bodhi_lora.analysis.extract import TextRule  # noqa: E402
from bodhi_lora.data.labels import label_value  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from tests.test_analysis_cell import PROMPTS, synthetic_prompts, synthetic_table  # noqa: E402

# Any HealthBench cell: the label lives in the benchmark record.
CELL = "healthbench__mistral-small-24b__bodhi__2026-07"


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


@pytest.mark.parametrize("text, hit", [
    ("If the pain spreads to your arm, call 911 right away.", True),
    ("Go to the nearest emergency department.", True),
    ("Please seek immediate medical attention if you faint.", True),
    ("In the UK, call 999.", True),
    ("Head to the ER now.", True),
    ("Guanfacine ER 1 mg at night.", False),
    ("Check the hormone receptor status (ER/PR).", False),
    ("Consider an **urgent care** clinic.", True),
    ("Your glucose of 112 mg/dL is a little high.", False),
    ("Er, that depends on the dose.", False),
    ("Book a visit with your doctor this week.", False),
])
def test_escalation_rule(text, hit):
    rule = TextRule(report_config().text_features["escalation"])
    assert rule(text) == float(hit)


def test_question_rule_needs_a_question_and_mention_rule_does_not():
    cfg = report_config().text_features
    assert TextRule(cfg["second_person_q"])("You should rest.") == 0.0
    assert TextRule(cfg["second_person_q"])("Do you have a fever?") == 1.0
    assert TextRule(cfg["escalation"])("Do you need to call 911?") == 1.0


def test_red_flag_label(reg):
    rule = reg.benchmarks["healthbench"].labels["red-flag-at-stake"]
    crit = lambda text, points=5: {"criterion": text, "points": points}  # noqa: E731
    assert rule.timing == "post_hoc"
    assert label_value(rule, ["theme:emergency_referrals"], []) is True
    assert label_value(rule, ["theme:hedging"], [crit("Advises the user to go to the emergency room.")]) is True
    # A criterion that penalizes needless escalation still marks urgency as at issue, so
    # the negative group stays prompts whose rubric is silent on urgency.
    assert label_value(rule, [], [crit("Recommends the ER, which is not needed here.", -6)]) is True
    assert label_value(rule, [], [crit("Lists warning signs of sepsis.")]) is True
    assert label_value(rule, ["theme:global_health"], [crit("Explains the vaccine schedule.")]) is False
    # The physician-agreed tags do not decide it.
    assert label_value(rule, ["physician_agreed_category:emergent"], [crit("Explains the dose.")]) is False


def test_rates_in_each_label_group(reg, tmp_path):
    cell = reg.cells[CELL]
    seeds = (1, 2, 3)
    table = synthetic_table(cell, seeds)
    prompts = synthetic_prompts()
    # p0..p2 may hold a red flag; p3..p5 hold none.
    prompts["label_red-flag-at-stake"] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    table["escalation"] = 0.0
    table.loc[table["prompt_id"].isin(PROMPTS[:3]), "escalation"] = 1.0
    # lora mentions emergency care on p3 in every seed, and on p4 too in seed 2.
    lora = table["condition"] == "lora"
    table.loc[lora & (table["prompt_id"] == "p3"), "escalation"] = 1.0
    table.loc[lora & (table["seed"] == 2) & (table["prompt_id"] == "p4"), "escalation"] = 1.0
    meta = [{"n_examples": {c: len(PROMPTS) for c in cell.eval.conditions}} for _ in seeds]
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(CELL)
    out = analyze_tables(reg, cell, table, prompts, meta, paths, report_config())

    res = out["discrimination"]["escalation"]["red-flag-at-stake"]
    assert res["exploratory"] and res["n_prompts"] == {"positive": 3, "negative": 3}
    base, lo = res["conditions"]["base"], res["conditions"]["lora"]
    assert base["negative_rate"] == 0.0 and base["positive_rate"] == 1.0
    assert lo["negative_rate"] == pytest.approx(4 / 9)
    assert lo["negative_rate_seed"]["per_seed"] == pytest.approx({"1": 1 / 3, "2": 2 / 3, "3": 1 / 3})
    assert lo["discrimination"] == pytest.approx(1 - 4 / 9)
    lo_lo, lo_hi = lo["negative_rate_ci95"]
    assert lo_lo <= 4 / 9 <= lo_hi
    ctr = res["contrasts"]["lora - base"]["negative_rate"]
    assert ctr["diff"] == pytest.approx(4 / 9)
    assert ctr["seed_test"]["diff"] == pytest.approx(4 / 9) and ctr["seed_test"]["df"] == 2
    # The evaluator's active inquiry counts scored replies; a benchmark label it lacks is
    # not measured with the reason.
    assert set(out["discrimination"]) == {"active_inquiry", "escalation"}
    assert np.isfinite(out["contrasts"]["lora - base"]["metrics"]["escalation"]["diff"])
