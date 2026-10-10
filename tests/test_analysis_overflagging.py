"""The over-flagging check (commitment C30) and what it rests on: the evaluator's red-flag
rate, the red-flag-at-stake label of the HealthBench record, and the label rates of the
discrimination section, on a synthetic table. No rule over the reply text is involved."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.cell import analyze_tables  # noqa: E402
from bodhi_lora.data.labels import label_value  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from tests.test_analysis_cell import PROMPTS, synthetic_prompts, synthetic_table  # noqa: E402

# Any HealthBench cell: the label lives in the benchmark record.
CELL = "healthbench__mistral-small-24b__bodhi__2026-07"


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


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
    table["red_flag_rate"] = 0.0
    table.loc[table["prompt_id"].isin(PROMPTS[:3]), "red_flag_rate"] = 1.0
    # The evaluator gives lora a red-flag score of 2 on p3 in every seed, and on p4 too
    # in seed 2.
    lora = table["condition"] == "lora"
    table.loc[lora & (table["prompt_id"] == "p3"), "red_flag_rate"] = 1.0
    table.loc[lora & (table["seed"] == 2) & (table["prompt_id"] == "p4"), "red_flag_rate"] = 1.0
    meta = [{"n_examples": {c: len(PROMPTS) for c in cell.eval.conditions}} for _ in seeds]
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(CELL)
    out = analyze_tables(reg, cell, table, prompts, meta, paths, report_config())

    res = out["discrimination"]["red_flag_rate"]["red-flag-at-stake"]
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
    # Both responses are the evaluator's grades, so a reply without a grade leaves the rate.
    assert set(out["discrimination"]) == {"active_inquiry", "red_flag_rate"}
    assert np.isfinite(out["contrasts"]["lora - base"]["metrics"]["red_flag_rate"]["diff"])


def test_an_unscored_reply_leaves_the_red_flag_rate(reg, tmp_path):
    cell = reg.cells[CELL]
    table = synthetic_table(cell, (1, 2))
    table["red_flag_rate"] = 1.0
    prompts = synthetic_prompts()
    prompts["label_red-flag-at-stake"] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    # One negative-group reply of lora has no grade; it must not count as a 0.
    miss = (table["condition"] == "lora") & (table["seed"] == 1) & (table["prompt_id"] == "p5")
    table.loc[miss, "epi_status"] = "parse_failure"
    table.loc[miss, "red_flag_rate"] = np.nan
    meta = [{"n_examples": {c: len(PROMPTS) for c in cell.eval.conditions}} for _ in (1, 2)]
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(CELL)
    out = analyze_tables(reg, cell, table, prompts, meta, paths, report_config())
    lo = out["discrimination"]["red_flag_rate"]["red-flag-at-stake"]["conditions"]["lora"]
    assert lo["negative_rate"] == 1.0 and lo["n_negative"] == 5
