"""The per-cell analysis on a synthetic response table: contrasts come from the cell's
conditions by rule, the seed is the unit of every test, the Holm family is the seven
dimensions of one contrast, and a statistic without its input says "not measured"."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.cell import EPISTEMIC_KIND, SEVEN, analyze_tables, contrasts_of  # noqa: E402
from bodhi_lora.analysis.extract import EPISTEMIC, KEY, RESPONSE, RUBRIC  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

SMOKE = "tiny-rubric__tiny-random-llama__bodhi"
PROMPTS = [f"p{i}" for i in range(6)]
TEXT = sorted(report_config().text_features)


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


def synthetic_table(cell, seeds=(1, 2, 3)):
    """Every condition answers every prompt in every seed. Active inquiry rises by
    condition and seed in a known way; the other dimensions are noise from a fixed seed."""
    rng = np.random.default_rng(0)
    rows = []
    for s in seeds:
        for ci, c in enumerate(cell.eval.conditions):
            for pi, p in enumerate(PROMPTS):
                row = {"cell": cell.id, "seed": s, "condition": c, "prompt_id": p,
                       "chars": 100.0 + 10 * ci + pi, "tokens": np.nan, "tokens_engine": np.nan,
                       "calls": np.nan, "tokens_total": np.nan, "finish_length": 0.0,
                       "rubric_score": 0.5 + 0.01 * ci, "rubric_score_worst": 0.5,
                       "rubric_items": 2.0, "rubric_parse_failures": 0.0, "neg_items_met": 0.0,
                       "neg_points_lost": 0.0, "epi_status": "scored", **{t: 0.0 for t in TEXT}}
                # 0-2 grades, 0/1 rates and question counts, as the evaluator gives them.
                row.update({k: float(rng.integers(0, 2 if EPISTEMIC_KIND.get(k) == "rate" else 3))
                            for k in EPISTEMIC})
                # Asks on prompts p0..p(k-1), with k growing with the condition and the seed.
                row["active_inquiry"] = float(pi < min(ci + s, len(PROMPTS)))
                rows.append(row)
    df = pd.DataFrame(rows)
    return df[KEY + RESPONSE + RUBRIC + ["epi_status", *EPISTEMIC, *TEXT]]


def synthetic_prompts():
    return pd.DataFrame({"prompt_id": PROMPTS, "in_population": 1.0, "evaluated": 1.0, "theme": None,
                         "rubric_points_pos": 5.0})


def run(reg, table, tmp_path):
    cell = reg.cells[SMOKE]
    meta = [{"n_examples": {c: len(PROMPTS) for c in cell.eval.conditions}} for _ in table["seed"].unique()]
    # An empty results root: the sections that read files (filter, recipe, validation)
    # find nothing.
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(SMOKE)
    return analyze_tables(reg, cell, table, synthetic_prompts(), meta, paths, report_config())


def test_contrasts_come_from_the_conditions_by_rule(reg):
    cell = reg.cells[SMOKE]
    conds = cell.eval.conditions
    pairs = contrasts_of(cell, reg)
    own = f"base+{cell.protocol}"
    baselines = [c for c in conds if c.startswith("base+") and c != own]
    assert {own, f"lora+{cell.protocol}", "lora"} <= set(conds) and baselines
    # every condition minus base; lora minus base+<protocol>; lora+<protocol> minus lora;
    # lora minus each other base+<prompt protocol>; nothing else, and no pair twice.
    expected = ([(c, "base") for c in conds if c != "base"] + [("lora", own), (f"lora+{cell.protocol}", "lora")]
                + [("lora", c) for c in baselines])
    assert sorted(pairs) == sorted(expected) and len(pairs) == len(set(pairs))


def test_seed_level_test_and_holm_family(reg, tmp_path):
    table = synthetic_table(reg.cells[SMOKE])
    out = run(reg, table, tmp_path)
    seed_means = table.groupby(["condition", "seed"])["active_inquiry"].mean()
    a, b = seed_means["lora"].to_numpy(), seed_means["base"].to_numpy()
    r = out["contrasts"]["lora - base"]["metrics"]["active_inquiry"]
    assert r["df"] == 2 and r["diff"] == pytest.approx((a - b).mean(), rel=1e-5)
    # Holm over the seven dimensions of the contrast; a dimension with no seed spread has
    # no p and stays out of the family.
    m = out["contrasts"]["lora - base"]["metrics"]
    tested = [k for k in SEVEN if m[k]["p"] is not None and np.isfinite(m[k]["p"])]
    assert len(tested) >= 2
    expected = multipletests([m[k]["p"] for k in tested], method="holm")[1]
    assert [m[k]["p_holm_within_contrast"] for k in tested] == pytest.approx(list(expected), rel=1e-5)
    # A dimension with spread across seeds gets scipy's paired t.
    for k in SEVEN:
        sm = table.groupby(["condition", "seed"])[k].mean()
        d = sm["lora+bodhi"].to_numpy() - sm["lora"].to_numpy()
        if np.ptp(d) > 0:
            res = stats.ttest_rel(sm["lora+bodhi"].to_numpy(), sm["lora"].to_numpy())
            assert out["contrasts"]["lora+bodhi - lora"]["metrics"][k]["p"] == pytest.approx(res.pvalue, rel=1e-5)
            break
    else:
        pytest.fail("no dimension with spread in the synthetic table")


def test_a_seed_without_a_value_leaves_the_seed_level_statistics(reg, tmp_path):
    """docs/ANALYSIS_PLAN.md 2.3: the seeds that remain carry the statistic, the degrees
    of freedom fall, and the output names the missing seed."""
    table = synthetic_table(reg.cells[SMOKE])
    # Seed 2's lora replies all fail their dimension grade.
    hit = (table["seed"] == 2) & (table["condition"] == "lora")
    table.loc[hit, "epi_status"] = "parse_failure"
    table.loc[hit, list(EPISTEMIC)] = np.nan
    out = run(reg, table, tmp_path)
    summary = out["conditions"]["active_inquiry"]["lora"]
    assert summary["per_seed"]["2"] is None and summary["seeds_missing"] == [2] and summary["n_seeds"] == 2
    seed_means = table[table["condition"] == "lora"].groupby("seed")["active_inquiry"].mean()
    assert summary["seed_mean"] == pytest.approx(seed_means.loc[[1, 3]].mean())
    r = out["contrasts"]["lora - base"]["metrics"]["active_inquiry"]
    assert r["n_seeds"] == 2 and r["df"] == 1 and r["seeds_missing"] == [2]
    # The worst case counts the failed grades at the undesirable end instead.
    worst = out["worst_case"]["active_inquiry"]["contrasts"]["lora - base"]
    assert worst["df"] == 2
    # The rubric score does not depend on the dimension grade.
    assert out["contrasts"]["lora - base"]["metrics"]["rubric_score"]["df"] == 2


def test_a_condition_with_no_seed_left_is_not_measured(reg, tmp_path):
    table = synthetic_table(reg.cells[SMOKE])
    hit = table["condition"] == "lora"
    table.loc[hit, "epi_status"] = "parse_failure"
    table.loc[hit, list(EPISTEMIC)] = np.nan
    out = run(reg, table, tmp_path)
    assert "not_measured" in out["conditions"]["active_inquiry"]["lora"]
    assert out["contrasts"]["lora - base"]["metrics"]["active_inquiry"]["n_seeds"] == 0


def test_absent_inputs_say_why(reg, tmp_path):
    out = run(reg, synthetic_table(reg.cells[SMOKE]), tmp_path)
    assert "not_measured" in out["themes"]
    assert "not_measured" in out["recipe"]
    assert "not_measured" in out["validation"]
    for resp, labels in out["discrimination"].items():
        for lab, res in labels.items():
            assert "not_measured" in res, (resp, lab)


def test_analyze_takes_cells_or_all():
    from bodhi_lora import cli

    with pytest.raises(SystemExit, match="--all"):
        cli.main(["analyze"])
    with pytest.raises(SystemExit, match="no cells"):
        cli.main(["analyze", "no-such__cell__id"])


def test_extract_runs_through_the_command_line():
    """analyze --extract reaches extract_cell. Importing the submodule
    bodhi_lora.analysis.extract once hid a package function of the same name, and
    make extract stopped with a TypeError; a cell without outputs must instead stop
    with the reason."""
    from bodhi_lora import cli

    with pytest.raises(SystemExit, match="no eval outputs"):
        cli.main(["analyze", "--extract", SMOKE])
