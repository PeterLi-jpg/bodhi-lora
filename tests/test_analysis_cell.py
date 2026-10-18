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
                row = {"cell": cell.id, "seed": s, "condition": c, "prompt_id": p, "pair_id": np.nan,
                       "chars": 100.0 + 10 * ci + pi, "zlib_ratio": 0.5, "tokens": np.nan, "tokens_engine": np.nan,
                       "calls": np.nan, "tokens_total": np.nan, "finish_length": 0.0,
                       "rubric_score": 0.5 + 0.01 * ci, "rubric_score_worst": 0.5,
                       "rubric_items": 2.0, "rubric_parse_failures": 0.0, "neg_items_met": 0.0,
                       "neg_points_lost": 0.0, "epi_status": "scored"}
                # 0-2 grades, 0/1 rates and question counts, as the evaluator gives them.
                row.update({k: float(rng.integers(0, 2 if EPISTEMIC_KIND.get(k) == "rate" else 3))
                            for k in EPISTEMIC})
                # Asks on prompts p0..p(k-1), with k growing with the condition and the seed.
                row["active_inquiry"] = float(pi < min(ci + s, len(PROMPTS)))
                rows.append(row)
    df = pd.DataFrame(rows)
    return df[KEY + RESPONSE + RUBRIC + ["epi_status", *EPISTEMIC]]


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
    # Then the record's own contrasts, which the rule does not make (H6 here).
    declared = [tuple(x.strip() for x in k.split(" - ")) for k in cell.report.contrasts]
    assert declared == [("lora+red-flags-first", "lora+bodhi")] and not set(declared) & set(expected)
    assert pairs == list(dict.fromkeys(pairs)) and pairs[-len(declared):] == declared
    assert sorted(pairs) == sorted(expected + declared)


def test_a_declared_contrast_must_name_two_conditions_of_the_cell(reg):
    from pydantic import ValidationError

    from bodhi_lora.registry.schema import CellConfig

    data = reg.cells[SMOKE].model_dump()
    for bad in ["lora+red-flags-first - lora+clarify", "lora+bodhi", "lora - lora"]:
        data["report"]["contrasts"] = [bad]
        with pytest.raises(ValidationError, match="report.contrasts"):
            CellConfig.model_validate(data)


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


PAIRED = "mediq-paired__mistral-small-24b__bodhi"


def test_the_paired_probe_gives_delta_and_the_difference_in_differences(reg, tmp_path):
    """docs/ANALYSIS_PLAN.md 5.1: Delta(c, s) is the mean over pairs of (ask on the
    incomplete row minus ask on the complete row); DiD(s) = Delta(lora, s) - Delta(base, s);
    a pair whose other row has no grade leaves that condition and seed."""
    cell = reg.cells[PAIRED]
    seeds, n_pairs = (1, 2, 3), 6
    ids = [(f"q{k}-{v}", f"q{k}", v) for k in range(n_pairs) for v in ("complete", "incomplete")]
    rows = []
    for s in seeds:
        for c in cell.eval.conditions:
            for pid, pair, variant in ids:
                k = int(pair[1:])
                # base asks on the first s pairs whatever is missing; lora also asks on
                # every incomplete row.
                ask = float(k < s or (c == "lora" and variant == "incomplete"))
                row = {"cell": cell.id, "seed": s, "condition": c, "prompt_id": pid, "pair_id": pair,
                       "chars": 100.0, "zlib_ratio": 0.5, "tokens": np.nan, "tokens_engine": np.nan, "calls": np.nan,
                       "tokens_total": np.nan, "finish_length": 0.0, "rubric_score": 0.5,
                       "rubric_score_worst": 0.5, "rubric_items": 1.0, "rubric_parse_failures": 0.0,
                       "neg_items_met": 0.0, "neg_points_lost": 0.0, "epi_status": "scored"}
                row.update({k2: 1.0 for k2 in EPISTEMIC})
                row["active_inquiry"] = ask
                rows.append(row)
    table = pd.DataFrame(rows)[KEY + RESPONSE + RUBRIC + ["epi_status", *EPISTEMIC]]
    # Seed 3: the lora grade of pair q5's complete row failed, so q5 leaves lora in seed 3.
    hit = (table["seed"] == 3) & (table["condition"] == "lora") & (table["prompt_id"] == "q5-complete")
    table.loc[hit, "epi_status"] = "parse_failure"
    table.loc[hit, list(EPISTEMIC)] = np.nan
    prompts = pd.DataFrame({"prompt_id": [i for i, _, _ in ids], "in_population": 1.0, "evaluated": 1.0,
                            "theme": None, "rubric_points_pos": 1.0,
                            "label_incomplete": [float(v == "incomplete") for _, _, v in ids]})
    meta = [{"n_examples": {c: len(ids) for c in cell.eval.conditions}} for _ in seeds]
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(PAIRED)
    out = analyze_tables(reg, cell, table, prompts, meta, paths, report_config())
    probe = out["paired_probe"]["active_inquiry"]["incomplete"]
    base, lora = probe["conditions"]["base"], probe["conditions"]["lora"]
    assert base["delta"]["per_seed"] == {"1": 0.0, "2": 0.0, "3": 0.0}
    # lora asks on every incomplete row: the pairs base did not ask on add 1 each.
    expected = {"1": 5 / 6, "2": 4 / 6, "3": 2 / 5}
    assert lora["delta"]["per_seed"] == pytest.approx(expected)
    assert lora["n_pairs"]["3"] == 5 and lora["n_pairs_dropped"]["3"] == 1 and lora["n_pairs_dropped"]["1"] == 0
    did = probe["contrasts"]["lora - base"]["did"]
    res = stats.ttest_1samp(list(expected.values()), 0.0)
    assert did["diff"] == pytest.approx(np.mean(list(expected.values()))) and did["p"] == pytest.approx(res.pvalue)
    # McNemar, seed 1, lora: 1 pair asks on both rows, 5 on the incomplete row only.
    assert lora["mcnemar"]["1"]["table"] == [[1, 5], [0, 0]]
    assert lora["mcnemar"]["1"]["exact_p"] == pytest.approx(stats.binomtest(0, 5).pvalue)
    # The plain discrimination index of the same label is there too, and a HealthBench
    # cell gets no paired statistics.
    assert "conditions" in out["discrimination"]["active_inquiry"]["incomplete"]
    smoke = run(reg, synthetic_table(reg.cells[SMOKE]), tmp_path)
    assert smoke["paired_probe"] == {}


def test_kappa_matches_sklearn():
    """stats.kappa is written out for the bootstrap; it must give sklearn's number, with
    weights too (the bootstrap passes resample counts as weights)."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    from bodhi_lora.analysis import stats as st

    rng = np.random.default_rng(1)
    for k, weights in ((3, "quadratic"), (2, None)):
        a, b = rng.integers(0, k, 200), rng.integers(0, k, 200)
        b[:120] = a[:120]
        w = rng.integers(0, 4, 200).astype(float)
        conf = np.zeros((k, k))
        np.add.at(conf, (a, b), w)
        expected = sklearn_metrics.cohen_kappa_score(a, b, labels=list(range(k)), weights=weights, sample_weight=w)
        assert st.kappa(conf, quadratic=weights == "quadratic") == pytest.approx(expected, rel=1e-9)


def test_judges_recompute_the_contrasts_and_decide_against_the_evaluator(reg, tmp_path):
    """H8 (docs/ANALYSIS_PLAN.md section 8): each contrast on the seven dimensions again
    from a judge's grades, Holm within the contrast, and confirmed, not confirmed or
    reversed against the evaluator; kappa (quadratic on the 0 to 2 grades) with intervals."""
    base_cell = reg.cells[SMOKE]
    cell = base_cell.model_copy(update={"grading": base_cell.grading.model_copy(
        update={"judges": ["same", "flipped", "absent"]})})
    seeds = (1, 2, 3, 4, 5)
    table = synthetic_table(cell, seeds=seeds)
    # lora asks on 3 or 4 of the 6 prompts by seed, base on 1: a rise with seed spread.
    pi = table["prompt_id"].str[1:].astype(int)
    table.loc[table["condition"] == "base", "active_inquiry"] = (pi == 0).astype(float)
    lora = table["condition"] == "lora"
    table.loc[lora, "active_inquiry"] = (pi < 3 + table["seed"] % 2)[lora].astype(float)
    for jid in ("same", "flipped"):
        j = f"j_{jid}__"
        table[f"{j}rubric_score"] = table["rubric_score"]
        for k in EPISTEMIC:
            table[f"{j}{k}"] = table[k]
        table[f"{j}crit_11"], table[f"{j}crit_10"], table[f"{j}crit_01"], table[f"{j}crit_00"] = 1.0, 0.0, 0.0, 1.0
    table["j_flipped__active_inquiry"] = 1.0 - table["active_inquiry"]
    meta = [{"n_examples": {c: len(PROMPTS) for c in cell.eval.conditions}} for _ in seeds]
    paths = replace(project(CONFIGS_DIR), results_root=tmp_path).cell(SMOKE)
    out = analyze_tables(reg, cell, table, synthetic_prompts(), meta, paths, report_config())
    judges, ev = out["judges"], out["contrasts"]
    assert judges["absent"] == {"not_measured": "no judge file in the cell's grade folders"}

    same = judges["same"]
    assert same["criterion_kappa"] == pytest.approx(1.0) and same["criterion_kappa_ci95"] == pytest.approx([1.0, 1.0])
    for k in SEVEN:
        kap = same["dimension_kappa"][k]
        assert kap["kappa"] == pytest.approx(1.0) and kap["n_off_scale"] == 0
        assert kap["weights"] == ("none" if k == "active_inquiry" else "quadratic")
    for key, res in same["contrasts"].items():
        for k in SEVEN:
            r, e = res["metrics"][k], ev[key]["metrics"][k]
            if "not_measured" in r:
                continue
            assert r["diff"] == pytest.approx(e["diff"])
            p = e["p_holm_within_contrast"]
            assert r["decision"] == ("confirmed" if p is not None and p < 0.05 and e["diff"] != 0 else "not confirmed")
        assert "decision" not in res["metrics"]["rubric_score"]
        assert res["metrics"]["rubric_score"]["diff"] == pytest.approx(ev[key]["metrics"]["rubric_score"]["diff"])

    ai = ev["lora - base"]["metrics"]["active_inquiry"]
    assert ai["diff"] > 0 and ai["p_holm_within_contrast"] < 0.05
    assert same["contrasts"]["lora - base"]["metrics"]["active_inquiry"]["decision"] == "confirmed"
    flipped = judges["flipped"]
    assert flipped["contrasts"]["lora - base"]["metrics"]["active_inquiry"]["decision"] == "reversed"
    assert flipped["dimension_kappa"]["active_inquiry"]["kappa"] < 0


def test_reply_health_counts_loops_and_the_token_limit(reg, tmp_path):
    """A loop: over loop_min_chars and a zlib ratio under loop_max_zlib_ratio; an empty
    reply (no ratio) is not one. The limit count reads finish_length where it is stored."""
    table = synthetic_table(reg.cells[SMOKE])
    lora = table["condition"] == "lora"
    first = table["prompt_id"] == "p0"
    table.loc[lora & first, ["chars", "zlib_ratio"]] = [900.0, 0.05]          # a loop in every seed
    table.loc[lora & (table["prompt_id"] == "p1"), ["chars", "zlib_ratio"]] = [300.0, 0.05]   # too short
    table.loc[lora & (table["prompt_id"] == "p2"), ["chars", "zlib_ratio"]] = [0.0, np.nan]   # empty
    table.loc[lora & (table["prompt_id"] == "p3") & (table["seed"] == 1), "finish_length"] = 1.0
    table.loc[table["condition"] == "base", "finish_length"] = np.nan          # not stored
    rh = run(reg, table, tmp_path)["reply_health"]
    assert rh["measured"] and (rh["loop_min_chars"], rh["loop_max_zlib_ratio"]) == (400, 0.15)
    r = rh["conditions"]["lora"]
    assert (r["n"], r["n_loops"], r["per_seed_loops"]) == (18, 3, {"1": 1, "2": 1, "3": 1})
    assert r["loop_share"] == pytest.approx(3 / 18)
    assert (r["n_limit_known"], r["n_at_limit"]) == (18, 1)
    b = rh["conditions"]["base"]
    assert (b["n_loops"], b["n_limit_known"], b["n_at_limit"], b["limit_share"]) == (0, 0, 0, None)
    # a table from before the column says so
    assert run(reg, table.drop(columns=["zlib_ratio"]), tmp_path)["reply_health"] == {"measured": False}


def test_zlib_ratio_of_prose_and_of_a_loop():
    from bodhi_lora.analysis.extract import zlib_ratio
    prose = ("Chest pain with shortness of breath can have many causes, from a pulled muscle to a "
             "heart problem. Since yours started an hour ago and spreads to your left arm, call "
             "emergency services now. While you wait, sit down and stay calm.")
    loop = "If symptoms worsen, seek urgent care. " * 40
    assert zlib_ratio(prose) > 0.5 and zlib_ratio(loop) < 0.15
    assert np.isnan(zlib_ratio(""))

