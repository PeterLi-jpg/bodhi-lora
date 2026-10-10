"""Tests for scripts/paper_stats.py.

The golden tests run the generator on the committed results of the submitted cell and
pin the values the paper reports, as they actually compute. Where the paper's printed value
differs, the test pins the computed one and the comment says what the paper had.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import paper_stats as ps  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
RESULTS = ROOT / "results"
# The submitted cell and the 2026-04 pilot (results/PROVENANCE.md).
M = RESULTS / "healthbench__medgemma-27b-text__bodhi__2026-05"
PILOT = RESULTS / "healthbench__medgemma-27b-text__bodhi__2026-04"
CROSS = RESULTS / "cross_cell"
POSTED = ROOT / "rebuttal" / "posted_values.json"


def _run(out_dir, posted=None):
    # Optional inputs off, except the committed kappa and rebuttal outputs, so the
    # test does not depend on gitignored data.
    return ps.run(M, out_dir, None, None, M / "validation" / "kappa_results.json",
                  M / "analysis", posted, cross_cell=CROSS)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("generated")
    return _run(out_dir), out_dir


def _macro(out_dir, name):
    for line in (out_dir / "numbers.tex").read_text().splitlines():
        if line.startswith(f"\\newcommand{{\\stat{name}}}{{"):
            body = line.split("}{", 1)[1].rsplit("} %", 1)[0]
            return body.replace("\\ensuremath{", "").rstrip("}")
    raise KeyError(name)


# ---------------------------------------------------------------- golden values

def test_headline_means(generated):
    out, d = generated
    c = out["conditions"]
    assert round(100 * c["active_inquiry"]["base_no_wrapper"]["seed_mean"], 1) == 17.5
    assert round(100 * c["active_inquiry"]["lora_no_wrapper"]["seed_mean"], 1) == 45.6
    assert round(100 * c["active_inquiry"]["lora_bodhi"]["seed_mean"], 1) == 80.0
    assert round(c["context_seeking"]["base_no_wrapper"]["seed_mean"], 2) == 1.40
    assert round(c["context_seeking"]["lora_no_wrapper"]["seed_mean"], 2) == 1.75
    assert round(c["red_flag"]["base_no_wrapper"]["seed_mean"], 2) == 1.52
    assert round(c["red_flag"]["lora_no_wrapper"]["seed_mean"], 2) == 1.67
    assert round(c["scope_bounding"]["lora_no_wrapper"]["seed_mean"], 2) == 1.97
    assert _macro(d, "ActiveInquiryBase") == "17.5"
    assert _macro(d, "ActiveInquiryLoRA") == "45.6"


def test_pooled_d_matches_paper(generated):
    m = generated[0]["contrasts"]["LoRAvsBase"]["metrics"]
    assert [round(m[k]["pooled_d"], 2) for k in
            ("active_inquiry", "context_seeking", "red_flag", "scope_bounding")] == [0.63, 0.44, 0.20, 0.22]
    # Paper put d=0.22 on the scope-bounded RATE; the rate's d is 0.23 (PROV-9b).
    assert round(m["scope_bounded_rate"]["pooled_d"], 2) == 0.23
    assert round(generated[0]["contrasts"]["LoRACoTvsLoRA"]["metrics"]["red_flag"]["pooled_d"], 2) == -0.25


def test_seed_level_tests(generated):
    m = generated[0]["contrasts"]["LoRAvsBase"]["metrics"]
    assert [round(m[k]["t"], 2) for k in
            ("active_inquiry", "context_seeking", "red_flag", "scope_bounding")] == [15.14, 14.78, 18.44, 7.59]
    assert m["uncertainty_ack"]["p"] > 0.9
    # Seed-level LoRA - Base aggregate: +0.004, 95% CI [-0.011, +0.020] (EVAL-1).
    hb = m["healthbench"]
    assert round(hb["diff"], 4) == 0.0042
    assert [round(x, 3) for x in hb["ci95"]] == [-0.011, 0.020]
    # The pooled Welch p the paper quoted as p=0.57 (descriptive only).
    assert round(hb["pooled_welch_p"], 2) == 0.57
    # Interference: seed-level p 0.012, and Holm over the 7 dimensions gives 0.058,
    # not the pooled p < 1e-7 in the paper.
    rf = generated[0]["contrasts"]["LoRACoTvsLoRA"]["metrics"]["red_flag"]
    assert round(rf["p"], 3) == 0.012
    assert round(rf["p_holm_within_contrast"], 3) == 0.058


def test_holm_family_is_seven_dimensions(generated):
    m = generated[0]["contrasts"]["LoRAvsBase"]["metrics"]
    ps_raw = [m[k]["p"] for k in ps.SEVEN_DIMS]
    order = np.argsort(ps_raw)
    # Holm step-down by hand.
    adj, running = [0.0] * 7, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (7 - rank) * ps_raw[i]))
        adj[i] = running
    assert [m[k]["p_holm_within_contrast"] for k in ps.SEVEN_DIMS] == pytest.approx(adj)
    assert "p_holm_within_contrast" not in m["healthbench"]


def test_table_1_uses_t_intervals(generated):
    c = generated[0]["conditions"]["healthbench"]
    # EVAL-11: the paper's LoRA+BODHI row had the z interval [0.453, 0.457].
    assert [round(x, 3) for x in c["lora_bodhi"]["seed_ci95_t"]] == [0.452, 0.458]
    assert [round(x, 3) for x in c["base_no_wrapper"]["seed_ci95_t"]] == [0.445, 0.473]
    tab = (generated[1] / "tab_main_healthbench.tex").read_text()
    assert "[0.452,\\;0.458]" in tab


def test_counts_and_overlap(generated):
    cnt = generated[0]["counts"]
    pc = cnt["per_condition"]
    assert [pc[c]["completions"] for c in ps.CONDITIONS] == [1000, 957, 1000, 959]
    assert [pc[c]["epistemic_scored"] for c in ps.CONDITIONS] == [998, 947, 985, 928]
    assert [pc[c]["epistemic_parse_failures"] for c in ps.CONDITIONS] == [2, 9, 14, 14]
    assert [pc[c]["epistemic_missing"] for c in ps.CONDITIONS] == [0, 1, 1, 17]
    assert cnt["unique_prompts"] == 667
    assert cnt["prompts_in_more_than_one_seed"] == 271
    assert cnt["prompt_seed_multiplicity"] == {"1": 396, "2": 215, "3": 50, "4": 6}
    # EVAL-9: the paper's "<0.1%" holds for the HealthBench grader only.
    assert round(100 * max(pc[c]["epistemic_parse_failure_rate"] for c in pc), 2) == 1.49
    assert round(100 * max(pc[c]["hb_item_parse_failure_rate_max_run"] for c in pc), 3) == 0.087


def test_blanket_disclaimers_fisher(generated):
    bd = generated[0]["blanket_disclaimers"]
    assert (bd["base_no_wrapper"]["count"], bd["base_no_wrapper"]["n"]) == (5, 998)
    assert (bd["lora_no_wrapper"]["count"], bd["lora_no_wrapper"]["n"]) == (0, 985)
    assert round(bd["lora_no_wrapper"]["fisher_p_vs_base"], 3) == 0.062
    # 4 of the 5 flagged disclaimers (all in Base) still get hedging quality 2.
    assert bd["all_conditions"] == {"flagged": 5, "flagged_hedging_2": 4}


def test_worst_case_keeps_headline_effects(generated):
    wc = generated[0]["worst_case"]
    assert round(100 * wc["active_inquiry"]["contrasts"]["LoRAvsBase"]["diff"], 1) == 27.4
    assert wc["active_inquiry"]["contrasts"]["LoRAvsBase"]["p"] < 0.001
    # Failed grades counted as disclaimers: LoRA no longer has fewer than Base.
    assert wc["blanket_disclaimer"]["contrasts"]["LoRAvsBase"]["diff"] > 0


def test_length_quartiles(generated):
    lq = generated[0]["length_quartiles"]
    lora = lq["conditions"]["lora_no_wrapper"]
    # PROV-9c: LoRA cells are not ~245 each.
    assert [q["n"] for q in lora["quartiles"]] == [152, 330, 222, 281]
    assert round(100 * lora["quartiles"][0]["rate"], 1) == 35.5
    assert round(100 * lq["conditions"]["lora_bodhi"]["overall"], 1) == 80.1


def test_response_size_and_correlations(generated):
    rs = generated[0]["response_size"]
    assert round(rs["base_no_wrapper"]["chars_mean"]) == 3224
    assert round(rs["lora_no_wrapper"]["chars_mean"]) == 3407
    sc = generated[0]["shift_correlations"]
    assert sc["n_pairs"] == 983
    assert {k: round(v, 2) for k, v in sc["pearson_r_with_active_inquiry"].items()} == {
        "context_seeking": 0.23, "red_flag": -0.08, "scope_bounding": 0.02, "specificity": -0.02}


def test_context_seeking_anchor(generated):
    # Golden counts from a direct pass over the cell's seed_*/grade/epistemic__*.json:
    # grades of 2 on context-seeking, and how many of them carry no direct question.
    cs = generated[0]["context_seeking_anchor"]
    assert {c: (v["n_score_2"], v["n_score_2_no_question"]) for c, v in cs.items()} == {
        "base_no_wrapper": (684, 509), "base_bodhi": (829, 375),
        "lora_no_wrapper": (850, 401), "lora_bodhi": (880, 137)}
    assert _macro(generated[1], "ContextSeekingTwoNoQuestionPctBase") == "74.4"


def test_axis_scores(generated):
    ax = generated[0]["axis_scores"]
    assert round(ax["accuracy"]["base_no_wrapper"], 3) == 0.117
    assert round(ax["instruction_following"]["lora_no_wrapper"], 3) == 0.286
    assert round(ax["communication_quality"]["lora_bodhi"], 3) == 0.246


def test_kappa_and_leakage_macros(generated):
    d = generated[1]
    assert _macro(d, "KappaAll") == "0.35"
    assert _macro(d, "KappaAllN") == "509"
    # The pre-registered comparison was against the Qwen filter grader (10 filter traces).
    assert _macro(d, "KappaFilter") == "0.30"
    assert _macro(d, "KappaFilterN") == "125"
    # Leakage by the seven Pass 1 headers of the protocol record, from the uniform analysis.
    assert _macro(d, "HBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhi") == "53.2"
    assert _macro(d, "HBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhiN") == "494"


@pytest.mark.parametrize("name", ["LeakagePctLoRACoT", "LeakagePassOnePctLoRACoT", "AskOddsRatioPrimaryLoRA",
                                  "DiscriminationPrimaryLoRAvsBase", "SecondPersonPctLoRA",
                                  "ClinicianKeywordPrompts"])
def test_no_macro_comes_from_a_keyword_analysis(generated, name):
    """The nine-marker leakage, the rubric-keyword labelling, the second-person count and the
    clinician keyword count are withdrawn (rebuttal/withdrawals.md W14, W23 to W25)."""
    with pytest.raises(KeyError):
        _macro(generated[1], name)


def test_every_macro_has_a_source_comment(generated):
    for line in (generated[1] / "numbers.tex").read_text().splitlines():
        if line.startswith("\\newcommand"):
            assert " % " in line and len(line.split(" % ", 1)[1]) > 5, line


def test_no_absolute_paths_in_outputs(generated):
    # Neither the repo location nor the home folder may leak into committed files.
    for f in generated[1].iterdir():
        text = f.read_text()
        assert str(ROOT) not in text and str(Path.home()) not in text, f.name


def test_numbers_carry_the_report_of_the_uniform_analysis(generated):
    """make numbers writes bodhi_lora.report's macros and tables next to its own, and
    names each analyzed cell's stats.json among its inputs."""
    out, out_dir = generated
    rec = out["uniform_report"]
    assert rec["cells"] and "tab_cross_cell_checks" in rec["tables"]
    for name in rec["tables"]:
        assert (out_dir / f"{name}.tex").is_file(), name
    for cell in rec["cells"].values():
        assert out["meta"]["inputs"][cell["stats"]] == cell["sha256"]
        assert _macro(out_dir, f"{cell['token']}Seeds")


# ---------------------------------------------------------------- determinism

def test_two_runs_are_byte_identical(generated, tmp_path):
    _run(tmp_path)
    first = {f.name: f.read_bytes() for f in generated[1].iterdir()}
    second = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def test_committed_outputs_are_current():
    """paper/generated/ must be what `make numbers` writes from the committed data.

    numbers.json names its inputs by the paths they had when the paper's numbers were
    first written. results/provenance.json maps each old path to the file's place now.
    """
    committed = ROOT / "paper" / "generated" / "numbers.json"
    if not committed.is_file():
        pytest.skip("paper/generated not committed yet")
    moved = {e["old"]: e["new"][0] for e in json.loads((RESULTS / "provenance.json").read_text())}
    # The HealthBench file moved to data/raw/healthbench/ before the results root; it
    # is ignored by git, so a clean checkout may not have it.
    moved["data/raw/healthbench_hard.jsonl"] = "data/raw/healthbench/hard.jsonl"
    meta = json.loads(committed.read_text())["meta"]["inputs"]
    for label, sha in meta.items():
        p = ROOT / moved.get(label, label)
        if label == "data/raw/healthbench_hard.jsonl" and not p.is_file():
            continue
        assert p.is_file(), f"{label}: input not found at {p.relative_to(ROOT)}"
        assert ps._sha256(p) == sha, f"{label} changed since paper/generated was built; run make numbers"


# ---------------------------------------------------------------- rebuttal cells

def test_posted_values_fixture_builds_generality_table(tmp_path):
    out = _run(tmp_path, FIXTURES / "posted_values.json")
    cells = {c["id"]: c for c in out["generality"]["cells"]}
    mistral = cells["healthbench__mistral-small-24b"]["aggregate_lora_minus_base"]
    assert mistral["diff"] == pytest.approx(0.4111 - 0.4248)
    assert "point estimate only" in mistral["note"]
    assert cells["medqa__mistral-small-24b"]["aggregate_lora_minus_base"]["diff"] is None
    tab = (tmp_path / "tab_generality.tex").read_text()
    # Dagger: same family as the evaluator (D6). Asterisk: transcribed from the posts.
    assert "Med42-8B$^{\\dagger\\ast}$" in tab
    assert "[pending]" in tab
    # Set-aside cell sits in its own block after the kept cells.
    assert tab.index("MedQA") > tab.index("Med42-8B")


def test_committed_posted_values_main_cell_once_and_agrees(tmp_path):
    # The committed file repeats the submitted cell as posted. Its rows must come
    # from the cell's results once, and the posted values must match those seed means.
    out = _run(tmp_path, POSTED)
    check = out["generality"]["main_cell_check"]
    assert len(check) == 8
    assert all(r["agrees"] for r in check), [r for r in check if not r["agrees"]]
    for name in ("tab_generality", "tab_generality_aggregate"):
        assert (tmp_path / f"{name}.tex").read_text().count("MedGemma-27B") == 1
    agg = (tmp_path / "tab_generality_aggregate.tex").read_text()
    assert "MedQuAD & Mistral-Small-24B$^{\\ast}$ & \\ensuremath{0.672}" in agg
    assert "\\ensuremath{-0.086}" in agg


def test_cross_cell_counts_leave_out_same_family_cells(tmp_path):
    # D6: the Med42-8B cells share the evaluator's family and stay out of every
    # cross-cell count; the submitted cell is in.
    out = _run(tmp_path, POSTED)
    c = out["generality"]["cross_cell_counts"]
    assert c["excluded_same_family"] == ["chatdoctor__med42-8b", "healthbench__med42-8b"]
    # MedGemma, Mistral HB, BioMistral, ChatDoctor Mistral, MedQuAD Mistral.
    assert c["cells_compared"] == 5
    # Adapter at or above its teacher: Mistral HB 56.6 vs 54.2, ChatDoctor 58.6 vs 54.5,
    # MedQuAD 26.5 vs 23.1. Behind: MedGemma 45.6 vs 47.9, BioMistral 10.1 vs 14.5.
    assert c["adapter_at_or_above_teacher"] == 3
    assert c["ai_rose"] == 4  # all but BioMistral (11.7 -> 10.1)
    assert (c["added_cells_compared"], c["added_ai_rose"]) == (4, 3)
    assert c["added_cells_teacher_below_base"] == 4
    assert c["added_cells_aggregate_fell"] == c["added_cells_with_aggregate"] == 4
    assert c["added_cells_drop_min"] == pytest.approx(0.014)
    assert c["added_cells_drop_max"] == pytest.approx(0.086)
    assert _macro(tmp_path, "CellMedQuADMistralBaseCoTScopeBounding") == "1.61"
    assert _macro(tmp_path, "CellHBBioMistralBaseCoTActiveInquiry") == "14.5"


def test_synced_cell_replaces_posted_values_and_gets_a_t_interval(tmp_path):
    # Once make cells has written generated_values.json, a synced cell's row comes from
    # it, loses the transcription mark and gets a paired t interval. The fake synced
    # Mistral cell here is the MedGemma output, so its interval must equal Table 1's.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cell_values", ROOT / "rebuttal" / "analyses" / "cell_values.py")
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    gen = tmp_path / "generated_values.json"
    # An empty results root, so no synced copy of the other cells is read.
    cv.run(POSTED, tmp_path / "none", {"healthbench__medgemma-27b": M, "healthbench__mistral-small-24b": M}, gen)
    out_dir = tmp_path / "out"
    # No report: it reads the committed stats.json of each cell, which this fake Mistral
    # cell does not match by design, so check_cells_match_report would stop the run.
    out = ps.run(M, out_dir, None, None, None, None, POSTED, gen, with_report=False)
    cells = {c["id"]: c for c in out["generality"]["cells"]}
    m = cells["healthbench__mistral-small-24b"]
    assert m["source"] == "synced"
    main = out["contrasts"]["LoRAvsBase"]["metrics"]["healthbench"]
    # cell_values.py runs the test on the unrounded seed means and stores it at 6
    # significant digits, hence the tolerance.
    assert m["aggregate_lora_minus_base"]["diff"] == pytest.approx(main["diff"], rel=1e-5)
    assert m["aggregate_lora_minus_base"]["ci95"] == pytest.approx(main["ci95"], rel=1e-5)
    assert m["aggregate_lora_minus_base"]["p"] == pytest.approx(main["p"], rel=1e-5)
    assert cells["healthbench__med42-8b"]["source"] == "posted"
    # Mixed sources: the provenance names both, and the posted fallback keeps its mark.
    assert out["generality"]["provenance"] == f"{ps.SRC_SYNCED} when synced, else {ps.SRC_POSTED}, marked *"
    tab = (out_dir / "tab_generality.tex").read_text()
    row = next(line for line in tab.splitlines() if line.startswith("HB-Hard & Mistral-Small-24B"))
    assert "ast" not in row and "pending" not in row
    assert "\\ensuremath{[-0.011,\\,+0.020]}" in row
    assert _macro(out_dir, "CellHBMistralAggregateDiffCI") == "[-0.011,\\,+0.020]"


def test_committed_cells_give_tests_recipe_and_filter_table(tmp_path):
    # The committed generated_values.json and cell_recipe.json (make cells), as the
    # paper uses them: every added cell synced, no transcription mark, one recipe.
    out = ps.run(M, tmp_path, None, None, None, M / "analysis", POSTED,
                 CROSS / "generated_values.json", cross_cell=CROSS)
    cells = {c["id"]: c for c in out["generality"]["cells"]}
    assert {c["source"] for c in cells.values()} == {"synced"}
    c = out["generality"]["cross_cell_counts"]
    # Four drops outside the evaluator's family, Holm over the six cells not set aside.
    assert c["holm_family_cells"] == 6
    assert c["added_cells_drop_p_max"] < 0.02 and c["added_cells_drop_p_holm_max"] < 0.05
    assert (c["added_cells_askers_gap_positive"], c["added_cells_askers_gap_negative"]) == (3, 1)
    # MedQA: seed 42's epistemic grading found none of its prompts; four seeds remain.
    assert _macro(tmp_path, "CellMedQAMistralEpistemicSeeds") == "4"
    assert float(_macro(tmp_path, "CellMedQAMistralBaseActiveInquirySeedMin")) >= 99.0
    assert "stat" + "CellMedQAMistralAggregatePHolm" not in (tmp_path / "numbers.tex").read_text()
    r = out["cell_recipe"]
    assert (r["tau"], r["r"], r["lora_alpha"], r["lora_dropout"], r["epochs"]) == (0.4, 16, 32, 0.05, 3)
    assert r["effective_batch"] == [16] and r["adapters"] == 35 and r["refilter_lines"] == 0
    assert r["refilter_rule_met"] == []
    # Four host runs did not finish; Appendix J names each one.
    assert r["incomplete"] == {"medqa__BioMistral_BioMistral-7B": "partial",
                               "medqa__google_medgemma-27b-text-it": "not run",
                               "medquad__google_medgemma-27b-text-it": "not run",
                               "healthbench__google_gemma-2-9b-it": "no folder"}
    assert _macro(tmp_path, "CellsLoRATargets") == "q, k, v, o, gate, up, down"
    assert _macro(tmp_path, "CellMedQuADMistralFilterClearance") == "78.4"
    tab = (tmp_path / "tab_cell_filter.tex").read_text()
    assert "MedQuAD & Mistral-Small-24B & \\ensuremath{0.4} & 4{,}200 & \\ensuremath{78.4} & 2{,}965" in tab
    assert "Med42-8B}$^{\\dagger}$" in (tmp_path / "tab_cell_dimensions.tex").read_text()
    assert "ast" not in (tmp_path / "tab_generality.tex").read_text()
    # Every row is synced, so no output names the posted values as a source.
    assert out["generality"]["provenance"] == ps.SRC_SYNCED
    for name in ("tab_generality", "tab_generality_aggregate", "tab_cell_dimensions"):
        assert "posted_values" not in (tmp_path / f"{name}.tex").read_text().splitlines()[0]


def _uniform_run(out_dir, generated_values=CROSS / "generated_values.json"):
    return ps.run(M, out_dir, None, None, None, M / "analysis", POSTED, generated_values, cross_cell=CROSS)


def test_every_cell_gets_the_same_statistics(tmp_path):
    # Convention 14: one rule gives every cell the five contrasts on the seven dimensions
    # and the aggregate score, with Holm over the seven within each contrast.
    out = _uniform_run(tmp_path)
    u = out["uniform"]
    assert [c["id"] for c in u["cells"]][0] == "healthbench__medgemma-27b"
    assert len(u["cells"]) == 8
    for c in u["cells"]:
        assert set(c["contrasts"]) == {t for t, _, _ in ps.CONTRASTS}
        for ctok, res in c["contrasts"].items():
            assert set(res) == set(ps.POSTED_METRIC)
            assert all("p_holm" in res[m] for m in ps._gv_dims())
            assert "p_holm" not in res["aggregate"]
    # The submitted cell from generated_values.json equals the raw-output contrasts.
    main = u["cells"][0]["contrasts"]["LoRAvsBase"]
    raw = out["contrasts"]["LoRAvsBase"]["metrics"]
    assert main["active_inquiry_pct"]["diff"] == pytest.approx(100 * raw["active_inquiry"]["diff"], abs=1e-3)
    assert main["red_flag"]["p_holm"] == pytest.approx(raw["red_flag"]["p_holm_within_contrast"], rel=1e-3)
    assert main["aggregate"]["ci95"] == pytest.approx(raw["healthbench"]["ci95"], abs=1e-5)
    # MedQA: seed 42 has no dimension grades, so its dimension contrasts use four seeds.
    medqa = next(c for c in u["cells"] if c["id"] == "medqa__mistral-small-24b")
    assert medqa["contrasts"]["LoRAvsBase"]["active_inquiry_pct"]["n_seeds"] == 4
    assert medqa["contrasts"]["LoRAvsBase"]["aggregate"]["n_seeds"] == 5
    k = u["counts"]
    # MedGemma, Mistral HB, BioMistral, ChatDoctor Mistral, MedQuAD Mistral; all but
    # BioMistral raise active inquiry and context-seeking after Holm.
    assert (k["all"], k["added"], k["outside"], k["outside_ask_rose"]) == (8, 7, 5, 4)
    # The four: MedGemma-27B and Mistral-Small-24B, on HealthBench-Hard, ChatDoctor and MedQuAD.
    assert (k["outside_ask_rose_models"], k["outside_ask_rose_benchmarks"]) == (2, 3)
    # The red-flag counts of LoRA+BODHI against LoRA leave out the Med42-8B cells (D6):
    # Mistral HB, BioMistral, ChatDoctor Mistral and MedQuAD Mistral. Only MedQuAD rises.
    assert (k["added_outside"], k["added_red_flag_cot_fell"], k["added_red_flag_cot_rose"]) == (4, 0, 1)
    assert (k["same_family"], k["set_aside"]) == (2, 1)
    bio = next(c for c in u["cells"] if c["id"] == "healthbench__biomistral-7b")
    assert (bio["dims_rose"], bio["dims_fell"]) == (0, 0)
    assert _macro(tmp_path, "CellsOutsideAskRose") == "4"
    assert _macro(tmp_path, "CellHBMistralRedFlagLoRAvsBaseDiff") == "+0.64"
    # _macro strips trailing braces, so the exponent's closing brace is gone here.
    assert _macro(tmp_path, "CellMedQuADMistralAggregateLoRAvsBaseP") == "8.4\\times10^{-4"


def test_uniform_tables(tmp_path):
    _uniform_run(tmp_path)
    tab = (tmp_path / "tab_cross_cell.tex").read_text()
    rows = [line for line in tab.splitlines() if " & " in line and "\\ensuremath" in line]
    assert len(rows) == 8
    assert rows[0].startswith("HB-Hard & MedGemma-27B & \\ensuremath{17.5} & \\ensuremath{45.6}")
    # Holm star on the dimensions; the submitted cell's aggregate has none (p = 0.48).
    assert "\\ensuremath{+28.0\\pm5.1^{*}}" in rows[0]
    assert rows[0].endswith("\\ensuremath{+0.004\\pm0.015} \\\\")
    assert "Med42-8B$^{\\dagger}$" in tab
    # The set-aside cell sits last, in its own block.
    assert rows[-1].startswith("MedQA & ") and tab.rindex("\\midrule") < tab.index("MedQA & ")
    detail = (tmp_path / "tab_cell_detail_1.tex").read_text() + (tmp_path / "tab_cell_detail_2.tex").read_text()
    assert detail.count("\\multicolumn{9}{l}") == 8
    assert "(4 of 5 seeds have dimension grades)" in detail


def test_main_cell_mismatch_stops_the_run(tmp_path):
    gv = json.loads((CROSS / "generated_values.json").read_text())
    main = next(c for c in gv["cells"] if c["id"] == "healthbench__medgemma-27b")
    main["conditions"]["lora"]["red_flag"]["per_seed"]["42"] += 0.01
    path = tmp_path / "generated_values.json"
    path.write_text(json.dumps(gv))
    with pytest.raises(SystemExit, match="healthbench__medgemma-27b lora_no_wrapper red_flag seed 42"):
        _uniform_run(tmp_path / "out", path)


def test_table_1_and_the_report_agree_on_every_cell():
    # Table 1 reads generated_values.json; the appendix checks table and the per-cell
    # macros read each cell's stats.json. Every per-seed mean must agree: 8 cells, 4
    # conditions, 8 metrics, 5 seeds, less the 28 dimension means of MedQA seed 42.
    gv = json.loads((CROSS / "generated_values.json").read_text())
    assert ps.check_cells_match_report(gv) == 8 * 4 * 8 * 5 - 4 * 7
    bio = next(c for c in gv["cells"] if c["id"] == "healthbench__biomistral-7b")
    bio["conditions"]["lora"]["red_flag"]["per_seed"]["7"] += 0.01
    with pytest.raises(SystemExit, match="healthbench__biomistral-7b lora red_flag seed 7"):
        ps.check_cells_match_report(gv)


def test_token_counts_come_from_the_uniform_analysis(generated):
    out, d = generated
    c = out["cost"]["conditions"]
    # The engine stored a count for the one-pass conditions only.
    assert c["base_bodhi"]["tokens_engine_n"] == c["lora_bodhi"]["tokens_engine_n"] == 0
    assert c["base_no_wrapper"]["tokens_engine_n"] == c["lora_no_wrapper"]["tokens_engine_n"] == 1000
    assert _macro(d, "TokensEngineBase") == "753" and _macro(d, "TokensEngineLoRA") == "781"
    assert "statTokensEngineBaseCoT" not in (d / "numbers.tex").read_text()


def test_completion_drop_and_filter_macros(generated):
    out, out_dir = generated
    assert _macro(out_dir, "CompletionDropPctBase") == "0.0"
    assert _macro(out_dir, "CompletionDropPctBaseCoT") == "4.3"   # 957 of 1,000
    assert _macro(out_dir, "CompletionDropPctLoRACoT") == "4.1"   # 959 of 1,000


# The tracked score copies (scripts/export_derived_inputs.py).
FILTER_SCORES = M / "seed_7" / "filter" / "filter_scores.json"
QWEN_SCORES = PILOT / "analysis" / "qwen_grader_scores.json"


def test_cross_grader_side_run(tmp_path):
    # Reads the tracked score copy, so it runs in a clean checkout (audit REPRO-6).
    out = ps.run(M, tmp_path, None, None, None, None, None, None, QWEN_SCORES)
    cg = out["cross_grader"]
    assert {c["n_shared"] for c in cg["conditions"].values()} == {43}
    assert all(c["grader"] == "Qwen/Qwen2.5-14B-Instruct" for c in cg["conditions"].values())
    # The paper had r = 0.42 pooled (0.35, 0.37, 0.51) and Spearman 0.34; Spearman is 0.354.
    assert cg["pooled"]["pearson_r"] == pytest.approx(0.4237, abs=1e-3)
    assert cg["pooled"]["spearman_rho"] == pytest.approx(0.3542, abs=1e-3)
    assert cg["conditions"]["lora_no_wrapper"]["minus_base_qwen"] == pytest.approx(-0.008, abs=5e-4)


def test_tau_table_from_tracked_scores(tmp_path):
    out = ps.run(M, tmp_path, None, FILTER_SCORES, None, None, None)
    tc = out["tau_calibration"]
    assert tc["n_train_split"] == 2816
    assert tc["rows"][0] == {"tau": 0.4, "share_of_train_split": 1.0,
                             "share_of_pool_estimate": pytest.approx(3129 / 4000)}
    assert round(100 * tc["rows"][1]["share_of_train_split"], 1) == 77.1   # tau 0.5
    assert (tmp_path / "tab_tau_calibration.tex").is_file()


def _without_source_paths(obj):
    """The copy minus its "source" path labels, which name where the export read."""
    if isinstance(obj, dict):
        return {k: _without_source_paths(v) for k, v in obj.items() if k != "source"}
    return obj


@pytest.mark.parametrize("which", ["filter", "qwen"])
def test_derived_inputs_match_their_sources(tmp_path, which):
    """The tracked score copies equal a fresh export from the ignored source files:
    the same rows and the same source sha256."""
    from scripts import export_derived_inputs as ex
    src = {"filter": M / "seed_7" / "filter" / "train.jsonl",
           "qwen": PILOT / "seed_42" / "eval" / "base.json"}[which]
    if not src.is_file():
        pytest.skip("ignored source file not present")
    tracked = FILTER_SCORES if which == "filter" else QWEN_SCORES
    fresh = ex.export_filter_scores(src) if which == "filter" else ex.export_qwen_scores(src.parent)
    ex._write(tmp_path / "copy.json", fresh)
    assert (_without_source_paths(json.loads((tmp_path / "copy.json").read_text()))
            == _without_source_paths(json.loads(tracked.read_text())))


def test_missing_posted_values_prints_skip(tmp_path, capsys):
    _run(tmp_path, tmp_path / "absent.json")
    assert "SKIP generality table" in capsys.readouterr().out
    assert not (tmp_path / "tab_generality.tex").exists()
    assert not (tmp_path / "tab_generality_aggregate.tex").exists()


def test_posted_values_schema_is_checked(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cells": [{"id": "x"}]}))
    with pytest.raises(SystemExit):
        ps.load_posted_values(bad)


# ---------------------------------------------------------------- units

def test_small_p_values_survive_in_numbers_json(generated):
    p = json.loads((generated[1] / "numbers.json").read_text())[
        "contrasts"]["LoRACoTvsLoRA"]["metrics"]["red_flag"]["pooled_welch_p"]
    assert 0 < p < 1e-6


def test_formatting():
    assert ps.fmt(0.1234, 2) == "\\ensuremath{0.12}"
    assert ps.fmt(-0.0004, 3, sign=True) == "\\ensuremath{0.000}"
    assert ps.fmt(float("nan"), 2) == "n/a"
    assert ps.fmt_p(0.0622) == "\\ensuremath{0.062}"
    assert ps.fmt_p(1.12e-4) == "\\ensuremath{1.1\\times10^{-4}}"
    assert ps.fmt_ci(-0.011, 0.01953, 3) == "\\ensuremath{[-0.011,\\,+0.020]}"
    assert ps.fmt_int(1000) == "1{,}000"


def test_t_interval_matches_textbook():
    m, lo, hi = ps.t_interval(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    half = 2.7764451051977987 * math.sqrt(2.5) / math.sqrt(5)
    assert (m, lo, hi) == pytest.approx((3.0, 3.0 - half, 3.0 + half))


def test_cluster_bootstrap_identical_conditions_give_zero_width():
    frame = pd.DataFrame({"prompt_id": ["a", "a", "b", "c"], "v": [1.0, 0.0, 1.0, 0.5]})
    boot = ps.ClusterBootstrap(["a", "b", "c"], resamples=200, seed=0)
    r = boot.diff_and_d(frame, frame)
    assert r["diff_ci95"] == [0.0, 0.0]


def test_macro_names_must_be_letters():
    m = ps.Macros()
    with pytest.raises(ValueError):
        m.add("Seed7Mean", "1", "x")
    m.add("Ok", "1", "x")
    with pytest.raises(ValueError):
        m.add("Ok", "2", "x")


def test_check_paper_macros_names_missing(tmp_path, monkeypatch, capsys):
    from scripts import check_paper_macros as cm
    full = (ROOT / "paper" / "generated" / "numbers.tex").read_text()
    assert cm.main() == 0
    # Drop one macro the paper uses: the check must fail and name it.
    cut = "\n".join(line for line in full.splitlines()
                    if not line.startswith("\\newcommand{\\statFilterTrainNSeedSeven}"))
    (tmp_path / "numbers.tex").write_text(cut)
    monkeypatch.setattr(cm, "NUMBERS", tmp_path / "numbers.tex")
    monkeypatch.setattr(cm, "ROOT", ROOT)
    capsys.readouterr()
    assert cm.main() == 1
    assert "statFilterTrainNSeedSeven" in capsys.readouterr().out

