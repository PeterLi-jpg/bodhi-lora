"""Tests for scripts/paper_stats.py.

The golden tests run the generator on the committed results_modal/ data and pin the
values the paper reports, as they actually compute. Where the paper's printed value
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


def _run(out_dir, posted=None):
    # Optional inputs off, except the committed kappa and rebuttal outputs, so the
    # test does not depend on gitignored data.
    return ps.run(ROOT / "results_modal", out_dir, None, None,
                  ROOT / "results_modal" / "irr" / "kappa_results.json",
                  ROOT / "rebuttal" / "results", posted)


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


def test_axis_scores(generated):
    ax = generated[0]["axis_scores"]
    assert round(ax["accuracy"]["base_no_wrapper"], 3) == 0.117
    assert round(ax["instruction_following"]["lora_no_wrapper"], 3) == 0.286
    assert round(ax["communication_quality"]["lora_bodhi"], 3) == 0.246


def test_kappa_and_rebuttal_macros(generated):
    d = generated[1]
    assert _macro(d, "KappaAll") == "0.35"
    assert _macro(d, "KappaAllN") == "509"
    assert _macro(d, "LeakagePctLoRACoT") == "54.6"
    assert _macro(d, "AskOddsRatioPrimaryLoRA") == "1.54"


def test_every_macro_has_a_source_comment(generated):
    for line in (generated[1] / "numbers.tex").read_text().splitlines():
        if line.startswith("\\newcommand"):
            assert " % " in line and len(line.split(" % ", 1)[1]) > 5, line


def test_no_absolute_paths_in_outputs(generated):
    # Neither the repo location nor the home folder may leak into committed files.
    for f in generated[1].iterdir():
        text = f.read_text()
        assert str(ROOT) not in text and str(Path.home()) not in text, f.name


# ---------------------------------------------------------------- determinism

def test_two_runs_are_byte_identical(generated, tmp_path):
    _run(tmp_path)
    first = {f.name: f.read_bytes() for f in generated[1].iterdir()}
    second = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def test_committed_outputs_are_current():
    """paper/generated/ must be what `make numbers` writes from the committed data."""
    committed = ROOT / "paper" / "generated" / "numbers.json"
    if not committed.is_file():
        pytest.skip("paper/generated not committed yet")
    meta = json.loads(committed.read_text())["meta"]["inputs"]
    for rel, sha in meta.items():
        p = ROOT / rel
        if p.is_file():
            assert ps._sha256(p) == sha, f"{rel} changed since paper/generated was built; run make numbers"


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
    # from results_modal once, and the posted values must match those seed means.
    out = _run(tmp_path, ROOT / "data" / "rebuttal_cells" / "posted_values.json")
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
    out = _run(tmp_path, ROOT / "data" / "rebuttal_cells" / "posted_values.json")
    c = out["generality"]["cross_cell_counts"]
    assert c["excluded_same_family"] == ["chatdoctor__med42-8b", "healthbench__med42-8b"]
    # MedGemma, Mistral HB, BioMistral, ChatDoctor Mistral, MedQuAD Mistral.
    assert c["cells_compared"] == 5
    # Adapter at or above its teacher: Mistral HB 56.6 vs 54.2, ChatDoctor 58.6 vs 54.5,
    # MedQuAD 26.5 vs 23.1. Behind: MedGemma 45.6 vs 47.9, BioMistral 10.1 vs 14.5.
    assert c["adapter_at_or_above_teacher"] == 3
    assert c["ai_rose"] == 4  # all but BioMistral (11.7 -> 10.1)
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
    cv.run(ROOT / "data" / "rebuttal_cells" / "posted_values.json", tmp_path / "none",
           ROOT / "results_modal", {"healthbench__mistral-small-24b": ROOT / "results_modal"}, gen)
    out_dir = tmp_path / "out"
    out = ps.run(ROOT / "results_modal", out_dir, None, None, None, None,
                 ROOT / "data" / "rebuttal_cells" / "posted_values.json", gen)
    cells = {c["id"]: c for c in out["generality"]["cells"]}
    m = cells["healthbench__mistral-small-24b"]
    assert m["source"] == "synced"
    main = out["contrasts"]["LoRAvsBase"]["metrics"]["healthbench"]
    # generated_values.json keeps 6 significant digits, hence the tolerance.
    assert m["aggregate_lora_minus_base"]["diff"] == pytest.approx(main["diff"], abs=1e-5)
    assert m["aggregate_lora_minus_base"]["ci95"] == pytest.approx(main["ci95"], abs=1e-5)
    assert cells["healthbench__med42-8b"]["source"] == "posted"
    tab = (out_dir / "tab_generality.tex").read_text()
    row = next(line for line in tab.splitlines() if line.startswith("HB-Hard & Mistral-Small-24B"))
    assert "ast" not in row and "pending" not in row
    assert "\\ensuremath{[-0.011,\\,+0.020]}" in row
    assert _macro(out_dir, "CellHBMistralAggregateDiffCI") == "[-0.011,\\,+0.020]"


def test_completion_drop_and_filter_macros(generated):
    out, out_dir = generated
    assert _macro(out_dir, "CompletionDropPctBase") == "0.0"
    assert _macro(out_dir, "CompletionDropPctBaseCoT") == "4.3"   # 957 of 1,000
    assert _macro(out_dir, "CompletionDropPctLoRACoT") == "4.1"   # 959 of 1,000


def test_cross_grader_side_run(tmp_path):
    qdir = ROOT / "results" / "seed_42" / "eval"
    if not (qdir / "base_no_wrapper.json").is_file():
        pytest.skip("gitignored Qwen-graded side run not present")
    out = ps.run(ROOT / "results_modal", tmp_path, None, None, None, None, None, None, qdir)
    cg = out["cross_grader"]
    assert {c["n_shared"] for c in cg["conditions"].values()} == {43}
    assert all(c["grader"] == "Qwen/Qwen2.5-14B-Instruct" for c in cg["conditions"].values())
    # The paper had r = 0.42 pooled (0.35, 0.37, 0.51) and Spearman 0.34; Spearman is 0.354.
    assert cg["pooled"]["pearson_r"] == pytest.approx(0.4237, abs=1e-3)
    assert cg["pooled"]["spearman_rho"] == pytest.approx(0.3542, abs=1e-3)
    assert cg["conditions"]["lora_no_wrapper"]["minus_base_qwen"] == pytest.approx(-0.008, abs=5e-4)


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
