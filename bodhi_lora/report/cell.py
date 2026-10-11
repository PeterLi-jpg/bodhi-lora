"""One cell's macros and tables, from its analysis/stats.json.

Every analyzed cell gets the same macros (cell_macros) and its table tab_cell_<token>
(cell_table). A cell that configs/report.yaml lists under detail_cells also gets the
detail tables (detail_tables), which the paper prints for the cell of the submitted
version. Nothing here names a cell, a model or a benchmark.
"""
from __future__ import annotations

import math

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.analysis.extract import EPISTEMIC, SEVEN
from bodhi_lora.registry import CellConfig, condition_from_legacy
from bodhi_lora.report.common import (DIFF_DECIMALS, KIND_FORMAT, Names, contrast_pair, scaled, with_protocol,
                                      x)
from bodhi_lora.report.latex import (MISSING, escape, finite, fmt, fmt_ci, fmt_int, fmt_p, fmt_pm, tabular,
                                     token)

# The metrics that get condition and contrast macros, in emission order.
METRICS = (*EPISTEMIC, "rubric_score", "neg_any", "chars", "tokens")
# The metrics whose per-seed range gets macros (a cell with a seed that has no grade
# states the range of the seeds that have one).
SEED_RANGE = (*SEVEN, "rubric_score")
# The internalization gap table: how far the adapter is from its teacher.
GAP_METRICS = ("active_inquiry", "red_flag", "context_seeking", "scope_bounding", "specificity",
               "uncertainty_ack", "appropriate_hedging")
# The dimension table: the six 0-2 grades, then the counts and rates. Higher is better,
# except where DIRECTION says otherwise.
DIMENSION_ROWS = (("uncertainty_ack", "context_seeking", "red_flag", "scope_bounding", "hedging_quality",
                   "specificity"),
                  ("questions", "active_inquiry", "red_flag_rate", "scope_bounded_rate", "specificity_rate",
                   "blanket_disclaimer", "appropriate_hedging"))
# Asking helps only where facts are missing, so neither questions per response nor the
# share of replies that ask has a better direction (audit CR-19); a blanket disclaimer is
# worse.
DIRECTION = {"blanket_disclaimer": "$\\downarrow$", "questions": "$\\circ$", "active_inquiry": "$\\circ$"}


def _metrics(stats: dict) -> list[str]:
    return [k for k in METRICS if k in stats["metrics"]]


def _kind(stats: dict, m: str) -> str:
    return stats["metrics"][m]["kind"]


# -- macros -----------------------------------------------------------------------------

def cell_macros(rep, stats: dict, cell: CellConfig, cfg: ReportConfig, src: str) -> None:
    """Every macro of one cell, each named \\stat<token><...> after the cell record's
    report.token. A value the cell does not have gets no macro."""
    n = Names(cfg)
    T = cell.report.token
    conds, ctrs, seeds = stats["conditions_order"], stats["contrasts_order"], stats["seeds"]
    counts = stats["counts"]
    _counts(rep, n, T, stats, src)
    for m in _metrics(stats):
        scale, dp = KIND_FORMAT[_kind(stats, m)]
        ddp = DIFF_DECIMALS[_kind(stats, m)]
        M = token(m)
        for c in conds:
            r = stats["conditions"][m][c]
            C = n.cond_token(c)
            if finite(r.get("seed_mean")):
                rep.add(f"{T}{M}{C}", fmt(r["seed_mean"] * scale, dp),
                        f"{src}: conditions.{m}.{c}.seed_mean, mean over {r['n_seeds']} seeds")
            if m in SEED_RANGE:
                per = [v for v in r["per_seed"].values() if finite(v)]
                if per:
                    rep.add(f"{T}{M}{C}SeedMin", fmt(min(per) * scale, dp), f"{src}: conditions.{m}.{c}.per_seed, lowest")
                    rep.add(f"{T}{M}{C}SeedMax", fmt(max(per) * scale, dp), f"{src}: conditions.{m}.{c}.per_seed, highest")
        for k in ctrs:
            r = stats["contrasts"][k]["metrics"][m]
            if not finite(r.get("diff")):
                continue
            name = f"{T}{M}{n.ctr_token(k)}"
            where = f"{src}: contrasts.{k}.metrics.{m}"
            rep.add(name + "Diff", fmt(r["diff"] * scale, ddp, sign=True), f"{where}.diff, over {r['n_seeds']} seeds")
            rep.add(name + "CI", fmt_ci(scaled(r["ci95"], scale), ddp), f"{where}.ci95, t({r['df']})")
            rep.add(name + "P", fmt_p(r.get("p")), f"{where}.p, paired t over seeds")
            if m in SEVEN:
                rep.add(name + "PHolm", fmt_p(r.get("p_holm_within_contrast")),
                        f"{where}.p_holm_within_contrast, Holm over the seven dimensions")
                rep.add(name + "PHolmAll", fmt_p(r.get("p_holm_all_contrasts")),
                        f"{where}.p_holm_all_contrasts, Holm over the seven dimensions of every contrast")
            if "pooled_d" in r:
                rep.add(name + "D", fmt(r["pooled_d"], 2, sign=True), f"{where}.pooled_d, descriptive")
                rep.add(name + "DCI", fmt_ci(r.get("d_ci95"), 2), f"{where}.d_ci95, prompt-clustered bootstrap")
                rep.add(name + "BootCI", fmt_ci(scaled(r.get("diff_ci95"), scale), ddp),
                        f"{where}.diff_ci95, prompt-clustered bootstrap of the pooled difference")
            if r["n_seeds"] < len(seeds):
                rep.add(name + "Seeds", fmt_int(r["n_seeds"]), f"{where}.n_seeds")
            wc = (stats["worst_case"].get(m) or {}).get("contrasts", {}).get(k)
            if wc and finite(wc.get("diff")):
                rep.add(name + "WorstCaseDiff", fmt(wc["diff"] * scale, ddp, sign=True),
                        f"{src}: worst_case.{m}.contrasts.{k}.diff (failed grades at the worst value)")
                rep.add(name + "WorstCaseP", fmt_p(wc.get("p")), f"{src}: worst_case.{m}.contrasts.{k}.p")
    for k in ctrs:
        res = stats["contrasts"][k]["metrics"]
        K = n.ctr_token(k)
        for word, sign in (("Up", 1), ("Down", -1)):
            hits = sum(finite(res[d].get("p_holm_within_contrast")) and res[d]["p_holm_within_contrast"] < cfg.alpha
                       and sign * res[d]["diff"] > 0 for d in SEVEN if d in res)
            rep.add(f"{T}Dims{word}{K}", fmt_int(hits),
                    f"{src}: contrasts.{k}: dimensions {'higher' if sign > 0 else 'lower'} with Holm p < {cfg.alpha}")
    _disclaimers(rep, n, T, stats, src)
    _anchor_lengths_axes(rep, n, T, stats, src)
    _discrimination(rep, n, T, stats, src)
    _leakage(rep, n, T, stats, src)
    for c, r in stats["askers_gap"].items():
        rep.add(f"{T}AskersGap{n.cond_token(c)}", fmt(r.get("gap"), 3, sign=True), f"{src}: askers_gap.{c}.gap")
        rep.add(f"{T}AskersGap{n.cond_token(c)}CI", fmt_ci(r.get("gap_ci95"), 3), "prompt-clustered bootstrap interval")
    cost = stats["cost"]
    for c in conds:
        r = cost["conditions"][c]
        if r["tokens_engine_n"]:
            rep.add(f"{T}TokensEngine{n.cond_token(c)}", fmt(r["tokens_engine_mean"], 0),
                    f"{src}: cost.conditions.{c}.tokens_engine_mean, {r['tokens_engine_n']} of {r['n']} replies")
    for k in ctrs:
        for col in ("chars", "tokens"):
            rep.add(f"{T}{token(col)}{n.ctr_token(k)}Pct", fmt(cost["contrasts"][k][f"{col}_pct"], 0, sign=True),
                    f"{src}: cost.contrasts.{k}.{col}_pct (pooled means, %)")
    _prompts_filter_recipe(rep, T, stats, src)
    _validation(rep, n, T, stats, src)


def _counts(rep, n: Names, T: str, stats: dict, src: str) -> None:
    counts = stats["counts"]
    per = counts["per_condition"]
    rep.add(f"{T}Seeds", fmt_int(len(stats["seeds"])), f"{src}: seeds")
    rep.add(f"{T}SeedsGraded", fmt_int(len(counts["seeds_graded"])), f"{src}: counts.seeds_graded")
    rep.add(f"{T}Prompts", fmt_int(counts["unique_prompts"]), f"{src}: counts.unique_prompts")
    rep.add(f"{T}PromptsRepeated", fmt_int(counts["prompts_in_more_than_one_seed"]),
            f"{src}: counts.prompts_in_more_than_one_seed")
    rep.add(f"{T}PromptDraws", fmt_int(counts["prompt_draws_per_condition"]),
            f"{src}: counts.prompt_draws_per_condition")
    for c in stats["conditions_order"]:
        r, C = per[c], n.cond_token(c)
        where = f"{src}: counts.per_condition.{c}"
        rep.add(f"{T}Completions{C}", fmt_int(r["completions"]), f"{where}.completions")
        rep.add(f"{T}CompletionDropPct{C}", fmt(x(r["completion_drop_rate"], 100), 1),
                f"{where}.completion_drop_rate (% of drawn prompts with no completion)")
        rep.add(f"{T}Scored{C}", fmt_int(r["epistemic_scored"]), f"{where}.epistemic_scored")
        rep.add(f"{T}EpistemicParseFailPct{C}", fmt(x(r["epistemic_parse_failure_rate"], 100), 2),
                f"{where}.epistemic_parse_failure_rate (%)")
    rep.add(f"{T}EpistemicParseFailPctMax", fmt(x(counts["epistemic_parse_failure_rate_max"], 100), 2),
            f"{src}: counts.epistemic_parse_failure_rate_max (%)")
    rep.add(f"{T}RubricParseFailPctMaxRun", fmt(x(counts["rubric_parse_failure_rate_max_run"], 100), 3),
            f"{src}: counts.rubric_parse_failure_rate_max_run (%)")


def _disclaimers(rep, n: Names, T: str, stats: dict, src: str) -> None:
    bd = stats["blanket_disclaimers"]
    for c in stats["conditions_order"]:
        r, C = bd[c], n.cond_token(c)
        rep.add(f"{T}BlanketDisclaimerCount{C}", fmt_int(r["count"]), f"{src}: blanket_disclaimers.{c}.count")
        rep.add(f"{T}BlanketDisclaimerN{C}", fmt_int(r["n"]), f"{src}: blanket_disclaimers.{c}.n (scored replies)")
        if r["n"]:
            rep.add(f"{T}BlanketDisclaimerPct{C}", fmt(100 * r["count"] / r["n"], 1),
                    f"{src}: blanket_disclaimers.{c}, count over scored replies (%)")
        rep.add(f"{T}BlanketDisclaimerFisherP{C}", fmt_p(r.get("fisher_p_vs_base")),
                f"{src}: blanket_disclaimers.{c}.fisher_p_vs_base, two-sided Fisher exact")
    a = bd["all_conditions"]
    rep.add(f"{T}BlanketDisclaimerFlagged", fmt_int(a["flagged"]), f"{src}: blanket_disclaimers.all_conditions.flagged")
    rep.add(f"{T}BlanketDisclaimerFlaggedHedgingTwo", fmt_int(a["flagged_hedging_2"]),
            f"{src}: blanket_disclaimers.all_conditions.flagged_hedging_2")


def _anchor_lengths_axes(rep, n: Names, T: str, stats: dict, src: str) -> None:
    for c, r in stats["context_seeking_anchor"].items():
        rep.add(f"{T}ContextSeekingTwoNoQuestionPct{n.cond_token(c)}", fmt(x(r["share_no_question"], 100), 1),
                f"{src}: context_seeking_anchor.{c}.share_no_question (%)")
    lq = stats["length_quartiles"]
    for i, v in enumerate(lq.get("first_length_of_quartile", [])):
        rep.add(f"{T}LengthQuartileBound{'ABC'[i]}", fmt_int(v),
                f"{src}: length_quartiles.first_length_of_quartile[{i}] (characters, from {lq['reference']})")
    for k, r in stats["shift_correlations"].items():
        K = n.ctr_token(k)
        rep.add(f"{T}ShiftCorrN{K}", fmt_int(r["n_pairs"]), f"{src}: shift_correlations.{k}.n_pairs")
        for d, v in r["pearson_r_with_active_inquiry"].items():
            rep.add(f"{T}ShiftCorr{token(d)}{K}", fmt(v, 2, sign=True),
                    f"{src}: shift_correlations.{k}.pearson_r_with_active_inquiry.{d}")
    for axis, byc in stats["axis_scores"].items():
        for c, v in byc.items():
            rep.add(f"{T}Axis{token(axis)}{n.cond_token(c)}", fmt(v, 3),
                    f"{src}: axis_scores.{axis}.{c} (mean over pooled completions)")


def _discrimination(rep, n: Names, T: str, stats: dict, src: str) -> None:
    conds, ctrs = stats["conditions_order"], stats["contrasts_order"]
    for resp, labels in stats["discrimination"].items():
        for lab, r in labels.items():
            if "not_measured" in r:
                continue
            P = f"{T}{token(resp)}{token(lab)}"
            where = f"{src}: discrimination.{resp}.{lab}"
            rep.add(P + "PosPrompts", fmt_int(r["n_prompts"]["positive"]), f"{where}.n_prompts.positive")
            rep.add(P + "NegPrompts", fmt_int(r["n_prompts"]["negative"]), f"{where}.n_prompts.negative")
            for c in conds:
                rc, C = r["conditions"][c], n.cond_token(c)
                rep.add(f"{P}Index{C}", fmt(x(rc["discrimination"], 100), 1, sign=True),
                        f"{where}.conditions.{c}.discrimination, pooled (pp)")
                rep.add(f"{P}Index{C}CI", fmt_ci(scaled(rc["discrimination_ci95"], 100), 1),
                        "prompt-clustered bootstrap interval")
                for g in ("positive", "negative"):
                    G = "Pos" if g == "positive" else "Neg"
                    rep.add(f"{P}{G}Rate{C}", fmt(x(rc[f"{g}_rate"], 100), 1),
                            f"{where}.conditions.{c}.{g}_rate, pooled (%)")
                    rep.add(f"{P}{G}Rate{C}CI", fmt_ci(scaled(rc[f"{g}_rate_ci95"], 100), 1, sign=False),
                            "prompt-clustered bootstrap interval")
            for k in ctrs:
                rk, K = r["contrasts"][k], n.ctr_token(k)
                rep.add(f"{P}Index{K}Diff", fmt(x(rk["diff"], 100), 1, sign=True), f"{where}.contrasts.{k}.diff (pp)")
                rep.add(f"{P}Index{K}CI", fmt_ci(scaled(rk["diff_ci95"], 100), 1), "prompt-clustered bootstrap interval")
                # The seed-level test of the index difference: the test of E02
                # (docs/ANALYSIS_PLAN.md section 5.2); the pooled difference above is descriptive.
                t = rk.get("seed_test") or {}
                if finite(t.get("diff")):
                    w = f"{where}.contrasts.{k}.seed_test"
                    rep.add(f"{P}Index{K}SeedDiff", fmt(t["diff"] * 100, 1, sign=True),
                            f"{w}.diff, over {t['n_seeds']} seeds (pp)")
                    rep.add(f"{P}Index{K}SeedCI", fmt_ci(scaled(t["ci95"], 100), 1), f"{w}.ci95, t({t['df']})")
                    rep.add(f"{P}Index{K}SeedP", fmt_p(t.get("p")), f"{w}.p, paired t over seeds")
                for g in ("positive", "negative"):
                    G = "Pos" if g == "positive" else "Neg"
                    t = rk[f"{g}_rate"]["seed_test"]
                    if not finite(t.get("diff")):
                        continue
                    w = f"{where}.contrasts.{k}.{g}_rate.seed_test"
                    rep.add(f"{P}{G}Rate{K}Diff", fmt(t["diff"] * 100, 1, sign=True),
                            f"{w}.diff, over {t['n_seeds']} seeds (pp)")
                    rep.add(f"{P}{G}Rate{K}CI", fmt_ci(scaled(t["ci95"], 100), 1), f"{w}.ci95, t({t['df']})")
                    rep.add(f"{P}{G}Rate{K}P", fmt_p(t.get("p")), f"{w}.p, paired t over seeds")


def _leakage(rep, n: Names, T: str, stats: dict, src: str) -> None:
    for col, r in stats["leakage"].items():
        L = f"{T}Leak{token(r['protocol'])}{token(r['rule'])}"
        for c in stats["conditions_order"]:
            rc, where = r["conditions"][c], f"{src}: leakage.{col}.conditions.{c}"
            C = n.cond_token(c)
            rep.add(f"{L}{C}", fmt(x(rc["leaked_share"], 100), 1), f"{where}.leaked_share (% of scored replies)")
            rep.add(f"{L}{C}N", fmt_int(rc["n_leaked"]), f"{where}.n_leaked (scored replies that leak)")
            rep.add(f"{L}{C}NClean", fmt_int(rc["n_clean"]), f"{where}.n_clean (scored replies that do not)")
            rep.add(f"{L}{C}NScored", fmt_int(rc["n_scored"]), f"{where}.n_scored")
            rep.add(f"{L}{C}NCompletions", fmt_int(rc["n_completions"]), f"{where}.n_completions (every reply)")
            rep.add(f"{L}{C}NLeakedCompletions", fmt_int(rc["n_leaked_completions"]),
                    f"{where}.n_leaked_completions (every reply, graded or not)")
            if rc["n_completions"]:
                rep.add(f"{L}{C}CompletionsPct", fmt(100 * rc["n_leaked_completions"] / rc["n_completions"], 1),
                        f"{where}: n_leaked_completions over n_completions (%)")
            for g, G in (("leaked", "Leaked"), ("clean", "Clean")):
                rep.add(f"{L}{C}RedFlag{G}", fmt((rc.get(f"dimensions_{g}") or {}).get("red_flag"), 2),
                        f"{where}.dimensions_{g}.red_flag (0-2, pooled over seeds)")
            for i, q in enumerate(rc.get("clean_red_flag_by_length_quartile") or [], start=1):
                rep.add(f"{L}{C}CleanRedFlagQ{token(str(i))}", fmt(q["red_flag"], 2),
                        f"{where}.clean_red_flag_by_length_quartile[{i - 1}].red_flag")
                rep.add(f"{L}{C}CleanCharsQ{token(str(i))}", fmt_int(round(q["chars_median"])) if finite(q["chars_median"])
                        else MISSING, f"{where}.clean_red_flag_by_length_quartile[{i - 1}].chars_median")
                rep.add(f"{L}{C}CleanNQ{token(str(i))}", fmt_int(q["n"]),
                        f"{where}.clean_red_flag_by_length_quartile[{i - 1}].n")


def _prompts_filter_recipe(rep, T: str, stats: dict, src: str) -> None:
    p = stats["prompts"]
    if p.get("n_population") is not None:
        rep.add(f"{T}PopulationPrompts", fmt_int(p["n_population"]), f"{src}: prompts.n_population")
    if finite(p.get("rubric_points_pos_min")):
        rep.add(f"{T}RubricPointsMin", fmt_int(p["rubric_points_pos_min"]),
                f"{src}: prompts.rubric_points_pos_min (positive rubric points, evaluated prompts)")
        rep.add(f"{T}RubricPointsMax", fmt_int(p["rubric_points_pos_max"]), f"{src}: prompts.rubric_points_pos_max")
    for lab, r in (p.get("population_labels") or {}).items():
        L = f"{T}Label{token(lab)}Pop"
        rep.add(L + "Pos", fmt_int(r["positive"]), f"{src}: prompts.population_labels.{lab}.positive")
        rep.add(L + "Neg", fmt_int(r["negative"]), f"{src}: prompts.population_labels.{lab}.negative")
        rep.add(L + "Labelled", fmt_int(r["positive"] + r["negative"]),
                f"{src}: prompts.population_labels.{lab}, positive plus negative")
    if p.get("filter_train_overlap_with_population") is not None:
        rep.add(f"{T}FilterTrainOverlap", fmt_int(p["filter_train_overlap_with_population"]),
                f"{src}: prompts.filter_train_overlap_with_population")
    f = stats["filter"]
    rec = f.get("recorded") or {}
    for key, name in (("tau", "Tau"), ("raw", "Raw"), ("graded", "Graded"), ("kept", "Kept"), ("train", "Train"),
                      ("val", "Val")):
        if rec.get(key) is not None:
            value = fmt(rec[key], 1) if key == "tau" else fmt_int(rec[key])
            rep.add(f"{T}Filter{name}", value, f"{src}: filter.recorded.{key} ({f.get('recorded_source')})")
    if f.get("kept_pct_of_graded") is not None:
        rep.add(f"{T}FilterKeptPct", fmt(f["kept_pct_of_graded"], 1), f"{src}: filter.kept_pct_of_graded")
    if f.get("scores"):
        s = f["scores"]
        rep.add(f"{T}FilterTrainSplit", fmt_int(s["n_train_split"]), f"{src}: filter.scores.n_train_split (seed {s['seed']})")
        rep.add(f"{T}FilterMinScore", fmt(s["min"], 2), f"{src}: filter.scores.min")
    rec = stats["recipe"]
    if rec.get("max_steps"):
        rep.add(f"{T}Steps", fmt_int(rec["max_steps"]), f"{src}: recipe.max_steps")


def _validation(rep, n: Names, T: str, stats: dict, src: str) -> None:
    """Physician agreement (scripts/compute_irr_kappa.py writes the file the analysis reads)."""
    k = stats["validation"]
    if "not_measured" in k:
        return
    where = f"{src}: validation"
    rep.add(f"{T}KappaRaters", fmt_int(k["n_raters"]), f"{where}.n_raters (physicians who returned grades)")
    for key, v in sorted(k["rows"].items()):
        rep.add(f"{T}KappaRows{token(key)}", fmt_int(v), f"{where}.rows.{key} (criterion rows)")
    for key, v in sorted(k["unsure_consensus"].items()):
        rep.add(f"{T}KappaUnsure{token(key)}", fmt_int(v), f"{where}.unsure_consensus.{key} (left out of kappa)")
    for grader, res in sorted(k["llm_vs_physician"].items()):
        G = f"{T}Kappa{token(grader)}"
        subsets = [(token(s), s, res[s]) for s in ("all", "positive", "negative") if s in res]
        subsets += [(n.cond_token(condition_from_legacy(c)), f"by_config.{c}", r)
                    for c, r in sorted(res.get("by_config", {}).items())]
        for S, path, r in subsets:
            w = f"{where}.llm_vs_physician.{grader}.{path}"
            rep.add(f"{G}{S}", fmt(r["kappa"], 2), f"{w}.kappa (Cohen's kappa against the physician)")
            rep.add(f"{G}{S}CI", fmt_ci(r["ci95"], 2), f"{w}.ci95, prompt-clustered bootstrap")
            rep.add(f"{G}{S}N", fmt_int(r["n"]), f"{w}.n (rows with a pass or fail from both)")
            if "n_prompts" in r:
                rep.add(f"{G}{S}Prompts", fmt_int(r["n_prompts"]), f"{w}.n_prompts")


# -- tables -----------------------------------------------------------------------------

def _notes(stats: dict, cell: CellConfig) -> str:
    """What a reader must know to read the cell's numbers."""
    seeds = stats["seeds"]
    notes = [f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''}"]
    mult = stats["counts"]["prompt_seed_multiplicity"]
    if len(seeds) > 1 and list(mult) == [str(len(seeds))]:
        notes.append(f"every seed evaluated the same {fmt_int(stats['counts']['unique_prompts'])} prompts")
    graded = min(stats["conditions"][k][c]["n_seeds"] for k in SEVEN for c in stats["conditions_order"])
    if graded < len(seeds):
        notes.append(f"dimension grades in {graded} of {len(seeds)} seeds")
    if cell.grading.same_family_waiver:
        notes.append("base model in the evaluator's family")
    if cell.report.set_aside:
        notes.append("set aside: " + escape(cell.report.set_aside.rstrip(".")))
    return "; ".join(notes)


def cell_table(stats: dict, cell: CellConfig, cfg: ReportConfig, src: str) -> str:
    """tab_cell_<token>: every condition and the two adapter contrasts (LoRA - Base,
    LoRA - Base+protocol) on the seven dimensions, the rubric score, the harm proxy, the
    label rates, the askers gap, format leakage, length and tokens."""
    n = Names(cfg)
    conds = stats["conditions_order"]
    # The adapter against the base and against the teacher prompt.
    ctrs = [k for k in stats["contrasts_order"] if k.startswith("lora - ")][:2]
    width = 1 + len(conds) + 2 * len(ctrs)
    header = ["", *(n.cond(c) for c in conds)]
    for k in ctrs:
        header += [n.ctr(k), "$p$"]

    def contrast_cells(r: dict, scale: float, dp: int, p_key: str) -> list[str]:
        return [fmt_pm(x(r.get("diff"), scale), scaled(r.get("ci95"), scale), dp), fmt_p(r.get(p_key))]

    def metric_row(m: str) -> list[str]:
        scale, dp = KIND_FORMAT[_kind(stats, m)]
        row = [n.metric(m)] + [fmt(x(stats["conditions"][m][c]["seed_mean"], scale), dp) for c in conds]
        for k in ctrs:
            row += contrast_cells(stats["contrasts"][k]["metrics"][m], scale, dp,
                                  "p_holm_within_contrast" if m in SEVEN else "p")
        return row

    title = [f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{escape(cell.report.label)}}} ({_notes(stats, cell)})}}"]
    seven = [metric_row(m) for m in SEVEN]
    scores = [metric_row(m) for m in ("rubric_score", "neg_any")]
    # Label rows in the order of the report config; a label the cell's benchmark lacks
    # has no rows here (its reason is in stats.json).
    labels = []
    for spec in cfg.discrimination:
        for lab in spec.labels:
            r = stats["discrimination"].get(spec.response, {}).get(lab, {"not_measured": "absent"})
            if "not_measured" in r:
                continue
            name = n.metric(spec.response).split(" (")[0]
            # A post hoc label is exploratory (decision D9), and its rows say so.
            when = ", post hoc" if r["exploratory"] else ""
            idx = [f"{name}, index for the label {n.label(lab)} (pp{when})"]
            neg = [f"{name}, prompts without the label {n.label(lab)} (\\%{when})"]
            for c in conds:
                idx.append(fmt(x(r["conditions"][c]["discrimination"], 100), 1, sign=True))
                neg.append(fmt(x(r["conditions"][c]["negative_rate"], 100), 1))
            for k in ctrs:
                idx += [fmt(x(r["contrasts"][k]["diff"], 100), 1, sign=True), MISSING]
                neg += contrast_cells(r["contrasts"][k]["negative_rate"]["seed_test"], 100, 1, "p")
            labels += [idx, neg]
    other = [["Askers gap (aggregate score)"] + [fmt(stats["askers_gap"].get(c, {}).get("gap"), 3, sign=True)
                                                 for c in conds] + [MISSING] * (2 * len(ctrs))]
    for col, r in stats["leakage"].items():
        other.append([f"Format leak, {escape(r['rule'])} (\\%)"]
                     + [fmt(x(r["conditions"][c]["leaked_share"], 100), 1) for c in conds]
                     + [MISSING] * (2 * len(ctrs)))
    size = [metric_row(m) for m in ("chars", "tokens") if m in stats["metrics"]]
    eng = stats["cost"]["conditions"]
    size.append(["Reply tokens (engine, one-pass conditions)"]
                + [fmt(eng[c]["tokens_engine_mean"], 0) if eng[c]["tokens_engine_n"] else MISSING for c in conds]
                + [MISSING] * (2 * len(ctrs)))
    blocks = [[title], seven, scores, labels, other, size]
    return tabular("l" + "r" * (width - 1), [header], [b for b in blocks if b], src)


def _mean_sd(r: dict, scale: float, dp: int) -> str:
    """A seed mean with the SD across seeds as a subscript."""
    if not finite(r.get("seed_mean")):
        return MISSING
    sd = f"_{{\\pm{r['seed_sd'] * scale:.{dp}f}}}" if finite(r.get("seed_sd")) else ""
    return f"${r['seed_mean'] * scale:.{dp}f}{sd}$"


def detail_tables(stats: dict, cell: CellConfig, cfg: ReportConfig, src: str) -> dict[str, str]:
    """The detail tables of one cell, tab_<kind>_<token>: what the paper prints for the
    cell of the submitted version, from the same stats.json as every other number."""
    n = Names(cfg)
    T = cell.report.token
    conds, seeds = stats["conditions_order"], stats["seeds"]
    cond, per = stats["conditions"], stats["counts"]["per_condition"]
    ns = len(seeds)
    own = f"base+{cell.protocol}"
    # One column per condition, whatever the cell's conditions are (four in the 2x2
    # design, more where prompt baselines run); a table that needs a condition the cell
    # lacks is left out instead of printing under the wrong header.
    by_cond = "c" * len(conds)
    out = {}

    def label(m: str) -> str:
        return n.metric(m)

    # The rubric score of each condition, with its t interval and the completions.
    rows = []
    for c in conds:
        r, k = cond["rubric_score"][c], per[c]
        cps = list(k["completions_per_seed"].values())
        lo, hi = r["seed_ci95_t"]
        rows.append([n.cond(c), f"{r['seed_mean']:.3f}", f"[{lo:.3f},\\;{hi:.3f}]", fmt_int(k["completions"]),
                     str(min(cps)) if min(cps) == max(cps) else f"{min(cps)}--{max(cps)}"])
    out[f"tab_conditions_{T}"] = tabular(
        "lcccc", [["Condition", "Score (mean)", f"95\\% CI ($t_{{{ns - 1}}}$)", "$n$ (pooled)", "$n$/seed"]],
        [rows], f"{src}: conditions.rubric_score, counts")

    rows = [[n.cond(c)] + [f"{cond['rubric_score'][c]['per_seed'][str(s)]:.3f}" for s in seeds] for c in conds]
    out[f"tab_per_seed_{T}"] = tabular("l" + "c" * ns, [["Condition"] + [f"Seed {s}" for s in seeds]], [rows],
                                       f"{src}: conditions.rubric_score.<condition>.per_seed")

    blocks = []
    for keys in DIMENSION_ROWS:
        blocks.append([[label(k) + " " + DIRECTION.get(k, "$\\uparrow$")]
                       + [_mean_sd(cond[k][c], *KIND_FORMAT[_kind(stats, k)]) for c in conds] for k in keys])
    blocks.append([["Scored responses (pooled)"] + [fmt_int(per[c]["epistemic_scored"]) for c in conds]])
    out[f"tab_dimensions_{T}"] = tabular("l" + by_cond, [["Dimension"] + [n.cond(c) for c in conds]], blocks,
                                         f"{src}: conditions (mean of seed means, SD across seeds)")

    blocks = []
    for k in stats["contrasts_order"]:
        block = [[f"\\multicolumn{{7}}{{l}}{{\\emph{{{n.ctr(k)}}}}}"]]
        for m in (*SEVEN, "rubric_score"):
            r = stats["contrasts"][k]["metrics"][m]
            scale, _ = KIND_FORMAT[_kind(stats, m)]
            ddp = DIFF_DECIMALS[_kind(stats, m)]
            block.append([label(m), fmt(x(r["diff"], scale), ddp, sign=True), fmt_ci(scaled(r["ci95"], scale), ddp),
                          fmt(r.get("t"), 2), fmt_p(r.get("p")),
                          fmt_p(r.get("p_holm_within_contrast")) if m in SEVEN else MISSING,
                          fmt(r.get("pooled_d"), 2, sign=True) + " " + fmt_ci(r.get("d_ci95"), 2)])
        blocks.append(block)
    out[f"tab_contrasts_{T}"] = tabular(
        "lcccccc", [["Dimension", "Diff.", f"95\\% CI ($t_{{{ns - 1}}}$)", f"$t_{{{ns - 1}}}$", "$p$",
                     "$p_{\\text{Holm}}$", "Pooled $d$ [bootstrap 95\\% CI]"]], blocks,
        f"{src}: contrasts (seed-level paired t; Holm over the seven dimensions within each contrast; "
        f"d interval from {stats['bootstrap']['resamples']} prompt-clustered bootstrap resamples)")

    # The seed-level t interval next to the prompt-clustered bootstrap interval.
    heads = (("lora - base", "active_inquiry"), (f"lora - {own}", "active_inquiry"),
             (f"lora+{cell.protocol} - lora", "red_flag"), (f"lora+{cell.protocol} - lora", "red_flag_rate"),
             ("lora - base", "rubric_score"))
    rows = []
    for k, m in heads:
        r = (stats["contrasts"].get(k) or {}).get("metrics", {}).get(m)
        if r is None:
            continue
        scale, _ = KIND_FORMAT[_kind(stats, m)]
        ddp = DIFF_DECIMALS[_kind(stats, m)]
        rows.append([f"{n.ctr(k)}, {label(m)}", fmt(x(r["diff"], scale), ddp, sign=True),
                     fmt_ci(scaled(r["ci95"], scale), ddp), fmt_ci(scaled(r.get("diff_ci95"), scale), ddp)])
    out[f"tab_bootstrap_{T}"] = tabular(
        "lrrr", [["Contrast", "Effect", f"95\\% CI, seeds ($t_{{{ns - 1}}}$)", "95\\% CI, prompt bootstrap"]],
        [rows], f"{src}: contrasts (effect = mean of seed-level differences; the bootstrap interval is for the "
                "pooled difference)")

    lq = stats["length_quartiles"]
    if "conditions" in lq:
        rate_rows, n_rows = [], []
        for c in conds:
            r = lq["conditions"][c]
            rate_rows.append([n.cond(c)] + [f"{100 * q['rate']:.1f}" if finite(q["rate"]) else MISSING
                                            for q in r["quartiles"]] + [f"{100 * r['overall']:.1f}"])
            n_rows.append([f"$n$, {n.cond(c)}"] + [fmt_int(q["n"]) for q in r["quartiles"]] + [fmt_int(r["n"])])
        out[f"tab_length_quartiles_{T}"] = tabular(
            "lrrrrr", [["Condition", "Q1", "Q2", "Q3", "Q4", "Overall"]], [rate_rows, n_rows],
            f"{src}: length_quartiles (active-inquiry %, pooled scored responses; {lq['reference']} quartiles)")

    axes = stats["axis_scores"]
    if axes:
        rows = [[escape(a)] + [f"{axes[a][c]:.3f}" for c in conds] for a in axes]
        out[f"tab_axes_{T}"] = tabular("l" + "r" * len(conds), [["Axis"] + [n.cond(c) for c in conds]], [rows],
                                       f"{src}: axis_scores (mean over pooled completions)")

    if {"base", "lora", own} <= set(conds):
        gap_header = f"$|${n.cond('lora')}$-${n.cond(own)}$|$"
        rows = []
        def plain(v) -> str:
            return f"{v:.3f}" if finite(v) else MISSING

        for m in GAP_METRICS:
            mean = {c: cond[m][c]["seed_mean"] for c in conds}
            both = finite(mean["lora"]) and finite(mean["base"]) and finite(mean[own])
            rows.append([label(m).replace("(\\%)", "(rate)")] + [plain(mean[c]) for c in conds]
                        + [plain(abs(mean["lora"] - mean[own]) if both else None),
                           fmt(mean["lora"] - mean["base"] if both else None, 3, sign=True)])
        out[f"tab_gap_{T}"] = tabular("l" + "r" * (len(conds) + 2),
                                      [["Dimension"] + [n.cond(c) for c in conds]
                                       + [gap_header, f"{n.cond('lora')}$-${n.cond('base')}"]], [rows],
                                      f"{src}: conditions (mean of seed means; rates as proportions)")

    rows = []
    for c in conds:
        k = per[c]
        rows.append([n.cond(c), fmt_int(k["completions"]), fmt_int(k["epistemic_scored"]),
                     f"{fmt_int(k['epistemic_parse_failures'])} ({100 * k['epistemic_parse_failure_rate']:.2f}\\%)",
                     fmt_int(k["epistemic_missing"]), f"{k['rubric_parse_failures']}/{fmt_int(k['rubric_items'])}"])
    out[f"tab_counts_{T}"] = tabular(
        "lrrrrr", [["Condition", "Responses", "Graded", "Epistemic parse fail.", "Grade errors",
                    "Rubric criterion parse fail."]], [rows],
        f"{src}: counts (completions pooled over seeds; grade errors have no epistemic row)")

    bd = stats["blanket_disclaimers"]
    rows = [[n.cond(c), f"{bd[c]['count']}/{fmt_int(bd[c]['n'])}", f"{100 * bd[c]['count'] / bd[c]['n']:.1f}",
             fmt_p(bd[c].get("fisher_p_vs_base")) if c != "base" else MISSING] for c in conds]
    out[f"tab_disclaimers_{T}"] = tabular("lrrr", [["Condition", "Blanket disclaimers", "\\%",
                                                    "Fisher $p$ vs Base"]], [rows], f"{src}: blanket_disclaimers")

    if "lora - base" in stats["contrasts"]:
        rows = []
        for m in (*SEVEN, "blanket_disclaimer", "rubric_score"):
            scale, _ = KIND_FORMAT[_kind(stats, m)]
            ddp = DIFF_DECIMALS[_kind(stats, m)]
            rep_ = stats["contrasts"]["lora - base"]["metrics"][m]
            wc = (stats["worst_case"].get(m) or {}).get("contrasts", {}).get("lora - base") or {}
            rows.append([label(m), fmt(x(rep_.get("diff"), scale), ddp, sign=True), fmt_p(rep_.get("p")),
                         fmt(x(wc.get("diff"), scale), ddp, sign=True), fmt_p(wc.get("p"))])
        out[f"tab_worst_case_{T}"] = tabular(
            "lrrrr", [["Dimension", f"{n.cond('lora')}$-${n.cond('base')}", "$p$", "Worst case", "$p$"]], [rows],
            f"{src}: contrasts and worst_case (worst case: failed or missing grades take the worst value)")

    themes = stats["themes"]
    if "not_measured" not in themes:
        # Most evaluated prompts first, as the analysis chose them; stats.json keeps keys sorted.
        order = sorted(themes, key=lambda th: (-themes[th]["n_evaluated"], th))
        rows = [[f"\\texttt{{{escape(th)}}}", fmt_int(themes[th]["n_evaluated"])]
                + [f"{100 * themes[th]['conditions'][c]['rate']:.1f}" for c in conds] for th in order]
        out[f"tab_themes_{T}"] = tabular("lr" + "r" * len(conds), [["Theme", "$n$ prompts"] + [n.cond(c) for c in conds]], [rows],
                                         f"{src}: themes (n = unique evaluated prompts)")

    scores = (stats["filter"] or {}).get("scores")
    if scores:
        rows = [[f"{r['tau']:.1f}", f"{100 * r['share_of_train_split']:.1f}",
                 f"{100 * r['share_of_pool_estimate']:.1f}" if finite(r.get("share_of_pool_estimate")) else MISSING]
                for r in scores["tau_grid"]]
        out[f"tab_tau_{T}"] = tabular(
            "ccc", [["$\\tau$", f"\\% of seed-{scores['seed']} train split", "\\% of pool (est.)"]], [rows],
            f"{src}: filter.scores (pool column: the train split's shares times the recorded kept share, "
            f"{(stats['filter'].get('recorded_source') or 'no record')})")

    v = stats["validation"]
    if "not_measured" not in v:
        for grader, res in sorted(v["llm_vs_physician"].items()):
            def krow(name, r):
                return [name, fmt(r["kappa"], 2), fmt_ci(r["ci95"], 2), f"{100 * r['agree']:.0f}\\%", fmt_int(r["n"]),
                        f"{100 * r['physician_pass']:.0f}\\%", f"{100 * r['llm_pass']:.0f}\\%"]
            subsets = [b for b in ([krow(name, res[s]) for s, name in (("all", "All criteria"),
                                                                        ("positive", "Positive (meet)"),
                                                                        ("negative", "Negative (avoid)")) if s in res],
                                   [krow(n.cond(condition_from_legacy(c)), res["by_config"][c])
                                    for c in sorted(res.get("by_config", {}),
                                                    key=lambda c: conds.index(condition_from_legacy(c)))]) if b]
            out[f"tab_kappa_{token(grader)}_{T}"] = tabular(
                "lcccccc", [["Subset", "$\\kappa$", "95\\% CI", "Agree", "$n$", "Phys.\\ pass", "LLM pass"]], subsets,
                f"{src}: validation.llm_vs_physician.{grader} (CI: prompt-clustered bootstrap)")
    return out
