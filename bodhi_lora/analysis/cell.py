"""Every statistic of one cell, from its tables (docs/ARCHITECTURE.md section 7).

The same code runs on every cell. A statistic is absent only when its input is absent
(no grades, no labels, no tokenizer, a field an older file lacks), and the output then
says "not measured" with the reason. Nothing here names a cell, a model or a benchmark.

Statistical unit (decision D7): the seed. A seed-level summary or test uses the seeds
that have a value (for a contrast, in both conditions); a seed without one (for example
a seed whose dimension grading found none of its prompts) leaves it, the degrees of
freedom fall, n_seeds and seeds_missing say so, and the per-seed values show the gap
(docs/ANALYSIS_PLAN.md section 2.3). With fewer than two seeds there is no interval and
no test. Pooled numbers are descriptive, with intervals from one prompt-clustered
bootstrap per cell (stats.ClusterBootstrap).

Contrasts come from the cell's conditions by rule: every condition minus base; lora
minus base+<protocol>; lora+<protocol> minus lora; lora minus each base+<baseline>
(a prompt protocol that is not the cell's protocol). The cell record may add others
(report.contrasts), each one the analysis plan names. Holm runs over the seven
dimensions within each contrast, and over the seven dimensions of every contrast.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bodhi_lora.analysis import stats as st
from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.analysis.extract import EPISTEMIC, SEVEN, read_table
from bodhi_lora.paths import STAGES, CellPaths, Project
from bodhi_lora.registry import CellConfig, Registry, parse_condition

TAU_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# The dimension columns that are not 0-2 grades.
EPISTEMIC_KIND = {"active_inquiry": "rate", "questions": "count", "red_flag_rate": "rate",
                  "scope_bounded_rate": "rate", "specificity_rate": "rate", "blanket_disclaimer": "rate",
                  "appropriate_hedging": "rate"}


@dataclass(frozen=True)
class Metric:
    key: str
    # score02 (0-2 grade), rate (0/1), count, score (0-1 rubric score), chars, tokens
    kind: str
    # epistemic: scored replies only. response: every completion.
    source: str
    # The value a failed or missing grade takes in the worst case, or the column that
    # holds the worst-case value. None: no worst case.
    worst: object = None
    # Secondary metrics get condition summaries and seed-level contrasts, and stay out
    # of the bootstrap and the pooled effect sizes.
    secondary: bool = False


def metrics_of(table: pd.DataFrame, cfg: ReportConfig) -> list[Metric]:
    """The metrics of a cell: the dimensions and rates, the rubric score, length, the
    harm proxy, and each rubric axis in the table."""
    out = [Metric(k, EPISTEMIC_KIND.get(k, "score02"), "epistemic",
                  # Lower is better, so the worst case counts a failed grade as a disclaimer.
                  worst=1.0 if k == "blanket_disclaimer" else 0.0) for k in EPISTEMIC]
    out += [Metric("rubric_score", "score", "response", worst="rubric_score_worst"),
            Metric("chars", "chars", "response"),
            Metric("tokens", "tokens", "response", secondary=True),
            Metric("neg_any", "rate", "response", secondary=True),
            Metric("neg_points_lost", "count", "response", secondary=True)]
    out += [Metric(c, "score", "response", secondary=True) for c in sorted(table.columns) if c.startswith("axis_")]
    return out


def contrasts_of(cell: CellConfig, reg: Registry) -> list[tuple[str, str]]:
    """(a, b) pairs, a minus b, by the rule in the module docstring, then the cell
    record's own (report.contrasts)."""
    conds = list(cell.eval.conditions)
    out = [(c, "base") for c in conds if c != "base" and "base" in conds]
    own = f"base+{cell.protocol}"
    if "lora" in conds and own in conds:
        out.append(("lora", own))
    if f"lora+{cell.protocol}" in conds and "lora" in conds:
        out.append((f"lora+{cell.protocol}", "lora"))
    for c in conds:
        w, p = parse_condition(c)
        if w == "base" and p and p != cell.protocol and "lora" in conds:
            out.append(("lora", c))
    for k in cell.report.contrasts:
        a, _, b = k.partition(" - ")
        out.append((a.strip(), b.strip()))
    seen, uniq = set(), []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    return uniq


def contrast_key(a: str, b: str) -> str:
    return f"{a} - {b}"


def _f(x) -> Optional[float]:
    """A float for JSON, None when not finite."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class CellData:
    """A cell's response table, its prompts table and the frames the statistics read."""

    def __init__(self, table: pd.DataFrame, prompts: pd.DataFrame, cell: CellConfig):
        self.cell = cell
        self.conditions = list(cell.eval.conditions)
        self.seeds = sorted(int(s) for s in table["seed"].unique())
        table = table.copy()
        table["neg_any"] = (table["neg_items_met"] > 0).astype(float)
        self.table = table
        self.prompts = prompts
        self.scored = table[table["epi_status"] == "scored"]

    def frame(self, m: Metric, worst_case: bool = False, rows: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Rows that carry the metric, with the value in column v.

        Normal: dimensions use scored replies only, other metrics every completion.
        Worst case: every graded completion counts, and a failed or missing grade takes
        the metric's worst value. A reply that was never graded (not_graded) stays out.
        """
        t = self.table if rows is None else rows
        if m.source == "response":
            col = m.worst if (worst_case and isinstance(m.worst, str)) else m.key
            out = t[["seed", "condition", "prompt_id", col]].rename(columns={col: "v"})
            return out[out["v"].notna()]
        if not worst_case:
            sub = t[t["epi_status"] == "scored"]
            return sub[["seed", "condition", "prompt_id", m.key]].rename(columns={m.key: "v"})
        sub = t[t["epi_status"] != "not_graded"]
        out = sub[["seed", "condition", "prompt_id", m.key]].rename(columns={m.key: "v"}).copy()
        out["v"] = out["v"].fillna(m.worst)
        return out

    def seed_means(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Seed x condition means; NaN where a seed has no value."""
        t = frame.groupby(["seed", "condition"])["v"].mean().unstack("condition")
        return t.reindex(index=self.seeds, columns=self.conditions)


def _missing_seeds(values, seeds) -> list[int]:
    return [s for s, v in zip(seeds, values) if not np.isfinite(v)]


def summarize(values: np.ndarray, seeds: list[int], pooled: pd.Series) -> dict:
    """Seed-level summary of one metric in one condition, over the seeds with a value."""
    missing = _missing_seeds(values, seeds)
    have = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    out = {"per_seed": {str(s): _f(v) for s, v in zip(seeds, values)},
           "pooled_mean": _f(pooled.mean()) if len(pooled) else None, "n_pooled": int(pooled.count()),
           "n_seeds": int(len(have)), "seeds_missing": missing}
    if not len(have):
        out.update({"seed_mean": None, "seed_sd": None, "seed_ci95_t": [None, None],
                    "not_measured": f"no value in seed(s) {missing}" if missing else "no seeds"})
        return out
    mean, lo, hi = st.t_interval(have)
    out.update({"seed_mean": _f(mean), "seed_sd": _f(np.std(have, ddof=1)) if len(have) > 1 else None,
                "seed_ci95_t": [_f(lo), _f(hi)]})
    return out


def paired(sm: pd.DataFrame, a: str, b: str, seeds: list[int]) -> dict:
    """Seed-level paired test a - b over the seeds where both conditions have a value."""
    va, vb = sm[a].to_numpy(dtype=float), sm[b].to_numpy(dtype=float)
    missing = sorted(set(_missing_seeds(va, seeds)) | set(_missing_seeds(vb, seeds)))
    if len(missing) == len(seeds):
        return {"not_measured": f"no value in seed(s) {missing}", "n_seeds": 0, "seeds_missing": missing}
    r = st.paired_test(va, vb)
    r["seeds_missing"] = missing
    if r["n_seeds"] < 2:
        r["not_measured_test"] = "one seed: no interval and no test"
    return r


# -- sections ---------------------------------------------------------------------------

def conditions_section(d: CellData, metrics: list[Metric]) -> dict:
    res = {}
    for m in metrics:
        frame = d.frame(m)
        sm = d.seed_means(frame)
        res[m.key] = {c: summarize(sm[c].to_numpy(dtype=float), d.seeds, frame.loc[frame["condition"] == c, "v"])
                      for c in d.conditions}
    return res


def contrasts_section(d: CellData, metrics: list[Metric], pairs, boot: st.ClusterBootstrap) -> dict:
    res = {}
    for a, b in pairs:
        key = contrast_key(a, b)
        res[key] = {"a": a, "b": b, "metrics": {}}
        for m in metrics:
            frame = d.frame(m)
            r = paired(d.seed_means(frame), a, b, d.seeds)
            fa = frame[frame["condition"] == a]
            fb = frame[frame["condition"] == b]
            if not m.secondary and len(fa) > 1 and len(fb) > 1:
                r["pooled_diff"] = _f(fa["v"].mean() - fb["v"].mean())
                r["pooled_d"] = _f(st.cohens_d(fa["v"], fb["v"]))
                # Descriptive only: the submitted paper's test, which treats pooled replies
                # as independent. Kept so old text can be checked.
                r["pooled_welch_p"] = _f(st.welch_p(fa["v"], fb["v"]))
                r.update(boot.diff_and_d(fa, fb))
            res[key]["metrics"][m.key] = r
        family = [res[key]["metrics"][k].get("p", float("nan")) for k in SEVEN]
        for k, ph in zip(SEVEN, st.holm(family)):
            res[key]["metrics"][k]["p_holm_within_contrast"] = _f(ph)
        # The rubric axes are one family of their own (ANALYSIS_PLAN.md section 4, E05):
        # Holm over the axes the cell's rubric has, within the contrast.
        axes = [m.key for m in metrics if m.key.startswith("axis_")]
        for k, ph in zip(axes, st.holm([res[key]["metrics"][k].get("p", float("nan")) for k in axes])):
            res[key]["metrics"][k]["p_holm_axes"] = _f(ph)
    every = [(k2, k) for k2 in res for k in SEVEN]
    for (k2, k), ph in zip(every, st.holm([res[k2]["metrics"][k].get("p", float("nan")) for k2, k in every])):
        res[k2]["metrics"][k]["p_holm_all_contrasts"] = _f(ph)
    return res


def counts_section(d: CellData, meta: list[dict]) -> dict:
    t = d.table
    res = {"per_condition": {}}
    for c in d.conditions:
        sub = t[t["condition"] == c]
        status = sub["epi_status"].value_counts()
        drawn = sum(m["n_examples"][c] for m in meta)
        graded_rows = int(status.get("scored", 0) + status.get("parse_failure", 0))
        r = {"completions": int(len(sub)),
             "completions_per_seed": {str(s): int((sub["seed"] == s).sum()) for s in d.seeds},
             "drawn": int(drawn),
             "completion_drop_rate": _f(1 - len(sub) / drawn) if drawn else None,
             "epistemic_rows": graded_rows,
             "epistemic_scored": int(status.get("scored", 0)),
             "epistemic_parse_failures": int(status.get("parse_failure", 0)),
             "epistemic_missing": int(status.get("missing", 0)),
             "epistemic_not_graded": int(status.get("not_graded", 0)),
             "epistemic_parse_failure_rate": _f(status.get("parse_failure", 0) / graded_rows) if graded_rows else None,
             "epistemic_unscored_rate": _f(1 - status.get("scored", 0) / len(sub)) if len(sub) else None,
             "rubric_items": int(sub["rubric_items"].sum()),
             "rubric_parse_failures": int(sub["rubric_parse_failures"].sum())}
        r["rubric_parse_failure_rate"] = _f(r["rubric_parse_failures"] / r["rubric_items"]) if r["rubric_items"] else None
        per_run = sub.groupby("seed")[["rubric_parse_failures", "rubric_items"]].sum()
        r["rubric_parse_failure_rate_max_run"] = _f((per_run["rubric_parse_failures"] / per_run["rubric_items"]).max())
        res["per_condition"][c] = r
    # Prompts: unique ids over every condition's completions, and how many seeds drew each.
    per_prompt = t.groupby("prompt_id")["seed"].nunique()
    res["seeds"] = d.seeds
    res["unique_prompts"] = int(per_prompt.size)
    res["prompt_seed_multiplicity"] = {str(k): int(v) for k, v in sorted(Counter(per_prompt.tolist()).items())}
    res["prompts_in_more_than_one_seed"] = int((per_prompt > 1).sum())
    draws = {c: res["per_condition"][c]["drawn"] for c in d.conditions}
    res["prompt_draws_per_condition"] = max(draws.values()) if draws else 0
    graded = [r for r in res["per_condition"].values() if r["epistemic_parse_failure_rate"] is not None]
    res["epistemic_parse_failure_rate_max"] = _f(max(r["epistemic_parse_failure_rate"] for r in graded)) if graded else None
    # None on a benchmark without a rubric, where no condition has rubric items.
    rubric_runs = [r["rubric_parse_failure_rate_max_run"] for r in res["per_condition"].values()
                   if r["rubric_parse_failure_rate_max_run"] is not None]
    res["rubric_parse_failure_rate_max_run"] = _f(max(rubric_runs)) if rubric_runs else None
    res["seeds_graded"] = sorted(int(s) for s in d.scored["seed"].unique())
    return res


def worst_case_section(d: CellData, metrics: list[Metric], pairs) -> dict:
    res = {}
    for m in metrics:
        if m.worst is None:
            continue
        sm = d.seed_means(d.frame(m, worst_case=True))
        # pandas mean skips the seeds without a value, as summarize() does.
        res[m.key] = {"conditions": {c: _f(sm[c].mean()) if sm[c].notna().any() else None for c in d.conditions},
                      "contrasts": {contrast_key(a, b): paired(sm, a, b, d.seeds) for a, b in pairs}}
    return res


def below_cap_section(d: CellData, metrics: list[Metric], pairs) -> dict:
    """Every contrast again on the prompts where neither reply of the pair stopped at the
    token cap (docs/ANALYSIS_PLAN.md section 12, 2026-10-23): the same seed means, paired t
    over seeds and Holm family within the contrast as contrasts_section. A pair whose stop
    reason is unknown for either reply stays out, and so does a prompt that has a reply in
    only one of the two conditions (the primary keeps it in that condition's mean). Decides
    no claim."""
    t = d.table
    if "finish_length" not in t or t["finish_length"].isna().all():
        return {"not_measured": "the eval stored no finish_length"}
    res = {}
    for a, b in pairs:
        key = contrast_key(a, b)
        two = t[t["condition"].isin([a, b])]
        per = two.assign(below=two["finish_length"] == 0).groupby(["seed", "prompt_id"])["below"]
        both = per.size() == 2
        keep = per.all()[both]
        keep = keep[keep].index
        rows = two.set_index(["seed", "prompt_id"])
        rows = rows[rows.index.isin(keep)].reset_index()
        res[key] = {"a": a, "b": b, "pairs": int(both.sum()), "pairs_below_cap": int(len(keep)),
                    "pairs_below_cap_per_seed": {str(s): int(sum(1 for k in keep if k[0] == s)) for s in d.seeds},
                    "metrics": {m.key: paired(d.seed_means(d.frame(m, rows=rows)), a, b, d.seeds) for m in metrics}}
        family = [res[key]["metrics"][k].get("p", float("nan")) for k in SEVEN]
        for k, ph in zip(SEVEN, st.holm(family)):
            res[key]["metrics"][k]["p_holm_within_contrast"] = _f(ph)
    return {"contrasts": res}


def disclaimers_section(d: CellData) -> dict:
    s = d.scored
    counts = {c: (int(s.loc[s["condition"] == c, "blanket_disclaimer"].sum()), int((s["condition"] == c).sum()))
              for c in d.conditions}
    res = {}
    kb, nb = counts.get("base", (0, 0))
    for c in d.conditions:
        k, n = counts[c]
        res[c] = {"count": k, "n": n}
        if c != "base" and n and nb:
            res[c]["fisher_p_vs_base"] = _f(st.fisher_p(k, n, kb, nb))
    flagged = s[s["blanket_disclaimer"] == 1]
    # The grader flags a blanket disclaimer and scores hedging 0-2 in the same grade;
    # how often a flagged disclaimer still gets the top hedging score.
    res["all_conditions"] = {"flagged": int(len(flagged)),
                             "flagged_hedging_2": int((flagged["hedging_quality"] == 2).sum())}
    return res


def length_quartiles_section(d: CellData, reference: str) -> dict:
    """Active inquiry by reply-length quartile, edges from the reference condition."""
    s = d.scored
    ref = s.loc[s["condition"] == reference, "chars"].to_numpy()
    if reference not in d.conditions or len(ref) < 4:
        return {"not_measured": f"no scored replies of {reference}"}
    # numpy's default (linear) quantile.
    q = np.quantile(ref, [0.25, 0.5, 0.75])
    res = {"reference": reference, "boundaries_chars": [float(x) for x in q],
           "first_length_of_quartile": [int(math.ceil(x)) for x in q], "conditions": {}}
    for c in d.conditions:
        sub = s[s["condition"] == c]
        # Right-open bins: Q1 is len < q1, Q4 is len >= q3.
        bins = np.digitize(sub["chars"].to_numpy(), q, right=False)
        ai = sub["active_inquiry"].to_numpy()
        rows = [{"n": int((bins == i).sum()), "rate": _f(ai[bins == i].mean()) if (bins == i).any() else None}
                for i in range(4)]
        res["conditions"][c] = {"quartiles": rows, "overall": _f(ai.mean()) if len(ai) else None, "n": int(len(sub))}
    return res


def cost_section(d: CellData, reg: Registry, pairs, meta: list[dict]) -> dict:
    """Reply length and token cost per condition, from the stored replies.

    tokens: the model's tokenizer on the final reply. tokens_engine: the count the
    serving engine stored (single-call conditions). calls_design: the protocol's passes,
    a design fact, not a measurement. calls and tokens_total: measured per reply by the
    package's eval stage; older files lack them, and the first pass of a multi-pass
    protocol was not stored, so its tokens are not in tokens.
    """
    t = d.table
    # What counted the tokens column, or why nothing did (extract.local_tokenizer).
    res = {"conditions": {}, "contrasts": {}, "tokenizer": sorted({m["tokenizer"] for m in meta if "tokenizer" in m})}
    for c in d.conditions:
        sub = t[t["condition"] == c]
        _, proto = parse_condition(c)
        r = {"chars_mean": _f(sub["chars"].mean()), "n": int(len(sub)),
             "chars_mean_scored": _f(sub.loc[sub["epi_status"] == "scored", "chars"].mean()),
             "calls_design": reg.protocols[proto].passes if proto else 1}
        for col in ("tokens", "tokens_engine", "calls", "tokens_total"):
            v = sub[col].dropna()
            r[f"{col}_mean"] = _f(v.mean()) if len(v) else None
            r[f"{col}_n"] = int(len(v))
        res["conditions"][c] = r
    for a, b in pairs:
        ra, rb = res["conditions"][a], res["conditions"][b]
        out = {}
        for col in ("chars", "tokens", "tokens_total"):
            x, y = ra[f"{col}_mean"], rb[f"{col}_mean"]
            out[f"{col}_diff"] = _f(x - y) if x is not None and y is not None else None
            out[f"{col}_pct"] = _f(100 * (x - y) / y) if x is not None and y else None
        res["contrasts"][contrast_key(a, b)] = out
    # The tokenizer count against the engine's, on the replies that have both: a check
    # that the tokenizer is the one the engine used (the engine also counts the end token).
    both = t[t["tokens"].notna() & t["tokens_engine"].notna()]
    if len(both):
        gap = (both["tokens_engine"] - both["tokens"]).to_numpy()
        res["tokenizer_check"] = {"n": int(len(gap)), "median_engine_minus_tokenizer": _f(np.median(gap)),
                                  "share_within_one_of_median": _f(np.mean(np.abs(gap - np.median(gap)) <= 1))}
    else:
        res["tokenizer_check"] = {"n": 0}
    if t["tokens"].isna().all():
        res["not_measured_tokens"] = "no tokenizer count: the tokenizer was not in the local cache at extract"
    if t["calls"].isna().all():
        res["not_measured_calls"] = ("calls and total tokens per reply are not in these eval files; the first "
                                     "pass of a multi-pass protocol was not stored")
    return res


def shift_correlations_section(d: CellData, pairs) -> dict:
    """Per (seed, prompt): r between the a - b shift in active inquiry and in four other dimensions."""
    dims = ("active_inquiry", "context_seeking", "red_flag", "scope_bounding", "specificity")
    s = d.scored
    res = {}
    for a, b in pairs:
        fa = s[s["condition"] == a].set_index(["seed", "prompt_id"])[list(dims)]
        fb = s[s["condition"] == b].set_index(["seed", "prompt_id"])[list(dims)]
        delta = (fa - fb).dropna()
        res[contrast_key(a, b)] = {"n_pairs": int(len(delta)), "pearson_r_with_active_inquiry":
                                   {k: _f(st.pearson_r(delta["active_inquiry"], delta[k])) if len(delta) > 1 else None
                                    for k in dims[1:]}}
    return res


def context_seeking_anchor_section(d: CellData) -> dict:
    """How often a context-seeking grade of 2 comes with no direct question.

    The grader prompt says: score 2 only if at least one direct question is asked. The
    same grade carries the active-inquiry flag, so a 2 without it breaks the anchor.
    """
    s = d.scored
    res = {}
    for c in d.conditions:
        twos = s[(s["condition"] == c) & (s["context_seeking"] == 2)]
        k, n = int((twos["active_inquiry"] == 0).sum()), int(len(twos))
        res[c] = {"n_score_2": n, "n_score_2_no_question": k, "share_no_question": _f(k / n) if n else None}
    return res


def axis_section(d: CellData) -> dict:
    axes = sorted(c for c in d.table.columns if c.startswith("axis_"))
    t = d.table
    return {a.removeprefix("axis_"): {c: _f(t.loc[t["condition"] == c, a].mean()) for c in d.conditions}
            for a in axes}


def prompts_section(d: CellData, filter_train_ids: Optional[set], extract_rec: Optional[dict]) -> dict:
    """Counts of the eval population and the evaluated prompts. Label groups among the
    evaluated prompts only for the labels of prompts.csv (the report config's); every
    label's population counts from the extract record (the benchmark rows, no output)."""
    p = d.prompts
    ev = p[p["evaluated"] == 1]
    res = {"n_population": int(p["in_population"].sum()) if p["in_population"].notna().any() else None,
           "n_evaluated": int(len(ev)),
           "rubric_points_pos_min": _f(ev["rubric_points_pos"].min()),
           "rubric_points_pos_max": _f(ev["rubric_points_pos"].max()),
           "labels": {}}
    for col in sorted(c for c in p.columns if c.startswith("label_")):
        lab = col.removeprefix("label_")
        res["labels"][lab] = {
            "population": {"positive": int((p[col] == 1).sum()), "negative": int((p[col] == 0).sum()),
                           "unlabelled": int(p[col].isna().sum())},
            "evaluated": {"positive": int((ev[col] == 1).sum()), "negative": int((ev[col] == 0).sum()),
                          "unlabelled": int(ev[col].isna().sum())}}
    labels = ((extract_rec or {}).get("prompts") or {}).get("labels") or {}
    res["population_labels"] = {k: v["population"] for k, v in sorted(labels.items()) if "population" in v}
    if filter_train_ids is not None and res["n_population"]:
        res["filter_train_overlap_with_population"] = len(filter_train_ids & set(p.loc[p["in_population"] == 1, "prompt_id"]))
    return res


def themes_section(d: CellData) -> dict:
    """Active inquiry by theme, the five themes with most evaluated prompts."""
    p = d.prompts
    if p["theme"].isna().all():
        return {"not_measured": "no theme tags: the benchmark rows were not read at extract"}
    theme = dict(zip(p["prompt_id"], p["theme"]))
    s = d.scored.assign(theme=d.scored["prompt_id"].map(theme))
    ev = p[p["evaluated"] == 1]
    n_eval = ev["theme"].value_counts()
    n_pop = p.loc[p["in_population"] == 1, "theme"].value_counts()
    top = sorted(n_eval.index, key=lambda t: (-n_eval[t], t))[:5]
    res = {}
    for th in top:
        res[th] = {"n_evaluated": int(n_eval[th]), "n_population": int(n_pop.get(th, 0)), "conditions": {}}
        for c in d.conditions:
            sub = s[(s["condition"] == c) & (s["theme"] == th)]
            res[th]["conditions"][c] = {"rate": _f(sub["active_inquiry"].mean()) if len(sub) else None,
                                        "n": int(len(sub))}
    return res


def _group_rates(d: CellData, frame: pd.DataFrame, condition: str, boot: st.ClusterBootstrap) -> dict:
    """Rate of a 0/1 response in each label group of one condition: pooled with its
    bootstrap resamples, and per seed. frame holds columns seed, condition, prompt_id, v, g."""
    sc = frame[frame["condition"] == condition]
    out = {}
    for g, name in ((1.0, "positive"), (0.0, "negative")):
        grp = sc[sc["g"] == g]
        per_seed = [grp.loc[grp["seed"] == seed, "v"].mean() if (grp["seed"] == seed).any() else float("nan")
                    for seed in d.seeds]
        out[name] = {"rate": float(grp["v"].mean()) if len(grp) else float("nan"), "n": int(len(grp)),
                     "boot": boot.means(grp[["prompt_id", "v"]]), "per_seed": np.asarray(per_seed, dtype=float)}
    return out


def _seed_summary(values: np.ndarray, seeds: list[int]) -> dict:
    """Mean and t interval over the seeds with a value; the per-seed values."""
    have = values[np.isfinite(values)]
    mean, lo, hi = st.t_interval(have) if len(have) else (np.nan,) * 3
    return {"per_seed": {str(k): _f(v) for k, v in zip(seeds, values)}, "seed_mean": _f(mean),
            "seed_ci95_t": [_f(lo), _f(hi)], "n_seeds": int(len(have)),
            "seeds_missing": [k for k, v in zip(seeds, values) if not np.isfinite(v)]}


def discrimination_section(d: CellData, reg: Registry, cfg: ReportConfig, metrics: list[Metric], pairs,
                           boot: st.ClusterBootstrap) -> dict:
    """For each discrimination set of the report config: the rate of a 0/1 response among
    the label's positive and negative prompts, and the index P(response | positive) -
    P(response | negative), per condition.

    Pooled over seeds (every reply to a prompt counts), with a prompt-clustered bootstrap
    interval for each rate, the index and the odds ratio; per-seed values with a t
    interval over seeds. Each contrast a - b gets the difference of the indices and of
    each group's rate, with a bootstrap interval and the seed-level paired test. An
    evaluator response counts scored replies only; a response-level rate counts every
    completion (Metric.source).
    """
    bench = reg.benchmarks[d.cell.benchmark]
    by_key = {m.key: m for m in metrics}
    analyzed = cfg.labels_for(d.cell)
    res = {}
    for spec in cfg.discrimination:
        resp = spec.response
        if resp not in by_key or by_key[resp].kind != "rate":
            raise ValueError(f"discrimination response {resp!r} is not a 0/1 column of the response table")
        base = d.frame(by_key[resp])
        res[resp] = {}
        for lab in [x for x in spec.labels if x in analyzed]:
            col = f"label_{lab}"
            if lab not in bench.labels:
                res[resp][lab] = {"not_measured": f"benchmark {bench.id} has no label {lab}"}
                continue
            if col not in d.prompts:
                res[resp][lab] = {"not_measured": f"prompts.csv has no column {col}; run analyze --extract "
                                                  "where the cell's raw outputs are"}
                continue
            value = dict(zip(d.prompts["prompt_id"], d.prompts[col]))
            frame = base.assign(g=base["prompt_id"].map(value))
            frame = frame[frame["g"].notna()]
            groups = frame.groupby("g")["prompt_id"].nunique()
            if groups.get(1.0, 0) == 0 or groups.get(0.0, 0) == 0:
                res[resp][lab] = {"not_measured": "both groups are not present among the evaluated prompts"}
                continue
            out = {"timing": bench.labels[lab].timing, "source": bench.labels[lab].source,
                   "exploratory": bench.labels[lab].timing == "post_hoc",
                   "n_prompts": {"positive": int(groups[1.0]), "negative": int(groups[0.0])},
                   "conditions": {}, "contrasts": {}}
            rates = {c: _group_rates(d, frame, c, boot) for c in d.conditions}
            for c, r in rates.items():
                pos, neg = r["positive"], r["negative"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    or_boot = np.array([st.odds_ratio(x, y) for x, y in zip(pos["boot"], neg["boot"])])
                cond = {"positive_rate": _f(pos["rate"]), "negative_rate": _f(neg["rate"]),
                        "n_positive": pos["n"], "n_negative": neg["n"],
                        "positive_rate_ci95": st.percentile_ci(pos["boot"]),
                        "negative_rate_ci95": st.percentile_ci(neg["boot"]),
                        "discrimination": _f(pos["rate"] - neg["rate"]),
                        "discrimination_ci95": st.percentile_ci(pos["boot"] - neg["boot"]),
                        "odds_ratio": _f(st.odds_ratio(pos["rate"], neg["rate"])),
                        "odds_ratio_ci95": st.percentile_ci(or_boot)}
                cond.update(_seed_summary(pos["per_seed"] - neg["per_seed"], d.seeds))
                cond["negative_rate_seed"] = _seed_summary(neg["per_seed"], d.seeds)
                cond["positive_rate_seed"] = _seed_summary(pos["per_seed"], d.seeds)
                out["conditions"][c] = cond
            for a, b in pairs:
                ra, rb = rates[a], rates[b]
                ia = ra["positive"]["per_seed"] - ra["negative"]["per_seed"]
                ib = rb["positive"]["per_seed"] - rb["negative"]["per_seed"]
                ctr = {"diff": _f(out["conditions"][a]["discrimination"] - out["conditions"][b]["discrimination"]),
                       "diff_ci95": st.percentile_ci((ra["positive"]["boot"] - ra["negative"]["boot"])
                                                     - (rb["positive"]["boot"] - rb["negative"]["boot"])),
                       "seed_test": _seed_test(ia, ib, d.seeds)}
                for g in ("positive", "negative"):
                    ctr[f"{g}_rate"] = {"diff": _f(ra[g]["rate"] - rb[g]["rate"]),
                                        "diff_ci95": st.percentile_ci(ra[g]["boot"] - rb[g]["boot"]),
                                        "seed_test": _seed_test(ra[g]["per_seed"], rb[g]["per_seed"], d.seeds)}
                out["contrasts"][contrast_key(a, b)] = ctr
            res[resp][lab] = out
    return res


def _seed_test(a: np.ndarray, b: np.ndarray, seeds: list[int]) -> dict:
    """paired() on two per-seed arrays."""
    return paired(pd.DataFrame({"a": a, "b": b}), "a", "b", seeds)


def _leak_threshold(reg: Registry, col: str) -> int:
    pid, _, rid = col.removeprefix("leak_").partition("__")
    return reg.protocols[pid].leak_rules[rid].threshold


def leakage(d: CellData, reg: Registry, pairs) -> dict:
    """Per leak rule: the leak share of each condition (pooled over scored replies, and
    per seed), the dimensions by leak status, red flag by length quartile among clean
    replies, and each contrast of the per-seed leak shares (H6 secondary)."""
    cols = sorted(c for c in d.table.columns if c.startswith("leak_"))
    res = {}
    for col in cols:
        pid, _, rid = col.removeprefix("leak_").partition("__")
        thr = _leak_threshold(reg, col)
        out = {"protocol": pid, "rule": rid, "threshold": thr, "conditions": {}, "contrasts": {}}
        sc_all = d.scored.assign(v=(d.scored[col] >= thr).astype(float))
        share = d.seed_means(sc_all[["seed", "condition", "prompt_id", "v"]])
        for c in d.conditions:
            allc = d.table[d.table["condition"] == c]
            sc = d.scored[d.scored["condition"] == c]
            leaked = sc[sc[col] >= thr]
            clean = sc[sc[col] < thr]
            r = {"n_scored": int(len(sc)), "n_leaked": int(len(leaked)), "n_clean": int(len(clean)),
                 "leaked_share": _f(len(leaked) / len(sc)) if len(sc) else None,
                 "n_completions": int(len(allc)), "n_leaked_completions": int((allc[col] >= thr).sum()),
                 "chars_median_leaked": _f(leaked["chars"].median()) if len(leaked) else None,
                 "chars_median_clean": _f(clean["chars"].median()) if len(clean) else None,
                 "dimensions_leaked": {k: _f(leaked[k].mean()) for k in SEVEN} if len(leaked) else None,
                 "dimensions_clean": {k: _f(clean[k].mean()) for k in SEVEN} if len(clean) else None}
            # Red flag by length quartile among non-leaked replies: if long outputs diluted
            # attention, red flag would fall with length here too.
            if len(clean) >= 4:
                q = np.quantile(clean["chars"].to_numpy(), [0.25, 0.5, 0.75])
                bins = np.digitize(clean["chars"].to_numpy(), q, right=False)
                rf = clean["red_flag"].to_numpy()
                ch = clean["chars"].to_numpy()
                r["clean_red_flag_by_length_quartile"] = [
                    {"n": int((bins == i).sum()), "red_flag": _f(rf[bins == i].mean()) if (bins == i).any() else None,
                     "chars_median": _f(np.median(ch[bins == i])) if (bins == i).any() else None} for i in range(4)]
            r["leaked_share_seed"] = _seed_summary(share[c].to_numpy(dtype=float), d.seeds)
            out["conditions"][c] = r
        for a, b in pairs:
            out["contrasts"][contrast_key(a, b)] = paired(share, a, b, d.seeds)
        res[col] = out
    return res


def reply_health_section(d: CellData, rh) -> dict:
    """Per condition: the replies that loop and the replies the token limit stopped.

    A loop is a reply over rh.loop_min_chars whose zlib ratio is under
    rh.loop_max_zlib_ratio (configs/report.yaml). The limit count reads finish_length,
    which only the package's eval stage stores, so older files count none. Descriptive and
    post hoc (2026-10-18): it reports the EOS fault of the training and decides no claim
    (docs/ANALYSIS_PLAN.md section 12)."""
    t = d.table
    if "zlib_ratio" not in t:
        return {"measured": False}
    res = {"measured": True, "loop_min_chars": rh.loop_min_chars,
           "loop_max_zlib_ratio": rh.loop_max_zlib_ratio, "conditions": {}}
    for c in d.conditions:
        sub = t[t["condition"] == c]
        # an empty reply has no ratio (NaN), and NaN < x is False: not a loop
        loop = (sub["chars"] > rh.loop_min_chars) & (sub["zlib_ratio"] < rh.loop_max_zlib_ratio)
        limit = sub["finish_length"].dropna()
        res["conditions"][c] = {
            "n": int(len(sub)), "n_loops": int(loop.sum()), "loop_share": _f(loop.mean()) if len(sub) else None,
            "per_seed_loops": {str(s): int(loop[sub["seed"] == s].sum()) for s in d.seeds},
            "n_limit_known": int(len(limit)), "n_at_limit": int((limit == 1).sum()),
            "limit_share": _f((limit == 1).mean()) if len(limit) else None}
        # Older files have no finish_length. A one-call condition (no protocol) that the cap
        # stopped has exactly eval.max_new_tokens engine tokens, which the extract stores;
        # on the R2 cell this count equals the finish_length count in every such condition
        # (ANALYSIS_PLAN.md section 12, 2026-10-23).
        engine = sub["tokens_engine"].dropna() if "tokens_engine" in sub else pd.Series(dtype=float)
        if not len(limit) and len(engine) and parse_condition(c)[1] is None:
            cap = d.cell.eval.max_new_tokens
            res["conditions"][c].update({"n_tokens_known": int(len(engine)),
                                         "n_at_cap_by_tokens": int((engine == cap).sum()),
                                         "cap_share_by_tokens": _f((engine == cap).mean())})
    return res


def paired_probe_section(d: CellData, reg: Registry, cfg: ReportConfig, metrics: list[Metric], pairs) -> dict:
    """The paired probe of docs/ANALYSIS_PLAN.md section 5.1 (H3, E14).

    For each discrimination label whose rule pairs rows (LabelRule.pair_field), and each
    condition c and seed s: Delta(c, s) is the mean over the seed's pairs of (response on
    the positive row minus response on the negative row). A pair leaves a condition and
    seed when either row has no value (an evaluator response counts scored replies only);
    the counts say how many. Each contrast a - b gets the difference in differences
    DiD(s) = Delta(a, s) - Delta(b, s), tested over seeds (section 2.1). Descriptive: per
    condition and seed, the 2x2 table of the two rows' 0/1 responses and its exact
    McNemar p.
    """
    bench = reg.benchmarks[d.cell.benchmark]
    by_key = {m.key: m for m in metrics}
    t = d.table
    res = {}
    for spec in cfg.discrimination:
        for lab in spec.labels:
            rule = bench.labels.get(lab)
            if rule is None or rule.pair_field is None or lab not in cfg.labels_for(d.cell):
                continue
            col = f"label_{lab}"
            if col not in d.prompts or t["pair_id"].isna().all():
                res.setdefault(spec.response, {})[lab] = {
                    "not_measured": "no pair ids in the response table, or no label column in prompts.csv"}
                continue
            group = dict(zip(d.prompts["prompt_id"], d.prompts[col]))
            pair_of = t.drop_duplicates("prompt_id").set_index("prompt_id")["pair_id"]
            frame = d.frame(by_key[spec.response])
            frame = frame.assign(g=frame["prompt_id"].map(group), pair=frame["prompt_id"].map(pair_of))
            frame = frame[frame["g"].notna() & frame["pair"].notna()]
            idx = ["seed", "condition", "pair"]
            both = pd.concat({"pos": frame[frame["g"] == 1].set_index(idx)["v"],
                              "neg": frame[frame["g"] == 0].set_index(idx)["v"]}, axis=1).dropna()
            complete = {k: w for k, w in both.groupby(level=["seed", "condition"])}
            drawn = t[t["pair_id"].notna()].groupby(["seed", "condition"])["pair_id"].nunique()
            out = {"timing": rule.timing, "source": rule.source, "exploratory": rule.timing == "post_hoc",
                   "conditions": {}, "contrasts": {}}
            delta = {}
            for c in d.conditions:
                per_seed, n_pairs, dropped, mcn = [], {}, {}, {}
                for seed in d.seeds:
                    w = complete.get((seed, c), both.iloc[0:0])
                    n_pairs[str(seed)] = int(len(w))
                    dropped[str(seed)] = int(drawn.get((seed, c), 0)) - int(len(w))
                    per_seed.append(float((w["pos"] - w["neg"]).mean()) if len(w) else float("nan"))
                    if len(w) and by_key[spec.response].kind == "rate":
                        pos, neg = w["pos"].to_numpy() == 1, w["neg"].to_numpy() == 1
                        table = [[int((pos & neg).sum()), int((pos & ~neg).sum())],
                                 [int((~pos & neg).sum()), int((~pos & ~neg).sum())]]
                        mcn[str(seed)] = {"table": table, "exact_p": _f(st.mcnemar_exact_p(table))}
                delta[c] = np.asarray(per_seed, dtype=float)
                out["conditions"][c] = {"delta": _seed_summary(delta[c], d.seeds), "n_pairs": n_pairs,
                                        "n_pairs_dropped": dropped, "mcnemar": mcn}
            for a, b in pairs:
                out["contrasts"][contrast_key(a, b)] = {"did": _seed_test(delta[a], delta[b], d.seeds)}
            res.setdefault(spec.response, {})[lab] = out
    return res


def subsets_section(d: CellData, reg: Registry, cfg: ReportConfig, metrics: list[Metric], pairs) -> dict:
    """Every contrast of the cell again on one group of a label's prompts only, with the
    same seed-level tests and Holm families as the whole cell (docs/ANALYSIS_PLAN.md
    section 9.3, H11: H1's rule on the clinician-user positive prompts, H2's contrast as a
    secondary test). The bootstrap resamples the subset's own prompts."""
    bench = reg.benchmarks[d.cell.benchmark]
    res = {}
    for spec in cfg.subsets:
        if spec.label not in cfg.labels_for(d.cell):
            continue
        key = f"{spec.label}={spec.group}"
        col = f"label_{spec.label}"
        if spec.label not in bench.labels:
            res[key] = {"not_measured": f"benchmark {bench.id} has no label {spec.label}"}
            continue
        if col not in d.prompts:
            res[key] = {"not_measured": f"prompts.csv has no column {col}; run analyze --extract "
                                        "where the cell's raw outputs are"}
            continue
        want = 1.0 if spec.group == "positive" else 0.0
        ids = set(d.prompts.loc[d.prompts[col] == want, "prompt_id"])
        table = d.table[d.table["prompt_id"].isin(ids)]
        if table.empty:
            res[key] = {"not_measured": f"no evaluated prompt is in the {spec.group} group"}
            continue
        sub = CellData(table, d.prompts[d.prompts["prompt_id"].isin(ids)], d.cell)
        # Keep the cell's seeds: a seed with no prompt of the group then shows in
        # seeds_missing, as section 2.3 asks, instead of leaving without a word.
        sub.seeds = d.seeds
        boot = st.ClusterBootstrap(table["prompt_id"].unique().tolist(), cfg.bootstrap.resamples, cfg.bootstrap.seed)
        rule = bench.labels[spec.label]
        res[key] = {"label": spec.label, "group": spec.group, "timing": rule.timing, "source": rule.source,
                    "n_prompts": int(table["prompt_id"].nunique()), "seeds": sub.seeds,
                    "contrasts": contrasts_section(sub, metrics, pairs, boot)}
    return res


def askers_gap_section(d: CellData, boot: st.ClusterBootstrap) -> dict:
    """Rubric score of replies that ask minus replies that do not, within each condition.

    Pooled over seeds over scored replies. The two groups select themselves, so the gap
    is descriptive (withdrawal W7 tested whether the rubric penalizes asking with it).
    """
    s = d.scored
    res = {}
    for c in d.conditions:
        sub = s[s["condition"] == c]
        ask = sub[sub["active_inquiry"] == 1.0][["prompt_id", "rubric_score"]].rename(columns={"rubric_score": "v"})
        rest = sub[sub["active_inquiry"] == 0.0][["prompt_id", "rubric_score"]].rename(columns={"rubric_score": "v"})
        if not len(ask) or not len(rest):
            res[c] = {"gap": None, "n_ask": int(len(ask)), "n_no_ask": int(len(rest)),
                      "not_measured": "one of the two groups is empty"}
            continue
        res[c] = {"gap": _f(ask["v"].mean() - rest["v"].mean()), "n_ask": int(len(ask)), "n_no_ask": int(len(rest)),
                  "gap_ci95": st.percentile_ci(boot.means(ask) - boot.means(rest))}
    return res


def _read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.is_file() else None


def filter_section(cell: CellConfig, paths: CellPaths, seeds: list[int]) -> tuple[dict, Optional[set]]:
    """Tau, trace counts and the score distribution of the filter.

    Counts come from the cell's shared filter manifest (recorded values carry their
    source). The score distribution comes from a seed's filter scores, when one was kept.
    """
    res = {"tau_rule": cell.filter.tau.model_dump() if cell.filter else None}
    shared = _read_json(paths.shared("filter") / "manifest.json")
    rec = (shared or {}).get("recorded") or {}
    res["recorded"] = rec or None
    res["recorded_source"] = (shared or {}).get("source")
    if rec.get("kept") and (rec.get("graded") or rec.get("raw")):
        res["kept_share_of_graded"] = _f(rec["kept"] / (rec.get("graded") or rec["raw"]))
        # The percent from the counts themselves: 100 times the rounded share can land on
        # the other side of a half (814 of 4,000 is 20.35, not 20.349999...).
        res["kept_pct_of_graded"] = _f(100 * rec["kept"] / (rec.get("graded") or rec["raw"]))
    if not rec:
        res["not_measured_counts"] = "the cell's filter counts were not recorded"
    graded = _read_json(paths.shared("filter") / "scores.json")
    if graded and graded.get("rows"):
        # Every graded trace's score (the filter stage writes it since 2026-10-10): the
        # share at or above each tau is exact, not an estimate from a train split.
        s = np.asarray([x for _, x in graded["rows"] if x is not None], dtype=float)
        res["graded_scores"] = {"n": int(len(s)), "min": _f(s.min()), "median": _f(np.median(s)),
                                "tau_grid": [{"tau": tau, "share_of_graded": float((s >= tau - 1e-12).mean())}
                                             for tau in TAU_GRID]}
    train_ids = None
    for seed in seeds + [s for s in cell.seeds if s not in seeds]:
        f = paths.stage(seed, "filter") / "filter_scores.json"
        if not f.is_file():
            continue
        rows = json.loads(f.read_text())["rows"]
        scores = np.asarray([x for _, x in rows], dtype=float)
        train_ids = {p for p, _ in rows if p is not None}
        grid = []
        for tau in TAU_GRID:
            share = float((scores >= tau - 1e-12).mean())
            row = {"tau": tau, "share_of_train_split": share}
            if res.get("kept_share_of_graded") is not None:
                # The train split is a subset of the kept traces; its score shares stand for
                # the kept traces, scaled by the share the filter kept.
                row["share_of_pool_estimate"] = share * res["kept_share_of_graded"]
            grid.append(row)
        res["scores"] = {"seed": seed, "n_train_split": int(len(scores)), "min": _f(scores.min()), "tau_grid": grid}
        break
    return res, train_ids


def recipe_section(cell: CellConfig, paths: CellPaths, filt: dict) -> dict:
    """The adapter recipe each seed's files record, checked across seeds.

    The steps per epoch, the per-device batch and the train count must admit one
    gradient accumulation (HF Trainer: steps per epoch = ceil(ceil(train / batch) /
    accumulation)).
    """
    per_seed = {}
    for seed in cell.seeds:
        a = _read_json(paths.adapter_dir(seed) / "adapter_config.json")
        t = _read_json(paths.adapter_dir(seed) / "trainer_state.json")
        if a is None or t is None:
            continue
        per_seed[str(seed)] = {"r": a["r"], "lora_alpha": a["lora_alpha"], "lora_dropout": a["lora_dropout"],
                               "target_modules": sorted(a["target_modules"]), "max_steps": t["max_steps"],
                               "num_train_epochs": t["num_train_epochs"], "train_batch_size": t["train_batch_size"],
                               "best_global_step": t.get("best_global_step")}
    if not per_seed:
        return {"not_measured": "no adapter files in the cell's train folders", "seeds_trained": []}
    shared = ("r", "lora_alpha", "lora_dropout", "target_modules", "max_steps", "num_train_epochs", "train_batch_size")
    res = {"seeds_trained": sorted(int(s) for s in per_seed)}
    for k in shared:
        vals = {json.dumps(r[k]) for r in per_seed.values()}
        res[k] = json.loads(vals.pop()) if len(vals) == 1 else None
        if res[k] is None:
            res.setdefault("varies", []).append(k)
    if res["max_steps"] and res["num_train_epochs"]:
        spe = res["max_steps"] / res["num_train_epochs"]
        res["steps_per_epoch"] = int(spe) if spe == int(spe) else spe
        res["best_epoch"] = {s: (r["best_global_step"] / spe if r["best_global_step"] else None)
                             for s, r in per_seed.items()}
        best = [e for e in res["best_epoch"].values() if e is not None]
        res["best_epoch_min"] = min(best) if best else None
        res["best_epoch_max"] = max(best) if best else None
    train = ((filt.get("recorded") or {}).get("train"))
    if train and res.get("train_batch_size") and isinstance(res.get("steps_per_epoch"), int):
        batches = math.ceil(train / res["train_batch_size"])
        acc = [a for a in range(1, 65) if math.ceil(batches / a) == res["steps_per_epoch"]]
        res["grad_accumulation_consistent"] = acc
        res["effective_batch_consistent"] = [a * res["train_batch_size"] for a in acc]
    return res


def probe_section(paths: CellPaths) -> dict:
    """The contamination probe of the cell's base model (E11), as the probe stage wrote it
    in shared/probe/probe.json: the exact-completion rate on the benchmark and on the
    control text, and the stage's two-sided Fisher test. Copied here so that a correction
    family across cells (F7, configs/report.yaml `families`) can read it."""
    p = _read_json(paths.shared("probe") / "probe.json")
    if p is None:
        return {"not_measured": "no probe.json in the cell's shared/probe folder"}
    keep = ("n", "n_rows_in_set", "n_exact_match", "exact_match_rate", "mean_prefix_overlap_tokens")
    b, c = p["benchmark"], p["control"]
    res = {"model": p.get("model"), "model_revision": p.get("model_revision"), "method": p.get("method"),
           "benchmark_id": p.get("benchmark_id"), "control_id": p.get("control_id"),
           "benchmark": {k: b.get(k) for k in keep}, "control": {k: c.get(k) for k in keep},
           "fisher_p": _f(p.get("fisher_p_value")), "test": p.get("test_used")}
    if b.get("exact_match_rate") is not None and c.get("exact_match_rate") is not None:
        res["rate_diff"] = _f(b["exact_match_rate"] - c["exact_match_rate"])
    return res


def bench_section(d: CellData, paths: CellPaths, pairs) -> dict:
    """Measured inference cost (E20, C25): the bench stage's service time per reply, one
    reply at a time on its own engine, per condition and seed (bench/bench.json), and for
    each contrast the ratio and difference of the median times. Only the bench times an
    idle GPU; the stage times below include whatever else ran on the node."""
    runs = {s: b for s in d.cell.seeds if (b := _read_json(paths.stage(s, "bench") / "bench.json")) is not None}
    if not runs:
        return {"not_measured": "no bench.json in the cell's seed folders"}
    res = {"seeds": sorted(runs), "n_prompts": {str(s): b["n_prompts"] for s, b in runs.items()},
           "conditions": {}, "contrasts": {}}
    for c in d.conditions:
        res["conditions"][c] = {
            str(s): {"median_s": _f(r.get("median_s")), "mean_s": _f(r.get("mean_s")), "p90_s": _f(r.get("p90_s")),
                     "mean_calls": _f(np.mean(r["calls"])) if r.get("calls") else None,
                     "mean_completion_tokens": _f(r.get("mean_completion_tokens")),
                     "mean_prompt_tokens": _f(r.get("mean_prompt_tokens")),
                     "throughput_tok_per_s": _f(r.get("throughput_tok_per_s")),
                     "engine_start_s": _f(r.get("engine_start_s"))}
            for s, b in runs.items() if (r := b["conditions"].get(c)) is not None}
    for a, b in pairs:
        out = {}
        for s, ra in res["conditions"].get(a, {}).items():
            rb = res["conditions"].get(b, {}).get(s)
            if rb and ra["median_s"] is not None and rb["median_s"]:
                out[s] = {"median_s_ratio": _f(ra["median_s"] / rb["median_s"]),
                          "median_s_diff": _f(ra["median_s"] - rb["median_s"])}
        res["contrasts"][contrast_key(a, b)] = out
    return res


def stage_times_section(cell: CellConfig, paths: CellPaths) -> dict:
    """Wall-clock seconds and the GPUs of each stage, from the stage manifests (E08, C25):
    the shared stages once per cell, the others per seed (a folder with several manifests,
    such as eval with one per condition, sums them). An eval through one shared server
    (pipeline/eval.py, run_shared) serves several seeds at once, so its time is in one
    manifest in shared/eval and counts there, once; its condition manifests hold none. The
    manifests of the recorded cells were written after their runs and hold no times, so
    for them this says so."""
    def read(folder: Path) -> Optional[dict]:
        ms = [m for f in sorted(folder.glob("*manifest.json")) if "time" in (m := json.loads(f.read_text()))]
        if not ms:
            return None
        gpus = sorted({g.get("name") for m in ms for g in (m.get("hardware") or {}).get("gpus", []) if g.get("name")})
        return {"seconds": _f(sum(m["time"]["seconds"] for m in ms)), "n_manifests": len(ms), "gpus": gpus}

    shared = {st_: r for st_ in STAGES if (r := read(paths.shared(st_))) is not None}
    per_seed = {}
    for st_ in STAGES:
        rows = {str(s): r for s in cell.seeds if (r := read(paths.stage(s, st_))) is not None}
        if rows:
            per_seed[st_] = rows
    if not shared and not per_seed:
        return {"not_measured": "no stage manifest of this cell records a time"}
    return {"shared": shared, "per_seed": per_seed}


def _judge_decision(evaluator: dict, judge: dict, alpha: float) -> str:
    """One contrast under a judge against the same contrast under the evaluator
    (docs/ANALYSIS_PLAN.md section 8): confirmed when the judge's Holm-adjusted p is below
    alpha with the evaluator's sign, reversed when it is below alpha with the other sign,
    and not confirmed otherwise."""
    p, dj, de = judge.get("p_holm_within_contrast"), judge.get("diff"), evaluator.get("diff")
    if p is None or dj is None or de is None or not (np.isfinite(p) and np.isfinite(dj) and np.isfinite(de)):
        return "not confirmed"
    if p >= alpha or dj == 0 or de == 0:
        return "not confirmed"
    return "confirmed" if np.sign(dj) == np.sign(de) else "reversed"


def _agreement(boot: st.ClusterBootstrap, prompt_ids, a: np.ndarray, b: np.ndarray, k: int,
               quadratic: bool) -> dict:
    """Cohen's kappa of two raters on categories 0..k-1, with a prompt-clustered bootstrap
    interval. A value off the scale leaves its row out, and the output counts it."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    on = np.isin(a, np.arange(k)) & np.isin(b, np.arange(k))
    cell_of = (a[on] * k + b[on]).astype(int)
    onehot = np.zeros((len(cell_of), k * k))
    onehot[np.arange(len(cell_of)), cell_of] = 1.0
    point = st.kappa(onehot.sum(axis=0).reshape(k, k), quadratic)
    boots = st.kappa(boot.totals(np.asarray(prompt_ids)[on], onehot).reshape(-1, k, k), quadratic)
    return {"kappa": _f(point), "kappa_ci95": [_f(x) for x in st.percentile_ci(boots)],
            "weights": "quadratic" if quadratic else "none", "n": int(on.sum()), "n_off_scale": int((~on).sum())}


def judges_section(d: CellData, pairs, boot: st.ClusterBootstrap, evaluator: dict, alpha: float) -> dict:
    """Each judge against the evaluator, and the contrasts recomputed with it (H8,
    docs/ANALYSIS_PLAN.md section 8).

    Agreement (descriptive): Cohen's kappa per rubric criterion, kappa with quadratic
    weights on each 0 to 2 dimension, Cohen's kappa on active inquiry, Pearson r of the
    rubric score, each with a prompt-clustered bootstrap interval. Contrasts: every
    contrast of the cell on the rubric score and the seven dimensions, from the judge's
    grades alone, with Holm over the seven within each contrast (F1). Each dimension gets
    the plan's decision against the evaluator's result in `evaluator` (the contrasts
    section). The rubric score's correction runs across cells (F2), so it gets no decision
    here. No correction runs across judges.
    """
    t = d.table
    res = {}
    for jid in d.cell.grading.judges:
        j = f"j_{jid}__"
        if f"{j}rubric_score" not in t:
            res[jid] = {"not_measured": "no judge file in the cell's grade folders"}
            continue
        both = t[t[f"{j}rubric_score"].notna()]
        out = {"n_regraded": int(len(both)),
               "rubric_score_pearson_r": _f(st.pearson_r(both["rubric_score"], both[f"{j}rubric_score"]))}
        if len(both) > 1:
            x, y = both["rubric_score"].to_numpy(dtype=float), both[f"{j}rubric_score"].to_numpy(dtype=float)
            n, sx, sy, sxx, syy, sxy = boot.totals(both["prompt_id"], np.stack(
                [np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)).T
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
            out["rubric_score_pearson_r_ci95"] = [_f(v) for v in st.percentile_ci(r)]
        crit = [f"{j}crit_{k}" for k in ("11", "10", "01", "00")]
        counts = both[crit].fillna(0.0).to_numpy(dtype=float)
        if counts.sum():
            # Rows: the evaluator's verdict (met, not met); columns: the judge's.
            out["criterion_kappa"] = _f(st.kappa(counts.sum(axis=0).reshape(2, 2)))
            out["criterion_kappa_ci95"] = [_f(v) for v in st.percentile_ci(
                st.kappa(boot.totals(both["prompt_id"], counts).reshape(-1, 2, 2)))]
            out["criterion_n"] = int(counts.sum())
        sc = t[(t["epi_status"] == "scored") & t[f"{j}active_inquiry"].notna()]
        out["dimension_agreement"] = {k: _f(np.mean(sc[k] == sc[f"{j}{k}"])) if len(sc) else None for k in SEVEN}
        out["dimension_kappa"] = {}
        for k in SEVEN:
            graded = sc[sc[k].notna() & sc[f"{j}{k}"].notna()]
            # Active inquiry is yes or no; the other six are 0 to 2 grades.
            binary = EPISTEMIC_KIND.get(k) == "rate"
            out["dimension_kappa"][k] = (_agreement(boot, graded["prompt_id"], graded[k], graded[f"{j}{k}"],
                                                    2 if binary else 3, quadratic=not binary)
                                         if len(graded) else {"not_measured": "no reply graded by both"})
        contrasts = {}
        rubric = d.seed_means(both[["seed", "condition", "prompt_id", f"{j}rubric_score"]]
                              .rename(columns={f"{j}rubric_score": "v"}))
        for a, b in pairs:
            key = contrast_key(a, b)
            ms = {}
            for k in SEVEN:
                rows = t[t[f"{j}{k}"].notna()]
                ms[k] = paired(d.seed_means(rows[["seed", "condition", "prompt_id", f"{j}{k}"]]
                                            .rename(columns={f"{j}{k}": "v"})), a, b, d.seeds)
            for k, ph in zip(SEVEN, st.holm([ms[k].get("p", float("nan")) for k in SEVEN])):
                ms[k]["p_holm_within_contrast"] = _f(ph)
                if "not_measured" not in ms[k]:
                    ms[k]["decision"] = _judge_decision(evaluator[key]["metrics"][k], ms[k], alpha)
            ms["rubric_score"] = paired(rubric, a, b, d.seeds)
            contrasts[key] = {"a": a, "b": b, "metrics": ms}
        out["contrasts"] = contrasts
        res[jid] = out
    return res


def analyze_tables(reg: Registry, cell: CellConfig, table: pd.DataFrame, prompts: pd.DataFrame,
                   meta: list[dict], paths: CellPaths, cfg: ReportConfig) -> dict:
    """Every statistic of one cell, from its tables. Pure: the caller reads and writes files."""
    d = CellData(table, prompts, cell)
    metrics = metrics_of(table, cfg)
    pairs = contrasts_of(cell, reg)
    boot = st.ClusterBootstrap(table["prompt_id"].unique().tolist(), cfg.bootstrap.resamples, cfg.bootstrap.seed)
    # An eval-only cell (adapters_from) trained nothing itself: its adapters, and the
    # filter record they were trained on, are in the source cell's folders.
    trained = reg.cells[cell.adapters_from] if cell.adapters_from else cell
    tpaths = CellPaths(paths.root.parent / trained.id) if cell.adapters_from else paths
    filt, train_ids = filter_section(trained, tpaths, d.seeds)
    recipe = recipe_section(trained, tpaths, filt)
    if cell.adapters_from:
        filt["from_cell"] = recipe["from_cell"] = trained.id
    contrasts = contrasts_section(d, metrics, pairs, boot)
    out = {
        "cell": cell.id, "status": cell.report.status, "seeds": d.seeds, "conditions_order": d.conditions,
        "evaluator": cell.grading.evaluator,
        "marks": {"family_waiver": cell.grading.same_family_waiver, "set_aside": cell.report.set_aside},
        "contrasts_order": [contrast_key(a, b) for a, b in pairs],
        "bootstrap": {"resamples": cfg.bootstrap.resamples, "seed": cfg.bootstrap.seed,
                      "cluster": "prompt_id", "interval": "percentile"},
        "metrics": {m.key: {"kind": m.kind, "source": m.source, "secondary": m.secondary} for m in metrics},
        "conditions": conditions_section(d, metrics),
        "contrasts": contrasts,
        "counts": counts_section(d, meta),
        "worst_case": worst_case_section(d, metrics, pairs),
        "below_cap": below_cap_section(d, metrics, pairs),
        "blanket_disclaimers": disclaimers_section(d),
        "length_quartiles": length_quartiles_section(d, cfg.length_reference),
        "cost": cost_section(d, reg, pairs, meta),
        "shift_correlations": shift_correlations_section(d, pairs),
        "context_seeking_anchor": context_seeking_anchor_section(d),
        "axis_scores": axis_section(d),
        "prompts": prompts_section(d, train_ids, _read_json(paths.analysis() / "extract.json")),
        "themes": themes_section(d),
        "leakage": leakage(d, reg, pairs),
        "reply_health": reply_health_section(d, cfg.reply_health),
        "discrimination": discrimination_section(d, reg, cfg, metrics, pairs, boot),
        "paired_probe": paired_probe_section(d, reg, cfg, metrics, pairs),
        "subsets": subsets_section(d, reg, cfg, metrics, pairs),
        "askers_gap": askers_gap_section(d, boot),
        "filter": filt,
        "recipe": recipe,
        "judges": judges_section(d, pairs, boot, contrasts, cfg.alpha),
        "probe": probe_section(paths),
        "bench": bench_section(d, paths, pairs),
        "stage_times": stage_times_section(cell, paths),
    }
    kappa = _read_json(paths.validation() / "kappa_results.json")
    out["validation"] = kappa if kappa is not None else {"not_measured": "no physician agreement file"}
    return out


def load_tables(cell: CellConfig, paths: CellPaths) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    tables, meta = [], []
    for seed in cell.seeds:
        p = paths.table(seed)
        if p.is_file():
            tables.append(read_table(p))
            meta.append(json.loads(p.with_suffix(".meta.json").read_text()))
    if not tables:
        raise FileNotFoundError("no seed has a table.csv; run analyze --extract where the raw outputs are")
    return pd.concat(tables, ignore_index=True), read_table(paths.analysis() / "prompts.csv"), meta


def round6(obj):
    """Floats to 6 significant digits, so stats.json is stable across platforms.
    Significant digits, not decimal places: p-values go down to 1e-8."""
    if isinstance(obj, float):
        return float(f"{obj:.6g}") if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: round6(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round6(v) for v in obj]
    if isinstance(obj, np.floating):
        return round6(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def analyze(reg: Registry, cell_id: str, project: Project, cfg: ReportConfig) -> dict:
    """Read a cell's tables, write its analysis/stats.json, and return the statistics."""
    cell = reg.cells[cell_id]
    paths = project.cell(cell_id)
    table, prompts, meta = load_tables(cell, paths)
    out = round6(analyze_tables(reg, cell, table, prompts, meta, paths, cfg))
    paths.analysis().mkdir(parents=True, exist_ok=True)
    (paths.analysis() / "stats.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out
