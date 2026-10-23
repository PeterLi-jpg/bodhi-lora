"""The decision rules of docs/ANALYSIS_PLAN.md sections 3 to 7 and 9.3, applied to the
cells that ran each question (configs/report.yaml, `hypotheses`).

    hypotheses_summary(all_stats, cfg, reg) -> {question id: {cells: {cell: result}, ...}}

Each rule reads only a cell's analysis/stats.json and returns, per cell, the plan's
decision and the outcomes that decide it: the difference, its 95% t interval, the p-value
and, where the plan names a correction family across cells (section 2.7), the Holm-adjusted
p over the question's cells. Differences keep the scale of stats.json (a rate is a share,
not percentage points); the report scales them. A cell or an outcome that is missing is
listed, not decided.

The rules quote the plan. Where the plan's wording leaves a case open, the docstring of
the rule says how it is read.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from bodhi_lora.analysis import stats as st
from bodhi_lora.analysis.cell import _f
from bodhi_lora.registry import parse_condition

PRIMARY_ASKING = ("active_inquiry", "context_seeking")


def _contrast(stats: dict, contrast: str, subset: Optional[str] = None) -> dict:
    src = stats["subsets"].get(subset, {}) if subset else stats
    return src.get("contrasts", {}).get(contrast, {}).get("metrics", {})


def _outcome(r: Optional[dict], p_key: Optional[str] = None) -> Optional[dict]:
    """The fields of one seed-level test that the report prints; None when it did not run.
    p_adj is the p of the correction the plan names for this outcome: within the contrast
    (p_key) for the seven dimensions, Holm across the question's cells for a family (set
    by hypotheses_summary), none for a secondary outcome."""
    if not r or r.get("diff") is None:
        return None
    return {"diff": _f(r["diff"]), "ci95": [_f(v) for v in r["ci95"]] if r.get("ci95") else None,
            "p": _f(r.get("p")), "p_adj": _f(r.get(p_key)) if p_key else None, "n_seeds": r.get("n_seeds")}


def _up(o: Optional[dict], alpha: float) -> bool:
    return bool(o) and o["p_adj"] is not None and o["p_adj"] < alpha and o["diff"] > 0


def _down(o: Optional[dict], alpha: float) -> bool:
    return bool(o) and o["p_adj"] is not None and o["p_adj"] < alpha and o["diff"] < 0


def _ci_above_zero(o: Optional[dict]) -> bool:
    return bool(o) and o["ci95"] is not None and o["ci95"][0] > 0


def asking(stats: dict, alpha: float, contrast: str = "lora - base", subset: Optional[str] = None) -> dict:
    """H1 (section 3), and H5 and H11 by H1's rule: `lora` minus `base` on active inquiry and
    context-seeking, each Holm-adjusted within the contrast (F1).

    Supported: both rise with adjusted p below alpha. Not supported: neither rises, or
    either falls with adjusted p below alpha. Partial: one rises; the plan writes "the
    other's interval includes 0", and a rise whose interval clears 0 only before Holm is
    read as partial too, since it is not a second supported rise."""
    m = _contrast(stats, contrast, subset)
    out = {k: _outcome(m.get(k), "p_holm_within_contrast") for k in PRIMARY_ASKING}
    if any(o is None for o in out.values()):
        return {"decision": None, "outcomes": out, "not_measured": "a primary outcome is missing"}
    ups = [_up(o, alpha) for o in out.values()]
    downs = [_down(o, alpha) for o in out.values()]
    if all(ups):
        decision = "supported"
    elif not any(ups) or any(downs):
        decision = "not supported"
    else:
        decision = "partial"
    secondary = {"rubric_score": _outcome(m.get("rubric_score"))}
    return {"decision": decision, "outcomes": out, "secondary": secondary}


def cost(stats: dict) -> dict:
    """H2 (section 4): `lora` minus `base` on the rubric score. Its family F2 runs across
    the question's cells, so the decision comes after the Holm step (decide_cost)."""
    return {"outcomes": {"rubric_score": _outcome(_contrast(stats, "lora - base").get("rubric_score"))}}


def decide_cost(res: dict, alpha: float) -> None:
    o = res["outcomes"]["rubric_score"]
    if o is None:
        res["decision"] = None
    elif _down(o, alpha):
        res["decision"] = "lowers the score"
    elif _up(o, alpha):
        res["decision"] = "raises the score"
    else:
        res["decision"] = "no change detected"


def paired_probe(stats: dict) -> dict:
    """H3 (section 5.1) on a MediQ paired cell: DiD = Delta(lora) - Delta(base), with Delta
    the ask rate on the incomplete prompt minus the complete one; F3 across the cells, so
    the decision comes after the Holm step (decide_paired_probe). The mimicry test reads
    `lora` minus `base` on the ask rate of the complete prompts, the negative group of the
    `incomplete` label."""
    pp = stats.get("paired_probe", {}).get("active_inquiry", {}).get("incomplete")
    disc = stats.get("discrimination", {}).get("active_inquiry", {}).get("incomplete", {})
    if not pp:
        return {"outcomes": {}, "not_measured": "no paired probe in this cell"}
    lora = pp["conditions"].get("lora", {}).get("delta", {})
    base = pp["conditions"].get("base", {}).get("delta", {})
    complete = disc.get("contrasts", {}).get("lora - base", {}).get("negative_rate", {}).get("seed_test")
    return {"outcomes": {"did": _outcome(pp["contrasts"].get("lora - base", {}).get("did")),
                         "complete_rate": _outcome(complete)},
            "delta": {"lora": {"seed_mean": _f(lora.get("seed_mean")), "ci95": lora.get("seed_ci95_t")},
                      "base": {"seed_mean": _f(base.get("seed_mean")), "ci95": base.get("seed_ci95_t")}}}


def decide_paired_probe(res: dict, alpha: float) -> None:
    """Calibration supported: DiD above 0 with adjusted p below alpha, and the interval of
    Delta(lora) above 0. Mimicry indicated: the complete-prompt rise has an interval above
    0, and the interval of DiD includes 0 or lies below it. Not resolved: neither."""
    did, comp = res["outcomes"].get("did"), res["outcomes"].get("complete_rate")
    if did is None or comp is None:
        res["decision"] = None
        return
    lora_ci = (res.get("delta", {}).get("lora") or {}).get("ci95")
    if _up(did, alpha) and lora_ci and lora_ci[0] > 0:
        res["decision"] = "calibration supported"
    elif _ci_above_zero(comp) and not _ci_above_zero(did):
        res["decision"] = "mimicry indicated"
    else:
        res["decision"] = "not resolved"


def baselines(stats: dict, alpha: float, protocols: dict, names: list[str]) -> dict:
    """H4 (section 6.1): `lora` minus each prompt baseline in `names`, on active inquiry
    and context-seeking (F1 within each contrast). Adapter ahead: both higher for `lora`
    with adjusted p below alpha. Baseline ahead: either higher for the baseline. Not
    separated: otherwise. `narrows` is the plan's clause for the paper: a one-call baseline
    that is ahead or not separated while `lora` minus it on the rubric score is below 0
    with p below alpha. A baseline the cell did not run is listed with no decision."""
    out = {}
    for b in names:
        _, proto = parse_condition(b)
        m = _contrast(stats, f"lora - {b}")
        o = {k: _outcome(m.get(k), "p_holm_within_contrast") for k in PRIMARY_ASKING}
        rubric = _outcome(m.get("rubric_score"))
        if any(v is None for v in o.values()):
            out[b] = {"decision": None, "outcomes": o}
            continue
        if all(_up(v, alpha) for v in o.values()):
            decision = "adapter ahead"
        elif any(_down(v, alpha) for v in o.values()):
            decision = "baseline ahead"
        else:
            decision = "not separated"
        one_call = proto in protocols and protocols[proto].passes == 1
        narrows = (one_call and decision != "adapter ahead" and rubric is not None
                   and rubric["p"] is not None and rubric["p"] < alpha and rubric["diff"] < 0)
        out[b] = {"decision": decision, "outcomes": o, "secondary": {"rubric_score": rubric},
                  "one_call": one_call, "narrows": narrows}
    return {"baselines": out, "outcomes": {}}


def red_flags_first(stats: dict, protocol: str) -> dict:
    """H6 (section 7.1). Precondition: `lora+<protocol>` minus `lora` on red-flag
    identification; an interval that includes 0 or lies above it is "no drop to
    recover". Primary: `lora+red-flags-first` minus `lora+<protocol>`, F4 across the
    cells, so the decision comes after the Holm step (decide_red_flags_first)."""
    drop = _outcome(_contrast(stats, f"lora+{protocol} - lora").get("red_flag"))
    primary = _outcome(_contrast(stats, f"lora+red-flags-first - lora+{protocol}").get("red_flag"))
    return {"outcomes": {"red_flag": primary, "precondition": drop}}


def decide_red_flags_first(res: dict, alpha: float) -> None:
    drop = res["outcomes"].get("precondition")
    if drop is None:
        res["decision"] = None
    elif drop["ci95"] is None or drop["ci95"][1] >= 0:
        res["decision"] = "no drop to recover"
    elif res["outcomes"].get("red_flag") is None:
        res["decision"] = None
    else:
        res["decision"] = "recovered" if _up(res["outcomes"]["red_flag"], alpha) else "not recovered"


def over_flagging(stats: dict) -> dict:
    """H7 (section 7.2): `lora` minus `base` on the red-flag rate among the prompts that
    physicians labelled non-emergent (the negative group of `emergent-physician`), F5
    across the cells; the decision comes after the Holm step. The secondary outcomes:
    the same on the emergent prompts, and the rubric score and the harm proxy on the
    non-emergent prompts."""
    ctr = (stats.get("discrimination", {}).get("red_flag_rate", {}).get("emergent-physician", {})
           .get("contrasts", {}).get("lora - base", {}))
    sub = _contrast(stats, "lora - base", "emergent-physician=negative")
    return {"outcomes": {"non_emergent_rate": _outcome((ctr.get("negative_rate") or {}).get("seed_test"))},
            "secondary": {"emergent_rate": _outcome((ctr.get("positive_rate") or {}).get("seed_test")),
                          "non_emergent_rubric_score": _outcome(sub.get("rubric_score")),
                          "non_emergent_neg_any": _outcome(sub.get("neg_any"))}}


def physician_index(stats: dict) -> dict:
    """E02 (section 5.2), the secondary test of H3: `lora` minus `base` on the
    physician-label discrimination index, F6 across the question's cells."""
    ctr = (stats.get("discrimination", {}).get("active_inquiry", {}).get("needs-info-physician", {})
           .get("contrasts", {}).get("lora - base", {}))
    return {"outcomes": {"index": _outcome(ctr.get("seed_test"))}}


def context_pair(stats: dict, reference: Optional[dict], protocol: str) -> dict:
    """The 4,096-token pair (ANALYSIS_PLAN.md section 12, the 2026-10-19 entry on R7). In the
    short-context cell, section 7.1's precondition (`lora+<protocol>` minus `lora` on
    red-flag identification) and primary test (`lora+red-flags-first` minus
    `lora+<protocol>`) run with their own p; the cell joins no family. Context implicated:
    the precondition's interval lies below 0 here and not in the reference cell at the full
    context. Not implicated: its interval here includes 0 or lies above it. Not resolved: a
    drop at both. With no reference stats there is no decision."""
    short = _outcome(_contrast(stats, f"lora+{protocol} - lora").get("red_flag"))
    primary = _outcome(_contrast(stats, f"lora+red-flags-first - lora+{protocol}").get("red_flag"))
    full = _outcome(_contrast(reference, f"lora+{protocol} - lora").get("red_flag")) if reference else None
    out = {"outcomes": {"precondition": short, "red_flag": primary, "precondition_full": full}}

    def drop(o):
        return o is not None and o["ci95"] is not None and o["ci95"][1] < 0

    if short is None or short["ci95"] is None:
        out["decision"] = None
    elif not drop(short):
        out["decision"] = "context not implicated"
    elif full is None:
        out["decision"] = None
    else:
        out["decision"] = "not resolved" if drop(full) else "context implicated"
    return out


def decide_up(res: dict, alpha: float, key: str, yes: str, no: str) -> None:
    o = res["outcomes"].get(key)
    res["decision"] = None if o is None else (yes if _up(o, alpha) else no)


# Questions whose family runs across their cells: the outcome Holm reads, and the step
# that decides each cell once the adjusted p is in.
FAMILY = {
    "cost": ("rubric_score", decide_cost),
    "paired_probe": ("did", decide_paired_probe),
    "red_flags_first": ("red_flag", decide_red_flags_first),
    "over_flagging": ("non_emergent_rate",
                      lambda r, a: decide_up(r, a, "non_emergent_rate", "over-flagging found", "not detected")),
    "physician_index": ("index",
                        lambda r, a: decide_up(r, a, "index", "consistent with calibration", "not detected")),
}


# The questions whose decision the report repeats on the replies below the token cap
# (docs/ANALYSIS_PLAN.md section 12, 2026-10-23): H1, H2, H4 and H6. A question that reads
# a subset of prompts keeps its own rule only.
BELOW_CAP_RULES = ("asking", "cost", "baselines", "red_flags_first")


def _decide(h, cid: str, stats: dict, all_stats: dict, cfg, reg) -> dict:
    if h.rule == "asking":
        return asking(stats, cfg.alpha, subset=h.subset)
    if h.rule == "cost":
        return cost(stats)
    if h.rule == "paired_probe":
        return paired_probe(stats)
    if h.rule == "baselines":
        return baselines(stats, cfg.alpha, reg.protocols, h.baselines)
    if h.rule == "red_flags_first":
        return red_flags_first(stats, reg.cells[cid].protocol)
    if h.rule == "over_flagging":
        return over_flagging(stats)
    if h.rule == "physician_index":
        return physician_index(stats)
    if h.rule == "context_pair":
        return context_pair(stats, all_stats.get(h.reference), reg.cells[cid].protocol)
    raise ValueError(f"no rule {h.rule}")


def _decide_family(h, cells: dict, cfg) -> None:
    """Holm across the question's cells where the plan names a family, then each cell's
    decision on the adjusted p."""
    if h.rule not in FAMILY:
        return
    key, decide = FAMILY[h.rule]
    tested = [(cid, r["outcomes"][key]) for cid, r in cells.items()
              if r.get("outcomes", {}).get(key) and r["outcomes"][key]["p"] is not None]
    holm = st.holm([o["p"] for _, o in tested])
    for (cid, o), h_p in zip(tested, holm):
        o["p_adj"] = _f(h_p) if np.isfinite(h_p) else None
    for r in cells.values():
        if "outcomes" in r:
            decide(r, cfg.alpha)


def _below_cap(stats: Optional[dict]) -> Optional[dict]:
    """The cell's stats with the below-cap contrasts in place of the contrasts; None when
    the cell has none."""
    contrasts = ((stats or {}).get("below_cap") or {}).get("contrasts")
    return {**stats, "contrasts": contrasts, "subsets": {}} if contrasts else None


def hypotheses_summary(all_stats: dict, cfg, reg) -> dict:
    out = {}
    for h in cfg.hypotheses:
        cells = {}
        for cid in h.cells:
            stats = all_stats.get(cid)
            if stats is None:
                cells[cid] = {"decision": None, "not_measured": "no stats.json for this cell"}
                continue
            cells[cid] = _decide(h, cid, stats, all_stats, cfg, reg)
        _decide_family(h, cells, cfg)
        if h.rule in BELOW_CAP_RULES and not h.subset:
            below = {cid: _decide(h, cid, b, all_stats, cfg, reg) for cid in h.cells
                     if (b := _below_cap(all_stats.get(cid))) is not None}
            _decide_family(h, below, cfg)
            for cid, r in below.items():
                cells[cid]["below_cap"] = r
        out[h.id] = {"token": h.token, "rule": h.rule, "subset": h.subset, "family": h.rule in FAMILY,
                     "reference": h.reference, "cells": cells}
    return out
