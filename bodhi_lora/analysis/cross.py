"""Every cell's stats.json -> the cross-cell statistics (docs/ARCHITECTURE.md section 7).

    <results root>/cross_cell/stats.json

One row per cell of report.cross_cells with its marks; the groups of configs/report.yaml
with, per contrast and metric, how many cells move each way, the range of the
differences, and Holm over the group's cells; the recipe values every cell of a group
shares; the grader comparisons; the check of the posted values against the outputs; and
the decisions of the pre-registered questions on the cells that ran them (hypotheses.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from bodhi_lora.analysis import stats as st
from bodhi_lora.analysis.cell import _f, round6
from bodhi_lora.analysis.config import Group, ReportConfig
from bodhi_lora.analysis.extract import RESPONSE_ID, read_table
from bodhi_lora.analysis.hypotheses import hypotheses_summary
from bodhi_lora.paths import Project
from bodhi_lora.registry import CellConfig, Registry

RECIPE_FIELDS = ("r", "lora_alpha", "lora_dropout", "target_modules", "num_train_epochs", "train_batch_size",
                 "grad_accumulation_consistent", "effective_batch_consistent")


def members(group: Group, cells: list[CellConfig], reg: Registry) -> list[str]:
    """The cells of a group, in cross_cells order."""
    if group.cells:
        return [c.id for c in cells if c.id in group.cells]
    out = []
    for c in cells:
        w = group.where
        if w.status and c.report.status not in w.status:
            continue
        if w.model and c.model not in w.model:
            continue
        if w.family_waiver is not None and bool(c.grading.same_family_waiver) != w.family_waiver:
            continue
        if w.set_aside is not None and bool(c.report.set_aside) != w.set_aside:
            continue
        if c.id in group.exclude:
            continue
        out.append(c.id)
    return out


def _metric(stats: dict, contrast: str, metric: str) -> dict:
    return stats["contrasts"].get(contrast, {}).get("metrics", {}).get(metric, {})


def own_protocol(text: str, cell: CellConfig) -> str:
    """A config string with {protocol} standing for the cell's protocol id."""
    return text.replace("{protocol}", cell.protocol)


def _path(obj, path: str):
    """A dotted path into stats.json; None where any key is missing."""
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def group_summary(ids: list[str], all_stats: dict, cfg: ReportConfig, reg: Registry) -> dict:
    res = {"cells": ids, "n_cells": len(ids), "contrasts": {}, "recipe": {}, "askers_gap": {}}
    for gc in cfg.group_contrasts:
        for m in gc.metrics:
            # Each cell's own protocol: a cell on another protocol is measured, not skipped.
            rows = [(cid, _metric(all_stats[cid], own_protocol(gc.contrast, reg.cells[cid]), m)) for cid in ids]
            tested = [(cid, r) for cid, r in rows if r.get("diff") is not None]
            diffs = [r["diff"] for _, r in tested]
            ps = [r.get("p") if r.get("p") is not None else float("nan") for _, r in tested]
            holm = st.holm(ps)
            out = {"n_measured": len(tested), "not_measured": [cid for cid, r in rows if r.get("diff") is None],
                   "n_positive": sum(x > 0 for x in diffs), "n_negative": sum(x < 0 for x in diffs),
                   "n_zero": sum(x == 0 for x in diffs),
                   "min": _f(min(diffs)) if diffs else None, "max": _f(max(diffs)) if diffs else None,
                   "abs_min": _f(min(abs(x) for x in diffs)) if diffs else None,
                   "abs_max": _f(max(abs(x) for x in diffs)) if diffs else None,
                   "p_max": _f(max(ps)) if ps and all(np.isfinite(ps)) else None,
                   "p_holm": {cid: _f(h) for (cid, _), h in zip(tested, holm)},
                   "p_holm_max": _f(max(holm)) if holm and all(np.isfinite(holm)) else None}
            res["contrasts"].setdefault(gc.contrast, {})[m] = out
    # The recipe values that every cell of the group shares: the paper states one value.
    recipes = [all_stats[cid]["recipe"] for cid in ids]
    trained = [r for r in recipes if r.get("seeds_trained")]
    res["recipe"]["adapters"] = sum(len(r["seeds_trained"]) for r in trained)
    for k in RECIPE_FIELDS:
        vals = {json.dumps(r.get(k)) for r in trained}
        res["recipe"][k] = json.loads(vals.pop()) if len(vals) == 1 and trained else None
    best = [r.get(x) for r in trained for x in ("best_epoch_min", "best_epoch_max") if r.get(x) is not None]
    res["recipe"]["best_epoch_min"] = min(best) if best else None
    res["recipe"]["best_epoch_max"] = max(best) if best else None
    taus = {json.dumps((all_stats[cid]["filter"].get("recorded") or {}).get("tau")) for cid in ids}
    res["recipe"]["tau"] = json.loads(taus.pop()) if len(taus) == 1 else None
    # Askers gap of each condition: in how many cells the replies that ask score higher.
    conds = sorted({c for cid in ids for c in all_stats[cid]["askers_gap"]})
    for c in conds:
        gaps = [all_stats[cid]["askers_gap"].get(c, {}).get("gap") for cid in ids]
        gaps = [g for g in gaps if g is not None]
        res["askers_gap"][c] = {"n_measured": len(gaps), "n_positive": sum(g > 0 for g in gaps),
                                "n_negative": sum(g < 0 for g in gaps)}
    return res


def family_summary(ids: list[str], all_stats: dict, fam, reg: Registry, alpha: float) -> dict:
    """Holm over a group's cells of one p-value per cell (a correction family of
    docs/ANALYSIS_PLAN.md section 2.7). A cell without the value is listed, not tested."""
    rows = []
    for cid in ids:
        cell = reg.cells[cid]
        p = _path(all_stats[cid], own_protocol(fam.p, cell))
        diff = _path(all_stats[cid], own_protocol(fam.diff, cell))
        rows.append((cid, p, diff))
    tested = [(cid, p, d) for cid, p, d in rows if p is not None and d is not None and np.isfinite(p)]
    holm = st.holm([p for _, p, _ in tested])
    cells = {cid: {"p": _f(p), "diff": _f(d), "p_holm": _f(h)} for (cid, p, d), h in zip(tested, holm)}
    return {"group": fam.group, "p": fam.p, "diff": fam.diff, "cells": cells,
            "not_measured": [cid for cid, p, d in rows if cid not in cells],
            "n_up": sum(r["p_holm"] < alpha and r["diff"] > 0 for r in cells.values()),
            "n_down": sum(r["p_holm"] < alpha and r["diff"] < 0 for r in cells.values())}


def grader_comparison(project: Project, reg: Registry, cells: list[str], seed: int) -> dict:
    """Rubric scores of two cells on the prompts both drew in one seed.

    Pearson and Spearman r are over the shared prompts; the means and the differences
    from base are over each cell's own draw.
    """
    tables = []
    for cid in cells:
        p = project.cell(cid).table(seed)
        if not p.is_file():
            return {"not_measured": f"{cid} has no table for seed {seed}"}
        tables.append(read_table(p))
    a, b = tables
    graders = [reg.cells[c].grading.evaluator for c in cells]
    conds = [c for c in reg.cells[cells[0]].eval.conditions if c in set(b["condition"])]
    res = {"cells": cells, "graders": graders, "seed": seed, "conditions": {}}
    xs, ys = [], []
    for c in conds:
        sa = a[a["condition"] == c].set_index("prompt_id")["rubric_score"]
        sb = b[b["condition"] == c].set_index("prompt_id")["rubric_score"]
        shared = sorted(set(sa.index) & set(sb.index))
        res["conditions"][c] = {"n_shared": len(shared),
                                "pearson_r": _f(st.pearson_r(sa[shared], sb[shared])),
                                "mean_first": _f(sa.mean()), "mean_second": _f(sb.mean())}
        # How many shared prompts got the same reply in both runs: where the replies
        # differ, r mixes the graders with the replies (audit G5, 2026-10-11).
        if RESPONSE_ID in a and RESPONSE_ID in b:
            ra = a[a["condition"] == c].set_index("prompt_id")[RESPONSE_ID]
            rb = b[b["condition"] == c].set_index("prompt_id")[RESPONSE_ID]
            res["conditions"][c]["n_same_reply"] = int(sum(ra[p] == rb[p] for p in shared))
        xs += sa[shared].tolist()
        ys += sb[shared].tolist()
    res["pooled"] = {"pearson_r": _f(st.pearson_r(xs, ys)), "spearman_rho": _f(st.spearman_rho(xs, ys))}
    if "base" in res["conditions"]:
        base = res["conditions"]["base"]
        for c, r in res["conditions"].items():
            if c != "base":
                r["minus_base_first"] = _f(r["mean_first"] - base["mean_first"])
                r["minus_base_second"] = _f(r["mean_second"] - base["mean_second"])
    ns = {r["n_shared"] for r in res["conditions"].values()}
    res["n_shared"] = ns.pop() if len(ns) == 1 else None
    return res


def posted_check(root: Path, cfg: ReportConfig, all_stats: dict) -> Optional[dict]:
    """Each posted mean next to the generated seed mean; agrees = equal at the posted decimals."""
    if cfg.posted is None:
        return None
    posted = json.loads((root / cfg.posted.file).read_text())
    out = {"file": cfg.posted.file, "cells": {}, "mismatches": [], "analyses": posted.get("analyses", [])}
    for c in posted["cells"]:
        cid = cfg.posted.cells.get(c["id"])
        if cid is None or cid not in all_stats:
            out["cells"][c["id"]] = {"not_measured": "no analyzed cell for this posted cell"}
            continue
        rows = []
        for pcond, ms in sorted(c["conditions"].items()):
            cond = cfg.posted.conditions[pcond]
            for pm, v in sorted(ms.items()):
                if pm not in cfg.posted.metrics or v.get("mean") is None:
                    continue
                spec = cfg.posted.metrics[pm]
                g = all_stats[cid]["conditions"][spec.metric][cond]["seed_mean"]
                if g is None:
                    rows.append({"condition": pcond, "metric": pm, "posted": v["mean"], "generated": None,
                                 "agrees": None})
                    continue
                g *= spec.scale
                agrees = abs(g - v["mean"]) <= 0.5 * 10 ** -spec.decimals + 1e-9
                rows.append({"condition": pcond, "metric": pm, "posted": v["mean"], "generated": g,
                             "agrees": agrees})
                if not agrees:
                    out["mismatches"].append({"posted_cell": c["id"], "cell": cid, **rows[-1]})
        out["cells"][c["id"]] = {"cell": cid, "values": rows}
    return out


def analyze_cross(reg: Registry, project: Project, cfg: ReportConfig) -> dict:
    """Read every cross cell's stats.json; write and return <results root>/cross_cell/stats.json."""
    all_stats = {}
    for cid in cfg.cross_cells:
        p = project.cell(cid).analysis() / "stats.json"
        if not p.is_file():
            raise FileNotFoundError(f"{project.rel(p)} is missing; run python -m bodhi_lora analyze {cid}")
        all_stats[cid] = json.loads(p.read_text())
    # The cells of the pre-registered questions sit next to the cross cells (the plan's
    # choice 3), so they are read here but join no group.
    hyp_stats = dict(all_stats)
    for cid in sorted({c for h in cfg.hypotheses for c in [*h.cells, h.reference] if c} - set(all_stats)):
        p = project.cell(cid).analysis() / "stats.json"
        if p.is_file():
            hyp_stats[cid] = json.loads(p.read_text())
    cells = [reg.cells[cid] for cid in cfg.cross_cells]
    out = {"cells": cfg.cross_cells,
           "marks": {c.id: {"family_waiver": c.grading.same_family_waiver, "set_aside": c.report.set_aside,
                            "status": c.report.status} for c in cells},
           "groups": {}, "grader_comparisons": {}}
    for gid, g in cfg.groups.items():
        ids = members(g, cells, reg)
        out["groups"][gid] = {"token": g.token, "description": g.description,
                              **group_summary(ids, all_stats, cfg, reg)}
    out["families"] = {f.token: family_summary(out["groups"][f.group]["cells"], all_stats, f, reg, cfg.alpha)
                       for f in cfg.families}
    for gc in cfg.grader_comparisons:
        out["grader_comparisons"][gc.token] = {"description": gc.description,
                                               **grader_comparison(project, reg, gc.cells, gc.seed)}
    out["posted_check"] = posted_check(project.repo_root, cfg, all_stats)
    out["hypotheses"] = hypotheses_summary(hyp_stats, cfg, reg)
    out = round6(out)
    root = project.results_root / "cross_cell"
    root.mkdir(parents=True, exist_ok=True)
    (root / "stats.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out
