"""The report: macros and tables from the uniform analysis (docs/ARCHITECTURE.md section 7).

    report(reg, cfg, proj) -> Report      every analyzed cell's stats.json and the
                                          cross-cell stats.json -> macros, tables, record

It reads only the analysis outputs, never the raw outputs, and treats every cell the
same way. Each macro of a cell starts with the cell record's report.token, so
\\statHBMistralJulActiveInquiryLoRAvsBaseDiff is the LoRA - Base difference in active
inquiry of the cell whose token is HBMistralJul. A value the cell does not have (no
label on its benchmark, no tokenizer, one seed) gets no macro, and its table shows "--".

`make numbers` runs scripts/paper_stats.py, which calls report() and writes the macros
into paper/generated/numbers.tex with its own, the record into numbers.json under
"uniform_report", and the tables next to them:

    tab_cell_<token>.tex    one per analyzed cell: every condition and the two adapter
                            contrasts (LoRA - Base, LoRA - Base+protocol) on the seven
                            dimensions, the rubric score, the harm proxy, the label
                            rates, the askers gap, format leakage, length and tokens
    tab_cross_cell_checks.tex  one row per cell of report.cross_cells, the columns of
                            report.cross_table, the same for every cell
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.analysis.extract import SEVEN
from bodhi_lora.paths import Project
from bodhi_lora.registry import CellConfig, Registry, parse_condition
from bodhi_lora.report.latex import (MISSING, escape, finite, fmt, fmt_ci, fmt_int, fmt_p, fmt_pm, fmt_range,
                                     tabular, token)

# Printing scale and decimals of each metric kind (bodhi_lora.analysis.cell.Metric.kind).
KIND_FORMAT = {"score02": (1.0, 2), "rate": (100.0, 1), "count": (1.0, 2), "score": (1.0, 3),
               "chars": (1.0, 0), "tokens": (1.0, 0)}
# The metrics that get condition and contrast macros.
METRICS = (*SEVEN, "rubric_score", "neg_any", "chars", "tokens")
CROSS_TABLE = "tab_cross_cell_checks"


@dataclass
class Report:
    # (macro name without the \stat prefix, value, source comment), in emission order.
    macros: list[tuple[str, str, str]] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)
    record: dict = field(default_factory=dict)

    def add(self, name: str, value: str, source: str) -> None:
        if value == MISSING:
            return
        self.macros.append((name, value, source))


def _get(obj, path: str):
    """A dotted path into stats.json; None where any key is missing."""
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _x(value, scale: float = 1.0):
    """value * scale, or None when the value is missing (a 0.0 stays 0.0)."""
    return value * scale if finite(value) else None


def _scaled(ci, scale: float):
    return [x * scale if finite(x) else None for x in ci] if ci else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Names:
    """Tokens and printed names of conditions, contrasts, metrics and labels."""

    def __init__(self, cfg: ReportConfig):
        self.d = cfg.display

    def cond_token(self, condition: str) -> str:
        weights, proto = parse_condition(condition)
        return token(self.d.weights.get(weights, weights)) + (token(proto) if proto else "")

    def ctr_token(self, contrast: str) -> str:
        a, _, b = contrast.partition(" - ")
        return f"{self.cond_token(a)}vs{self.cond_token(b)}"

    def cond(self, condition: str) -> str:
        weights, proto = parse_condition(condition)
        name = self.d.weights.get(weights, weights)
        return f"{name}+{self.d.protocols.get(proto, escape(proto))}" if proto else name

    def ctr(self, contrast: str) -> str:
        a, _, b = contrast.partition(" - ")
        return f"{self.cond(a)} $-$ {self.cond(b)}"

    def metric(self, key: str) -> str:
        return self.d.metrics.get(key, escape(key))

    def label(self, key: str) -> str:
        return self.d.labels.get(key, escape(key))


def _metrics(stats: dict) -> list[str]:
    return [k for k in METRICS if k in stats["metrics"]]


# -- macros -----------------------------------------------------------------------------

def cell_macros(rep: Report, stats: dict, cell: CellConfig, cfg: ReportConfig, src: str) -> None:
    n = Names(cfg)
    T = cell.report.token
    conds, ctrs, seeds = stats["conditions_order"], stats["contrasts_order"], stats["seeds"]
    counts = stats["counts"]
    rep.add(f"{T}Seeds", fmt_int(len(seeds)), f"{src}: seeds")
    rep.add(f"{T}Prompts", fmt_int(counts["unique_prompts"]), f"{src}: counts.unique_prompts")
    for c in conds:
        rep.add(f"{T}Completions{n.cond_token(c)}", fmt_int(counts["per_condition"][c]["completions"]),
                f"{src}: counts.per_condition.{c}.completions")
    for m in _metrics(stats):
        scale, dp = KIND_FORMAT[stats["metrics"][m]["kind"]]
        M = token(m)
        for c in conds:
            r = stats["conditions"][m][c]
            if finite(r.get("seed_mean")):
                rep.add(f"{T}{M}{n.cond_token(c)}", fmt(r["seed_mean"] * scale, dp),
                        f"{src}: conditions.{m}.{c}.seed_mean, mean over {r['n_seeds']} seeds")
        for k in ctrs:
            r = stats["contrasts"][k]["metrics"][m]
            if not finite(r.get("diff")):
                continue
            name = f"{T}{M}{n.ctr_token(k)}"
            where = f"{src}: contrasts.{k}.metrics.{m}"
            rep.add(name + "Diff", fmt(r["diff"] * scale, dp, sign=True), f"{where}.diff, over {r['n_seeds']} seeds")
            rep.add(name + "CI", fmt_ci(_scaled(r["ci95"], scale), dp), f"{where}.ci95, t({r['df']})")
            if m in SEVEN:
                rep.add(name + "PHolm", fmt_p(r.get("p_holm_within_contrast")),
                        f"{where}.p_holm_within_contrast, Holm over the seven dimensions")
            else:
                rep.add(name + "P", fmt_p(r.get("p")), f"{where}.p, paired t over seeds")
            if r["n_seeds"] < len(seeds):
                rep.add(name + "Seeds", fmt_int(r["n_seeds"]), f"{where}.n_seeds")
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
                rep.add(f"{P}Index{C}", fmt(rc["discrimination"] * 100, 1, sign=True),
                        f"{where}.conditions.{c}.discrimination, pooled (pp)")
                rep.add(f"{P}Index{C}CI", fmt_ci(_scaled(rc["discrimination_ci95"], 100), 1),
                        "prompt-clustered bootstrap interval")
                for g in ("positive", "negative"):
                    G = "Pos" if g == "positive" else "Neg"
                    rep.add(f"{P}{G}Rate{C}", fmt(rc[f"{g}_rate"] * 100, 1),
                            f"{where}.conditions.{c}.{g}_rate, pooled (%)")
                    rep.add(f"{P}{G}Rate{C}CI", fmt_ci(_scaled(rc[f"{g}_rate_ci95"], 100), 1, sign=False),
                            "prompt-clustered bootstrap interval")
            for k in ctrs:
                rk, K = r["contrasts"][k], n.ctr_token(k)
                rep.add(f"{P}Index{K}Diff", fmt(rk["diff"] * 100, 1, sign=True), f"{where}.contrasts.{k}.diff (pp)")
                rep.add(f"{P}Index{K}CI", fmt_ci(_scaled(rk["diff_ci95"], 100), 1), "prompt-clustered bootstrap interval")
                for g in ("positive", "negative"):
                    G = "Pos" if g == "positive" else "Neg"
                    t = rk[f"{g}_rate"]["seed_test"]
                    if not finite(t.get("diff")):
                        continue
                    w = f"{where}.contrasts.{k}.{g}_rate.seed_test"
                    rep.add(f"{P}{G}Rate{K}Diff", fmt(t["diff"] * 100, 1, sign=True),
                            f"{w}.diff, over {t['n_seeds']} seeds (pp)")
                    rep.add(f"{P}{G}Rate{K}CI", fmt_ci(_scaled(t["ci95"], 100), 1), f"{w}.ci95, t({t['df']})")
                    rep.add(f"{P}{G}Rate{K}P", fmt_p(t.get("p")), f"{w}.p, paired t over seeds")
    for col, r in stats["leakage"].items():
        L = f"{T}Leak{token(r['protocol'])}{token(r['rule'])}"
        for c in conds:
            rc, where = r["conditions"][c], f"{src}: leakage.{col}.conditions.{c}"
            C = n.cond_token(c)
            rep.add(f"{L}{C}", fmt(_x(rc["leaked_share"], 100), 1), f"{where}.leaked_share (% of scored replies)")
            rep.add(f"{L}{C}N", fmt_int(rc["n_leaked"]), f"{where}.n_leaked (scored replies that leak)")
            rep.add(f"{L}{C}NClean", fmt_int(rc["n_clean"]), f"{where}.n_clean (scored replies that do not)")
            for g, G in (("leaked", "Leaked"), ("clean", "Clean")):
                rep.add(f"{L}{C}RedFlag{G}", fmt((rc.get(f"dimensions_{g}") or {}).get("red_flag"), 2),
                        f"{where}.dimensions_{g}.red_flag (0-2, pooled over seeds)")
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


def group_macros(rep: Report, cross: dict, cfg: ReportConfig, src: str) -> None:
    """Counts over the cell groups of configs/report.yaml (the cross-cell stats.json)."""
    n = Names(cfg)
    for gid, g in cross["groups"].items():
        G = f"Group{g['token']}"
        rep.add(f"{G}Cells", fmt_int(g["n_cells"]), f"{src}: groups.{gid}.n_cells")
        for k, metrics in g["contrasts"].items():
            for m, r in metrics.items():
                name = f"{G}{token(m)}{n.ctr_token(k)}"
                where = f"{src}: groups.{gid}.contrasts.{k}.{m}"
                rep.add(name + "Measured", fmt_int(r["n_measured"]), f"{where}.n_measured")
                rep.add(name + "Up", fmt_int(r["n_positive"]), f"{where}.n_positive (cells with a positive difference)")
                rep.add(name + "Down", fmt_int(r["n_negative"]), f"{where}.n_negative")
                rep.add(name + "PHolmMax", fmt_p(r.get("p_holm_max")), f"{where}.p_holm_max, Holm over the group's cells")


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
    n = Names(cfg)
    conds = stats["conditions_order"]
    # The adapter against the base and against the teacher prompt.
    ctrs = [k for k in stats["contrasts_order"] if k.startswith("lora - ")][:2]
    width = 1 + len(conds) + 2 * len(ctrs)
    header = ["", *(n.cond(c) for c in conds)]
    for k in ctrs:
        header += [n.ctr(k), "$p$"]

    def contrast_cells(r: dict, scale: float, dp: int, p_key: str) -> list[str]:
        return [fmt_pm(_x(r.get("diff"), scale), _scaled(r.get("ci95"), scale), dp),
                fmt_p(r.get(p_key))]

    def metric_row(m: str) -> list[str]:
        scale, dp = KIND_FORMAT[stats["metrics"][m]["kind"]]
        row = [n.metric(m)] + [fmt(_x(stats["conditions"][m][c]["seed_mean"], scale), dp)
                               for c in conds]
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
                idx.append(fmt(_x(r["conditions"][c]["discrimination"], 100), 1, sign=True))
                neg.append(fmt(_x(r["conditions"][c]["negative_rate"], 100), 1))
            for k in ctrs:
                idx += [fmt(_x(r["contrasts"][k]["diff"], 100), 1, sign=True), MISSING]
                neg += contrast_cells(r["contrasts"][k]["negative_rate"]["seed_test"], 100, 1, "p")
            labels += [idx, neg]
    other = [["Askers gap (aggregate score)"] + [fmt(stats["askers_gap"].get(c, {}).get("gap"), 3, sign=True)
                                                 for c in conds] + [MISSING] * (2 * len(ctrs))]
    for col, r in stats["leakage"].items():
        other.append([f"Format leak, {escape(r['rule'])} (\\%)"]
                     + [fmt(_x(r["conditions"][c]["leaked_share"], 100), 1) for c in conds]
                     + [MISSING] * (2 * len(ctrs)))
    size = [metric_row(m) for m in ("chars", "tokens") if m in stats["metrics"]]
    eng = stats["cost"]["conditions"]
    size.append(["Reply tokens (engine, one-pass conditions)"]
                + [fmt(eng[c]["tokens_engine_mean"], 0) if eng[c]["tokens_engine_n"] else MISSING for c in conds]
                + [MISSING] * (2 * len(ctrs)))
    blocks = [[title], seven, scores, labels, other, size]
    return tabular("l" + "r" * (width - 1), [header], [b for b in blocks if b], src)


def cross_label(label: str, mark: str = "") -> str:
    """A cell's label in the cross-cell table, on two lines to keep the first column
    narrow. Cell labels read "<benchmark>, <base model>"; a label without a comma stays
    on one line."""
    head, sep, tail = escape(label).rpartition(", ")
    if not sep:
        return escape(label) + mark
    return "\\shortstack[l]{" + head + ",\\\\" + tail + mark + "}"


def cross_table(all_stats: dict, reg: Registry, cfg: ReportConfig, src: str) -> str:
    cols = cfg.cross_table
    # Column groups: one spanning header each, ruled underneath.
    groups, rules, i = [""], [], 0
    while i < len(cols):
        j = i
        while j < len(cols) and cols[j].group == cols[i].group:
            j += 1
        if cols[i].group:
            groups.append(f"\\multicolumn{{{j - i}}}{{c}}{{{cols[i].group}}}")
            rules.append(f"\\cmidrule(lr){{{i + 2}-{j + 1}}}")
        else:
            groups += [""] * (j - i)
        i = j
    header = ["Cell", *(c.header for c in cols)]

    def row(cid: str) -> list[str]:
        cell, stats = reg.cells[cid], all_stats[cid]
        mark = ("$^{\\dagger}$" if cell.grading.same_family_waiver else "") + \
               ("$^{\\ddagger}$" if cell.report.set_aside else "")
        out = [cross_label(cell.report.label, mark)]
        for c in cols:
            value = _get(stats, c.value.format(protocol=cell.protocol))
            x = value * c.scale if finite(value) else None
            if c.ci:
                ci = _scaled(_get(stats, c.ci.format(protocol=cell.protocol)), c.scale)
                if c.ci_style == "range":
                    out.append(fmt_range(x, ci, c.decimals, c.sign, stack=True))
                else:
                    out.append(fmt_pm(x, ci, c.decimals, c.sign))
            else:
                out.append(fmt(x, c.decimals, c.sign))
        return out

    kept = [row(c) for c in cfg.cross_cells if not reg.cells[c].report.set_aside]
    aside = [row(c) for c in cfg.cross_cells if reg.cells[c].report.set_aside]
    note = ("dagger: base model in the evaluator's family; double dagger: set aside, reason in the cell's "
            "table; -- : the cell has no such value")
    if rules:
        # booktabs rules go right after the \\ that ends the group row.
        header[0] = "".join(rules) + " " + header[0]
    return tabular("l" + "r" * len(cols), ([groups] if rules else []) + [header], [b for b in (kept, aside) if b],
                   f"{src}. {note}")


# -- the report -------------------------------------------------------------------------

def report(reg: Registry, cfg: ReportConfig, proj: Project) -> Report:
    """Macros, tables and a record from every analyzed cell's stats.json."""
    from bodhi_lora.analysis import analyzable

    rep = Report()
    order = list(cfg.cross_cells) + [c for c in analyzable(reg, cfg) if c not in cfg.cross_cells]
    all_stats, cells_rec, missing = {}, {}, {}
    for cid in order:
        path = proj.cell(cid).analysis() / "stats.json"
        if not path.is_file():
            if cid in cfg.cross_cells:
                raise FileNotFoundError(f"{proj.rel(path)} is missing; run make analyze")
            missing[cid] = "no stats.json (the cell has no outputs in this checkout)"
            continue
        stats = json.loads(path.read_text())
        cell = reg.cells[cid]
        src = proj.rel(path)
        all_stats[cid] = stats
        cell_macros(rep, stats, cell, cfg, src)
        rep.tables[f"tab_cell_{cell.report.token}"] = cell_table(stats, cell, cfg, src)
        cells_rec[cid] = {"token": cell.report.token, "label": cell.report.label, "stats": src,
                          "sha256": _sha256(path), "in_cross_table": cid in cfg.cross_cells,
                          "set_aside": cell.report.set_aside, "family_waiver": cell.grading.same_family_waiver}
    cross_path = proj.results_root / "cross_cell" / "stats.json"
    cross_rec: Optional[dict] = None
    if cross_path.is_file():
        group_macros(rep, json.loads(cross_path.read_text()), cfg, proj.rel(cross_path))
        cross_rec = {"stats": proj.rel(cross_path), "sha256": _sha256(cross_path)}
    else:
        missing["cross_cell"] = f"{proj.rel(cross_path)} is missing; run make analyze"
    if cfg.cross_table:
        rep.tables[CROSS_TABLE] = cross_table(all_stats, reg, cfg, "each cell's analysis/stats.json")
    rep.record = {"cells": cells_rec, "cross_cell": cross_rec, "not_reported": missing,
                  "tables": sorted(rep.tables), "macros": len(rep.macros),
                  "cross_table_columns": [{"group": c.group, "header": c.header, "value": c.value, "ci": c.ci}
                                          for c in cfg.cross_table]}
    return rep
