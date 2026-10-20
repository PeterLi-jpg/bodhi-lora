"""Macros and the table of the pre-registered questions (configs/report.yaml,
`hypotheses`), from cross_cell/stats.json (bodhi_lora/analysis/hypotheses.py).

Macro names: \\statHyp<question token><cell token><outcome><field>, for example
\\statHypHOneHBMistralActiveInquiryDiff, and \\statHyp<question><cell>Decision for the
plan's decision in words. A baseline of H4 adds its condition token after the cell token
(\\statHypHFourHBMistralBaseClarifyDecision).
"""
from __future__ import annotations

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.registry import Registry
from bodhi_lora.report.common import DIFF_DECIMALS, KIND_FORMAT, Names, marks, scaled
from bodhi_lora.report.latex import MISSING, escape, fmt, fmt_ci, fmt_p, tabular, token

# The kind of each outcome a rule returns, for its scale and decimals.
OUTCOME_KIND = {"active_inquiry": "rate", "context_seeking": "score02", "rubric_score": "score",
                "did": "rate", "complete_rate": "rate", "index": "rate", "red_flag": "score02",
                "precondition": "score02", "non_emergent_rate": "rate", "emergent_rate": "rate",
                "non_emergent_rubric_score": "score", "non_emergent_neg_any": "rate"}
# Printed names of the outcomes in the table.
OUTCOME_NAME = {"active_inquiry": "Active inquiry (pp)", "context_seeking": "Context-seeking (0--2)",
                "rubric_score": "Aggregate score (0--1)", "did": "Ask gap, incomplete $-$ complete (pp)",
                "complete_rate": "Asking on complete prompts (pp)",
                "index": "Asking index, physician labels (pp)",
                "red_flag": "Red flags (0--2), red flags first $-$ \\bodhi{}",
                "precondition": "Red flags (0--2), LoRA+\\bodhi{} $-$ LoRA",
                "non_emergent_rate": "Red-flag rate, non-emergent (pp)"}
# The outcomes the table prints for each rule, in order; the others get macros only.
TABLE_OUTCOMES = {"asking": ("active_inquiry", "context_seeking"), "cost": ("rubric_score",),
                  "paired_probe": ("did", "complete_rate"), "physician_index": ("index",),
                  "baselines": ("active_inquiry", "context_seeking"), "red_flags_first": ("precondition", "red_flag"),
                  "over_flagging": ("non_emergent_rate",)}


def _outcome_macros(rep, name: str, key: str, o: dict, where: str) -> None:
    scale, _ = KIND_FORMAT[OUTCOME_KIND[key]]
    dp = DIFF_DECIMALS[OUTCOME_KIND[key]]
    K = token(key)
    rep.add(f"{name}{K}Diff", fmt(o["diff"] * scale, dp, sign=True), f"{where}.diff")
    rep.add(f"{name}{K}CI", fmt_ci(scaled(o["ci95"], scale), dp), f"{where}.ci95, t over seeds")
    rep.add(f"{name}{K}P", fmt_p(o["p"]), f"{where}.p, paired t over seeds")
    if o.get("p_adj") is not None:
        rep.add(f"{name}{K}PAdj", fmt_p(o["p_adj"]), f"{where}.p_adj, Holm as the plan names it")


def hypothesis_macros(rep, cross: dict, reg: Registry, cfg: ReportConfig, src: str) -> None:
    n = Names(cfg)
    for hid, h in cross.get("hypotheses", {}).items():
        for cid, r in h["cells"].items():
            name = f"Hyp{h['token']}{reg.cells[cid].report.token}"
            where = f"{src}: hypotheses.{hid}.cells.{cid}"
            subs = r.get("baselines", {}).items() if "baselines" in r else [("", r)]
            for b, rr in subs:
                nm = name + (n.cond_token(b) if b else "")
                wb = f"{where}.baselines.{b}" if b else where
                if rr.get("decision"):
                    rep.add(f"{nm}Decision", escape(rr["decision"]), f"{wb}.decision, the plan's rule")
                for group in ("outcomes", "secondary"):
                    for key, o in (rr.get(group) or {}).items():
                        if o is not None and key in OUTCOME_KIND:
                            _outcome_macros(rep, nm, key, o, f"{wb}.{group}.{key}")
            for c, d in (r.get("delta") or {}).items():
                if d.get("seed_mean") is not None:
                    rep.add(f"{name}Delta{n.cond_token(c)}", fmt(d["seed_mean"] * 100, 1, sign=True),
                            f"{where}.delta.{c}.seed_mean (pp)")
                    rep.add(f"{name}Delta{n.cond_token(c)}CI", fmt_ci(scaled(d["ci95"], 100), 1),
                            f"{where}.delta.{c}.ci95 (pp)")


def judge_table(all_stats: dict, order: list[str], reg: Registry, cfg: ReportConfig, src: str) -> str:
    """H8 (docs/ANALYSIS_PLAN.md section 8): one row per cell and judge, for every analyzed
    cell whose record lists a judge that graded it: the judge's agreement with the
    evaluator on the rubric criteria, its decision on the two primary outcomes of H1
    (LoRA - Base), and how many dimension results of all the cell's contrasts it reverses."""
    n = Names(cfg)
    blocks = [[], []]
    for cid in order:
        stats = all_stats.get(cid)
        if not stats:
            continue
        cell = reg.cells[cid]
        # The cross cells first; the cells of the pre-registered questions in their own block.
        rerun = cid not in cfg.cross_cells
        for jid, j in (stats.get("judges") or {}).items():
            if "contrasts" not in j:
                continue
            m = j["contrasts"].get("lora - base", {}).get("metrics", {})
            decisions = [escape(m.get(k, {}).get("decision") or MISSING) for k in ("active_inquiry", "context_seeking")]
            tested = [r for c in j["contrasts"].values() for r in c["metrics"].values() if r.get("decision")]
            reversed_ = sum(r["decision"] == "reversed" for r in tested)
            if rerun:
                label = [_cell_text(cell, n), escape(cell.report.label.split(", ")[1])]
            else:
                label = n.cell_parts(cell.report.label)
            blocks[rerun].append([*label, escape(n.d.models.get(jid, jid)) + marks(cell),
                         fmt(j.get("criterion_kappa"), 2), fmt_ci(j.get("criterion_kappa_ci95"), 2, sign=False),
                         *decisions, f"{reversed_} of {len(tested)}"])
    header = [["Prompts", "Base model", "Judge", "$\\kappa$, criteria", "95\\% CI",
               "Active inquiry", "Context-seeking", "Reversed"]]
    return tabular("lllrrllr", header, [b for b in blocks if b], f"{src} (judges)")


def _cell_text(cell, n: Names) -> str:
    """The prompts of a cell (its label up to the base model, shortened by
    display.benchmarks) and its teaching protocol."""
    prompts = cell.report.label.split(", ")[0]
    prompts = n.d.benchmarks.get(prompts, prompts)
    return f"{escape(prompts)}, {n.d.protocols.get(cell.protocol, escape(cell.protocol))}"


def hypothesis_table(cross: dict, reg: Registry, cfg: ReportConfig, src: str, short: bool = False) -> str:
    """One row per question, cell and primary outcome, in the order of
    configs/report.yaml: the difference with its 95% t interval, its p (Holm-adjusted as the
    plan names it: within the contrast for the seven dimensions, across the question's
    cells for a family; the paired t where the plan names no correction) and the plan's
    decision on the cell's first row. short keeps the first outcome of each rule, for the
    main text."""
    n = Names(cfg)
    blocks = []
    for spec in cfg.hypotheses:
        h = cross.get("hypotheses", {}).get(spec.id)
        if h is None:
            continue
        rows = []
        rule = h["rule"]
        for cid in spec.cells:
            r = h["cells"][cid]
            label = _cell_text(reg.cells[cid], n)
            subs = [(b, r["baselines"][b]) for b in spec.baselines] if "baselines" in r else [("", r)]
            for b, rr in subs:
                cell_text = label + (f" vs {n.cond(b)}" if b else "")
                if spec.subset:
                    lab, _, group = spec.subset.partition("=")
                    cell_text += f" ({'' if group == 'positive' else 'not '}{n.label(lab)})"
                decision = escape(rr.get("decision") or MISSING)
                keys = TABLE_OUTCOMES[rule][:1] if short else TABLE_OUTCOMES[rule]
                for i, key in enumerate(keys):
                    o = (rr.get("outcomes") or {}).get(key)
                    kind = OUTCOME_KIND[key]
                    scale, dp = KIND_FORMAT[kind][0], DIFF_DECIMALS[kind]
                    diff = fmt(o["diff"] * scale, dp, sign=True) if o else MISSING
                    ci = fmt_ci(scaled(o["ci95"], scale), dp) if o else MISSING
                    p = fmt_p(o["p_adj"] if o.get("p_adj") is not None else o["p"]) if o else MISSING
                    rows.append([spec.id if not rows else "", cell_text if i == 0 else "", OUTCOME_NAME[key],
                                 diff, ci, p, decision if i == 0 else ""])
        blocks.append(rows)
    header = [["Question", "Cell", "Outcome", "Difference", "95\\% CI", "$p$", "Decision"]]
    return tabular("lllrrrl", header, blocks, f"{src} (hypotheses)")
