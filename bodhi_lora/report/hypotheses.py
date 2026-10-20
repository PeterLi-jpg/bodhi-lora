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
from bodhi_lora.report.latex import MISSING, escape, finite, fmt, fmt_ci, fmt_int, fmt_p, tabular, token

# The kind of each outcome a rule returns, for its scale and decimals.
OUTCOME_KIND = {"active_inquiry": "rate", "context_seeking": "score02", "rubric_score": "score",
                "did": "rate", "complete_rate": "rate", "index": "rate", "red_flag": "score02",
                "precondition": "score02", "non_emergent_rate": "rate", "emergent_rate": "rate",
                "non_emergent_rubric_score": "score", "non_emergent_neg_any": "rate",
                "precondition_full": "score02"}
# Printed names of the outcomes in the table.
OUTCOME_NAME = {"active_inquiry": "Active inquiry (pp)", "context_seeking": "Context-seeking (0--2)",
                "rubric_score": "Aggregate score (0--1)", "did": "Ask gap, incomplete $-$ complete (pp)",
                "complete_rate": "Asking on complete prompts (pp)",
                "index": "Asking index, physician labels (pp)",
                "red_flag": "Red flags (0--2), red flags first $-$ \\bodhi{}",
                "precondition": "Red flags (0--2), LoRA+\\bodhi{} $-$ LoRA",
                "non_emergent_rate": "Red-flag rate, non-emergent (pp)",
                "precondition_full": "Red flags (0--2), LoRA+\\bodhi{} $-$ LoRA, full context"}
# The outcomes the table prints for each rule, in order; the others get macros only.
TABLE_OUTCOMES = {"asking": ("active_inquiry", "context_seeking"), "cost": ("rubric_score",),
                  "paired_probe": ("did", "complete_rate"), "physician_index": ("index",),
                  "baselines": ("active_inquiry", "context_seeking"), "red_flags_first": ("precondition", "red_flag"),
                  "over_flagging": ("non_emergent_rate",),
                  "context_pair": ("precondition", "precondition_full", "red_flag")}


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
                prompts = cell.report.label.split(", ")[0]
                label = [f"{escape(n.d.benchmarks.get(prompts, prompts))}, "
                         f"{n.d.protocols.get(cell.protocol, escape(cell.protocol))}",
                         escape(cell.report.label.split(", ")[1])]
            else:
                label = n.cell_parts(cell.report.label)
            blocks[rerun].append([*label, escape(n.d.models.get(jid, jid)) + marks(cell),
                         fmt(j.get("criterion_kappa"), 2), fmt_ci(j.get("criterion_kappa_ci95"), 2, sign=False),
                         *decisions, f"{reversed_} of {len(tested)}"])
    header = [["Prompts", "Base model", "Judge", "$\\kappa$, criteria", "95\\% CI",
               "Active inquiry", "Context-seeking", "Reversed"]]
    return tabular("lllrrllr", header, [b for b in blocks if b], f"{src} (judges)")


def cost_table(all_stats: dict, reg: Registry, cfg: ReportConfig, src: str) -> str:
    """E08, E20 and C25: what each training cell of the pre-registered questions (the cells
    of the rule `cost`) cost to make and to run. Hours of the teacher traces and the filter
    (once per cell), training hours per seed (mean), and the median seconds per reply on an
    otherwise idle card, one reply at a time, for Base, the teacher (Base with the protocol)
    and LoRA, with the GPU each ran on."""
    n = Names(cfg)
    rows = []
    for spec in cfg.hypotheses:
        if spec.rule != "cost":
            continue
        for cid in spec.cells:
            stats = all_stats.get(cid)
            if not stats:
                continue
            cell = reg.cells[cid]
            st = stats.get("stage_times") or {}
            shared, per_seed = st.get("shared") or {}, st.get("per_seed") or {}

            def hours(r):
                return fmt(r["seconds"] / 3600, 1) if r and finite(r.get("seconds")) else MISSING

            train = [r["seconds"] for r in (per_seed.get("train") or {}).values() if finite(r.get("seconds"))]
            train_h = fmt(sum(train) / len(train) / 3600, 1) if train else MISSING
            bench = (stats.get("bench") or {}).get("conditions") or {}

            def median(cond):
                r = next(iter((bench.get(cond) or {}).values()), None)
                return fmt(r.get("median_s"), 1) if r else MISSING

            gpus = sorted({g for r in [*shared.values(), *[x for v in per_seed.values() for x in v.values()]]
                           for g in r.get("gpus", [])})
            rows.append([_cell_text(cell, n), hours(shared.get("traces")), hours(shared.get("filter")), train_h,
                         median("base"), median(f"base+{cell.protocol}"), median("lora"),
                         escape("; ".join(g.replace("NVIDIA ", "") for g in gpus)) or MISSING])
    header = [["Cell", "Traces (h)", "Filter (h)", "Train (h/seed)", "Base (s)", "Teacher (s)", "LoRA (s)", "GPUs"]]
    return tabular("lrrrrrrl", header, [rows], f"{src} (stage_times, bench)")


def _cell_text(cell, n: Names) -> str:
    """The prompts of a cell (its label up to the base model, shortened by
    display.benchmarks), its base model (display.models) and its teaching protocol."""
    prompts = cell.report.label.split(", ")[0]
    prompts = n.d.benchmarks.get(prompts, prompts)
    model = n.d.models.get(cell.model, cell.model)
    return f"{escape(prompts)}, {escape(model)}, {n.d.protocols.get(cell.protocol, escape(cell.protocol))}"


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
            # A cell that has not run is in the plan's list but not in the table.
            if r.get("not_measured"):
                continue
            label = _cell_text(reg.cells[cid], n)
            if rule == "context_pair":
                label += f", {fmt_int(reg.cells[cid].eval.serving.max_model_len)} tokens"
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
        if rows:
            blocks.append(rows)
    header = [["Question", "Cell", "Outcome", "Difference", "95\\% CI", "$p$", "Decision"]]
    return tabular("lllrrrl", header, blocks, f"{src} (hypotheses)")
