"""The report over cells: tables with a row per cell, and the macros of the groups,
counts and comparisons that configs/report.yaml names.

Every number comes from a cell's analysis/stats.json or from the cross-cell stats.json
under the results root (group members, Holm over a group's cells, the grader
comparisons, the posted record). A table or count is a change in the config, not here.
"""
from __future__ import annotations

from bodhi_lora.analysis.config import CrossTable, ReportConfig
from bodhi_lora.analysis.extract import SEVEN
from bodhi_lora.registry import Registry
from bodhi_lora.report.common import (DIFF_DECIMALS, KIND_FORMAT, Names, contrast_pair, get, marks, scaled,
                                      star, with_protocol, x)
from bodhi_lora.report.latex import (MISSING, escape, finite, fmt, fmt_ci, fmt_int, fmt_p, fmt_pm, fmt_range,
                                     tabular, token)

RECIPE = (("tau", "Tau", 1), ("r", "Rank", None), ("lora_alpha", "Alpha", None), ("lora_dropout", "Dropout", 2),
          ("num_train_epochs", "Epochs", None), ("train_batch_size", "BatchPerDevice", None),
          ("grad_accumulation_consistent", "GradAccum", None), ("effective_batch_consistent", "EffectiveBatch", None),
          ("best_epoch_min", "BestEpochMin", None), ("best_epoch_max", "BestEpochMax", None),
          ("adapters", "Adapters", None))


def _ordered(cfg: ReportConfig, members: dict[str, list[str]], blocks: list[str], reg: Registry,
             leave_out: list[str] = (), name: str = "") -> list[list[str]]:
    """Cells in blocks: one per listed group, or the kept cells and then the set-aside ones.

    A cell of cross_cells that is in no block and not in leave_out stops the report: a
    group that lists its cells does not take in a cell added to cross_cells later, and
    the table would drop it without a word while a rule-based count still counts it."""
    if blocks:
        out = [[c for c in cfg.cross_cells if c in members[g]] for g in blocks]
        placed = {c for block in out for c in block}
        dropped = [c for c in cfg.cross_cells if c not in placed and c not in leave_out]
        if dropped:
            raise ValueError(f"{name}: cells of cross_cells in none of its blocks {blocks}: {dropped}. Add "
                             "each to a group the table prints, or to the table's leave_out "
                             "(configs/report.yaml)")
        return out
    kept = [c for c in cfg.cross_cells if not reg.cells[c].report.set_aside and c not in leave_out]
    aside = [c for c in cfg.cross_cells if reg.cells[c].report.set_aside and c not in leave_out]
    return [kept, aside]


def cross_table(spec: CrossTable, all_stats: dict, members: dict, reg: Registry, cfg: ReportConfig, src: str,
                name: str = "cross table") -> str:
    """A table with one row per cell, the same columns for every cell."""
    n = Names(cfg)
    cols = spec.columns
    lead = max(1, len(spec.label_columns))
    # Column groups: one spanning header each, ruled underneath.
    groups, rules, i = [""] * lead, [], 0
    while i < len(cols):
        j = i
        while j < len(cols) and cols[j].group == cols[i].group:
            j += 1
        if cols[i].group:
            groups.append(f"\\multicolumn{{{j - i}}}{{c}}{{{cols[i].group}}}")
            rules.append(f"\\cmidrule(lr){{{i + lead + 1}-{j + lead}}}")
        else:
            groups += [""] * (j - i)
        i = j
    header = (list(spec.label_columns) or ["Cell"]) + [c.header for c in cols]

    def value(cell, stats, c):
        v = get(stats, with_protocol(c.value, cell, stats))
        return v * c.scale if finite(v) else None

    def entry(cell, stats, c) -> str:
        v = value(cell, stats, c)
        if c.format == "int":
            return fmt_int(v) if v is not None else MISSING
        if c.format == "span":
            hi = get(stats, with_protocol(c.upper, cell, stats))
            if v is None or not finite(hi):
                return MISSING
            a, b = int(round(v)), int(round(hi * c.scale))
            return str(a) if a == b else f"{a}--{b}"
        if c.ci:
            ci = scaled(get(stats, with_protocol(c.ci, cell, stats)), c.scale)
            if c.ci_style == "interval":
                text = fmt_ci(ci, c.decimals, c.sign) if v is not None else MISSING
            elif c.ci_style == "range":
                text = fmt_range(v, ci, c.decimals, c.sign, stack=True)
            else:
                text = fmt_pm(v, ci, c.decimals, c.sign)
        else:
            text = fmt(v, c.decimals, c.sign)
        if c.star:
            text = star(text, get(stats, with_protocol(c.star, cell, stats)), cfg.alpha)
        return text

    def row(cid: str):
        cell, stats = reg.cells[cid], all_stats[cid]
        if spec.skip_missing and value(cell, stats, cols[0]) is None:
            return None
        if len(spec.label_columns) == 2:
            bench, model = n.cell_parts(cell.report.label)
            label = [bench, model + marks(cell)]
        else:
            label = [cross_label(cell.report.label, marks(cell))]
        return label + [entry(cell, stats, c) for c in cols]

    blocks = [[r for r in (row(c) for c in block) if r is not None]
              for block in _ordered(cfg, members, spec.blocks, reg, spec.leave_out, name)]
    note = ("dagger: base model in the evaluator's family; double dagger: set aside, reason in the cell's "
            "table; -- : the cell has no such value")
    if rules:
        # booktabs rules go right after the \\ that ends the group row.
        header[0] = "".join(rules) + " " + header[0]
    return tabular("l" * lead + "r" * len(cols), ([groups] if rules else []) + [header],
                   [b for b in blocks if b], f"{src}. {note}")


def cross_label(label: str, mark: str = "") -> str:
    """A cell's label on two lines, to keep the first column narrow. Cell labels read
    "<benchmark>, <base model>"; a label without a comma stays on one line."""
    head, sep, tail = escape(label).rpartition(", ")
    if not sep:
        return escape(label) + mark
    return "\\shortstack[l]{" + head + ",\\\\" + tail + mark + "}"


def _p_short(p) -> str:
    """A p-value for a dense table: two significant digits, "<0.001" below that."""
    if not finite(p):
        return MISSING
    return "\\ensuremath{<0.001}" if p < 0.001 else "\\ensuremath{" + f"{p:.2g}" + "}"


def cell_detail_tables(all_stats: dict, members: dict, reg: Registry, cfg: ReportConfig, src: str) -> dict[str, str]:
    """Every cell's metrics in each condition and a few contrasts, a few cells to a table.

    The columns are roles (base, base+{protocol}, lora, lora+{protocol}), filled from
    each cell's own conditions, so a cell on another protocol fills them too, and each
    contrast is that cell's own. When the cells of a table do not share one protocol, the
    headers say P and each cell's block row names its protocol."""
    spec = cfg.cell_detail_tables
    if spec is None:
        return {}
    n = Names(cfg)
    cells = [c for block in _ordered(cfg, members, spec.blocks, reg, spec.leave_out, "cell_detail_tables")
             for c in block]
    tables = {}
    for i in range(0, len(cells), spec.per_table):
        part = cells[i:i + spec.per_table]
        protocols = {reg.cells[c].protocol for c in part}
        shared = protocols.pop() if len(protocols) == 1 else None

        def name(role: str) -> str:
            # One protocol in the table: its name; several: P, named in each block row.
            return n.cond(role.replace("{protocol}", shared)) if shared else \
                n.cond(role.replace("{protocol}", "p")).replace("+p", "+P")

        def head(k: str) -> str:
            a, b = contrast_pair(k)
            # A contrast whose first condition carries a protocol is long: two lines.
            return f"\\shortstack[r]{{{name(a)}\\\\$-$ {name(b)}}}" if "+" in a else f"{name(a)} $-$ {name(b)}"

        header = [""] + [name(c) for c in spec.conditions]
        for k in spec.contrasts:
            header += [head(k), "$p_{\\text{Holm}}$"]
        width = len(header)
        blocks = []
        for cid in part:
            cell, stats = reg.cells[cid], all_stats[cid]
            conds = [with_protocol(c, cell) for c in spec.conditions]
            have = [c for c in conds if c in stats["conditions_order"]]
            graded = min((stats["conditions"][m][c]["n_seeds"] for c in have for m in spec.metrics if m in SEVEN),
                         default=0)
            seeds = len(stats["seeds"])
            note = f" ({graded} of {seeds} seeds have dimension grades)" if graded < seeds else ""
            proto = "" if shared else f", protocol {n.d.protocols.get(cell.protocol, escape(cell.protocol))}"
            block = [[f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{escape(cell.report.label)}}}{marks(cell)}{proto}{note}}}"]]
            for m in spec.metrics:
                scale, dp = KIND_FORMAT[stats["metrics"][m]["kind"]]
                row = [n.metric_short(m)] + [fmt(x(get(stats, f"conditions.{m}.{c}.seed_mean"), scale), dp)
                                             for c in conds]
                for k in spec.contrasts:
                    r = get(stats, f"contrasts.{with_protocol(k, cell)}.metrics.{m}") or {}
                    p = r.get("p_holm_within_contrast") if m in SEVEN else r.get("p")
                    row += [fmt_pm(x(r.get("diff"), scale), scaled(r.get("ci95"), scale), dp), _p_short(p)]
                block.append(row)
            blocks.append(block)
        tables[f"tab_cell_detail_{i // spec.per_table + 1}"] = tabular(
            "l" + "r" * (width - 1), [header], blocks, src)
    return tables


def family_macros(rep, cross: dict, all_stats: dict, reg: Registry, cfg: ReportConfig, src: str) -> None:
    """The correction families of configs/report.yaml: how many cells move each way with
    the Holm-adjusted p below alpha, and each cell's adjusted p. A cell's adjusted p is
    also put under cross.families.<token> in its stats, for the table columns."""
    for tok, f in (cross.get("families") or {}).items():
        where = f"{src}: families.{tok}"
        rep.add(f"Family{tok}Cells", fmt_int(len(f["cells"])), f"{where}.cells (cells tested)")
        rep.add(f"Family{tok}Up", fmt_int(f["n_up"]), f"{where}.n_up (positive, Holm p < {cfg.alpha})")
        rep.add(f"Family{tok}Down", fmt_int(f["n_down"]), f"{where}.n_down (negative, Holm p < {cfg.alpha})")
        for cid, r in f["cells"].items():
            all_stats[cid].setdefault("cross", {}).setdefault("families", {})[tok] = r
            rep.add(f"{reg.cells[cid].report.token}Family{tok}PHolm", fmt_p(r["p_holm"]),
                    f"{where}.cells.{cid}.p_holm, Holm over the family's cells")


# -- macros -----------------------------------------------------------------------------

def group_macros(rep, cross: dict, all_stats: dict, cfg: ReportConfig, src: str) -> None:
    """Counts over the cell groups of configs/report.yaml, and the recipe a group shares."""
    n = Names(cfg)
    for gid, g in cross["groups"].items():
        G = f"Group{g['token']}"
        rep.add(f"{G}Cells", fmt_int(g["n_cells"]), f"{src}: groups.{gid}.n_cells")
        for k, metrics in g["contrasts"].items():
            for m, r in metrics.items():
                name = f"{G}{token(m)}{n.ctr_token(k)}"
                where = f"{src}: groups.{gid}.contrasts.{k}.{m}"
                # The metric's kind sets how a difference prints; every cell has the same.
                kinds = {all_stats[c]["metrics"][m]["kind"] for c in g["cells"] if m in all_stats[c]["metrics"]}
                kind = kinds.pop() if len(kinds) == 1 else "score"
                scale, dp = KIND_FORMAT[kind][0], DIFF_DECIMALS[kind]
                rep.add(name + "Measured", fmt_int(r["n_measured"]), f"{where}.n_measured")
                rep.add(name + "Up", fmt_int(r["n_positive"]), f"{where}.n_positive (cells with a positive difference)")
                rep.add(name + "Down", fmt_int(r["n_negative"]), f"{where}.n_negative")
                rep.add(name + "AbsMin", fmt(x(r.get("abs_min"), scale), dp), f"{where}.abs_min (smallest size)")
                rep.add(name + "AbsMax", fmt(x(r.get("abs_max"), scale), dp), f"{where}.abs_max (largest size)")
                rep.add(name + "PMax", fmt_p(r.get("p_max")), f"{where}.p_max, the largest paired t p")
                rep.add(name + "PHolmMax", fmt_p(r.get("p_holm_max")), f"{where}.p_holm_max, Holm over the group's cells")
        for c, r in g["askers_gap"].items():
            name = f"{G}AskersGap{n.cond_token(c)}"
            rep.add(name + "Up", fmt_int(r["n_positive"]), f"{src}: groups.{gid}.askers_gap.{c}.n_positive")
            rep.add(name + "Down", fmt_int(r["n_negative"]), f"{src}: groups.{gid}.askers_gap.{c}.n_negative")
        _recipe(rep, G, g["recipe"], cfg, f"{src}: groups.{gid}.recipe")


def _recipe(rep, G: str, recipe: dict, cfg: ReportConfig, where: str) -> None:
    """The adapter recipe and filter tau that every cell of a group shares (the paper
    states one value). A field the cells do not share has no macro."""
    for key, name, dp in RECIPE:
        v = recipe.get(key)
        if isinstance(v, list):
            v = v[0] if len(v) == 1 else None
        if not finite(v):
            continue
        rep.add(f"{G}Recipe{name}", fmt(v, dp) if dp else fmt_int(v), f"{where}.{key}")
    mods = recipe.get("target_modules")
    if mods:
        short = [m.removesuffix("_proj") for m in mods]
        order = cfg.display.module_order
        ordered = [m for m in order if m in short] + sorted(m for m in short if m not in order)
        rep.add(f"{G}RecipeTargets", ", ".join(ordered), f"{where}.target_modules, without _proj")


def count_macros(rep, all_stats: dict, members: dict, reg: Registry, cfg: ReportConfig, groups: dict) -> None:
    """The counts, comparisons and ranges of configs/report.yaml, over each group's cells."""
    for spec in cfg.counts:
        G = f"Group{cfg.groups[spec.group].token}{spec.token}"
        hits = []
        for cid in members[spec.group]:
            cell, stats = reg.cells[cid], all_stats[cid]
            res = (stats["contrasts"].get(with_protocol(spec.contrast, cell)) or {}).get("metrics", {})
            sign = 1 if spec.direction == "up" else -1

            def moved(m: str) -> bool:
                r = res.get(m) or {}
                if spec.test == "bootstrap":
                    lo, hi = r.get("diff_ci95") or (None, None)
                    return finite(lo) and finite(hi) and (lo > 0 if sign > 0 else hi < 0)
                p = r.get("p_holm_within_contrast") if (spec.test == "holm" and m in SEVEN) else r.get("p")
                return finite(p) and p < cfg.alpha and finite(r.get("diff")) and sign * r["diff"] > 0

            if (any if spec.any else all)(moved(m) for m in spec.metrics):
                hits.append(cell)
        test = {"holm": f"Holm p < {cfg.alpha}", "p": f"paired t p < {cfg.alpha}",
                "bootstrap": "prompt-bootstrap 95% interval off zero"}[spec.test]
        what = (f"cells of group {spec.group} where {'any' if spec.any else 'each'} of {spec.metrics} "
                f"{'rose' if spec.direction == 'up' else 'fell'} in {spec.contrast}, {test}")
        rep.add(G, fmt_int(len(hits)), what)
        rep.add(G + "Models", fmt_int(len({c.model for c in hits})), f"base models of those cells")
        rep.add(G + "Benchmarks", fmt_int(len({c.benchmark for c in hits})), f"benchmarks of those cells")
    for spec in cfg.compares:
        G = f"Group{cfg.groups[spec.group].token}{spec.token}"
        k = 0
        for cid in members[spec.group]:
            cell, stats = reg.cells[cid], all_stats[cid]
            a = get(stats, f"conditions.{spec.metric}.{with_protocol(spec.a, cell)}.seed_mean")
            b = get(stats, f"conditions.{spec.metric}.{with_protocol(spec.b, cell)}.seed_mean")
            if not (finite(a) and finite(b)):
                continue
            a, b = a * spec.scale, b * spec.scale
            if spec.decimals is not None:
                a, b = round(a, spec.decimals), round(b, spec.decimals)
            k += {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[spec.relation]
        rep.add(G, fmt_int(k), f"cells of group {spec.group} where {spec.metric} {spec.a} {spec.relation} {spec.b}"
                + (f" at {spec.decimals} decimals" if spec.decimals is not None else ""))
    for spec in cfg.ranges:
        G = f"Group{cfg.groups[spec.group].token}{spec.token}"
        vals = [get(all_stats[cid], spec.value) for cid in members[spec.group]]
        vals = [v * spec.scale for v in vals if finite(v)]
        if vals:
            rep.add(G + "Min", fmt(min(vals), spec.decimals), f"smallest {spec.value} over group {spec.group}")
            rep.add(G + "Max", fmt(max(vals), spec.decimals), f"largest {spec.value} over group {spec.group}")
    rep.add("CrossCells", fmt_int(len(cfg.cross_cells)), "configs/report.yaml: cross_cells")


def grader_macros(rep, cross: dict, reg: Registry, cfg: ReportConfig, src: str) -> None:
    """Two cells that graded the same model with different evaluators (grader_comparisons):
    r over the shared prompts, each cell's mean over its own draw, and each condition minus
    base under each evaluator. First and Second are the cells in the order the config
    lists them."""
    n = Names(cfg)
    for tok, r in cross["grader_comparisons"].items():
        if "not_measured" in r:
            continue
        where = f"{src}: grader_comparisons.{tok}"
        rep.add(f"{tok}Shared", fmt_int(r["n_shared"]), f"{where}.n_shared (prompts both cells drew in the seed)")
        rep.add(f"{tok}RPooled", fmt(r["pooled"]["pearson_r"], 2), f"{where}.pooled.pearson_r")
        rep.add(f"{tok}RhoPooled", fmt(r["pooled"]["spearman_rho"], 2), f"{where}.pooled.spearman_rho")
        for c, rc in r["conditions"].items():
            C = n.cond_token(c)
            rep.add(f"{tok}R{C}", fmt(rc["pearson_r"], 2), f"{where}.conditions.{c}.pearson_r")
            if "n_same_reply" in rc:
                rep.add(f"{tok}SameReply{C}", fmt_int(rc["n_same_reply"]),
                        f"{where}.conditions.{c}.n_same_reply (shared prompts with the same reply in both runs)")
            for side, S in (("first", "First"), ("second", "Second")):
                rep.add(f"{tok}Mean{S}{C}", fmt(rc[f"mean_{side}"], 3), f"{where}.conditions.{c}.mean_{side}")
                if f"minus_base_{side}" in rc:
                    rep.add(f"{tok}{C}MinusBase{S}", fmt(rc[f"minus_base_{side}"], 3, sign=True),
                            f"{where}.conditions.{c}.minus_base_{side}")


def posted_macros(rep, cross: dict, cfg: ReportConfig, src: str) -> None:
    """The posted values the paper cites where it withdraws or replaces them."""
    if cfg.posted is None or not cross.get("posted_check"):
        return
    analyses = {a["id"]: a["values"] for a in cross["posted_check"]["analyses"]}
    for c in cfg.posted.cite:
        v = get(analyses.get(c.analysis, {}), c.path)
        if not finite(v):
            raise ValueError(f"posted value {c.analysis}.{c.path} is not in {cfg.posted.file}")
        rep.add(f"Posted{c.token}", fmt_int(v) if c.integer else fmt(v, c.decimals, c.sign),
                f"{cfg.posted.file} (via {src}): {c.analysis}.{c.path}, as posted")
