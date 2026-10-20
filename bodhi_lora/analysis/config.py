"""The report config, configs/report.yaml, as typed records.

pydantic checks the file the way the registry checks its records: unknown keys are
errors, so a misspelled key fails at load time instead of being ignored.

The config defines no rule over the reply text. What a reply means comes from the
evaluator's grades, never from a keyword or regular expression (docs/CONVENTIONS.md
rule 15; tests/test_no_lexicon.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bodhi_lora.registry.schema import CellId, MacroToken, RegistryId, parse_condition


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bootstrap(_Record):
    resamples: int = Field(gt=0)
    seed: int


class ReplyHealth(_Record):
    """A reply longer than loop_min_chars whose zlib-compressed size is under
    loop_max_zlib_ratio of its own size counts as a repetition loop: a loop repeats the
    same text, which compresses far more than prose does. It reads no meaning from the
    reply, only how much it repeats."""
    loop_min_chars: int = Field(default=400, gt=0)
    loop_max_zlib_ratio: float = Field(default=0.15, gt=0, lt=1)


class Discrimination(_Record):
    """P(response | label positive) - P(response | label negative) for each label.

    `response` is a 0/1 column of the response table: an evaluator rate (scored replies
    only) or a rate of every completion, such as the harm proxy neg_any."""
    response: str
    labels: list[RegistryId] = Field(min_length=1)
    # Labels whose analysis the plan names for the cells drawn by the label only (H7 on
    # emergent-physician: the labelled cells). They count in a cell whose eval.ids.only
    # names the label, and stay unseen in every other cell.
    drawn_only: list[RegistryId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _drawn_only(self) -> "Discrimination":
        if not set(self.drawn_only) <= set(self.labels):
            raise ValueError(f"discrimination {self.response}: drawn_only names labels it does not list")
        return self


class Subset(_Record):
    """Every contrast of the cell again, on the prompts of one group of a label only
    (docs/ANALYSIS_PLAN.md section 9.3, H11: the clinician-user positive prompts). It
    runs in a cell whose eval.ids.only drew from that group, the cells the plan names."""
    label: RegistryId
    group: Literal["positive", "negative"]


def drawn_groups(cell) -> dict[str, set[str]]:
    """The label groups a cell's eval ids were drawn from (eval.ids.only), by label."""
    return {g.label: set(g.groups) for g in cell.eval.ids.only}


class Where(_Record):
    status: list[str] = Field(default_factory=list)
    model: list[RegistryId] = Field(default_factory=list)
    family_waiver: Optional[bool] = None
    set_aside: Optional[bool] = None


class Group(_Record):
    token: MacroToken
    description: str
    where: Optional[Where] = None
    exclude: list[CellId] = Field(default_factory=list)
    cells: list[CellId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_way(self) -> "Group":
        if (self.where is None) == (not self.cells):
            raise ValueError("a group takes either where (with exclude) or cells")
        return self


def check_condition(name: str) -> None:
    """parse_condition for a config string, where {protocol} stands for each cell's own
    protocol id."""
    parse_condition(name.replace("{protocol}", "protocol"))


class GroupContrast(_Record):
    """A contrast counted over a group's cells; {protocol} stands for each cell's own
    protocol id, so "lora - base+{protocol}" is every cell's adapter against its teacher."""
    contrast: str
    metrics: list[str]

    @property
    def conditions(self) -> tuple[str, str]:
        a, sep, b = self.contrast.partition(" - ")
        return a.strip(), b.strip()

    @model_validator(mode="after")
    def _contrast(self) -> "GroupContrast":
        a, b = self.conditions
        if not a or not b:
            raise ValueError(f"contrast {self.contrast!r}: write it as '<condition> - <condition>'")
        check_condition(a)
        check_condition(b)
        return self


class Family(_Record):
    """A correction family across cells (docs/ANALYSIS_PLAN.md section 2.7): Holm over
    the cells of `group` of the p-value at `p`, a dotted path into each cell's stats.json
    ({protocol} stands for the cell's protocol id). `diff` is the path of the difference
    that p tests; the report counts the cells where it moves each way with adjusted p
    below alpha. A table column reads a cell's adjusted p at
    cross.families.<token>.p_holm."""
    token: MacroToken
    group: RegistryId
    p: str
    diff: str


class Hypothesis(_Record):
    """One pre-registered question of docs/ANALYSIS_PLAN.md sections 3 to 9 on the cells
    that ran it. `rule` names the decision rule (bodhi_lora/analysis/hypotheses.py), and
    the cells are the question's correction family across cells (section 2.7), so Holm
    runs over them where the plan names a family. `subset` ("<label>=<group>") reads the
    contrasts of that subset of each cell (H11). `baselines` lists the prompt baselines
    that the rule `baselines` compares the adapter with (H4, E12)."""
    id: str = Field(pattern=r"^[A-Z][0-9]+$")
    token: MacroToken
    rule: Literal["asking", "cost", "paired_probe", "baselines", "red_flags_first", "over_flagging",
                  "physician_index"]
    cells: list[CellId] = Field(min_length=1)
    subset: Optional[str] = Field(default=None, pattern=r"^[a-z0-9-]+=(positive|negative)$")
    baselines: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _baselines(self) -> "Hypothesis":
        for b in self.baselines:
            check_condition(b)
        if (self.rule == "baselines") != bool(self.baselines):
            raise ValueError(f"hypothesis {self.id}: the rule baselines, and only it, lists baselines")
        return self


class GraderComparison(_Record):
    token: MacroToken
    description: str
    cells: list[CellId] = Field(min_length=2, max_length=2)
    seed: int


class PostedMetric(_Record):
    metric: str
    scale: float
    decimals: int = Field(ge=0)


class PostedCite(_Record):
    """A posted value the paper cites, where it withdraws or replaces it: `path` is a
    dotted path into the values of the analysis `analysis` of the posted record."""
    token: MacroToken
    analysis: str
    path: str
    decimals: int = Field(default=0, ge=0)
    sign: bool = False
    integer: bool = False


class Posted(_Record):
    file: str
    conditions: dict[str, str]
    metrics: dict[str, PostedMetric]
    cells: dict[str, CellId]
    cite: list[PostedCite] = Field(default_factory=list)


class CrossColumn(_Record):
    """One column of a cross-cell table: a number of each cell's stats.json.

    `value` and `ci` are dotted paths into stats.json; {protocol} stands for the cell's
    protocol id. With ci_style "pm", `ci` names a t interval, printed as plus or minus its
    half width. A bootstrap interval need not be symmetric, so it takes ci_style "range",
    which prints the value and both ends, or "interval", which prints only the two ends
    (for a column next to the value it qualifies). `star` names a p-value: the cell gets a star when it is below
    alpha. format "int" prints a count with a thousands mark; "span" prints `value` and
    `upper` as whole numbers, "a" when they are equal and "a--b" when not."""
    group: str = ""
    header: str
    value: str
    ci: Optional[str] = None
    ci_style: Literal["pm", "range", "interval"] = "pm"
    star: Optional[str] = None
    upper: Optional[str] = None
    format: Literal["num", "int", "span"] = "num"
    scale: float = 1.0
    decimals: int = Field(default=0, ge=0)
    sign: bool = False

    @model_validator(mode="after")
    def _span(self) -> "CrossColumn":
        if (self.format == "span") != (self.upper is not None):
            raise ValueError(f"column {self.header!r}: format span takes upper, and only span does")
        return self


class CrossTable(_Record):
    """A table with one row per cell of cross_cells. `blocks` lists groups, one block of
    rows each in cross_cells order; without blocks, the cells that are not set aside and
    then those that are. `label_columns` splits each cell's label at its last comma into
    that many columns (two: benchmark and base model); without it, one Cell column.
    With skip_missing, a cell without a value in the first column has no row.

    Blocks name groups, and a group may list its cells, so a cell added to cross_cells
    can fall outside every block. The report then stops, unless `leave_out` names the
    cell: a table leaves a cell out only on purpose."""
    columns: list[CrossColumn] = Field(min_length=1)
    label_columns: list[str] = Field(default_factory=list, max_length=2)
    blocks: list[RegistryId] = Field(default_factory=list)
    leave_out: list[CellId] = Field(default_factory=list)
    skip_missing: bool = False


class CellDetailTables(_Record):
    """Every cell's metrics in each condition and a few contrasts, `per_table` cells to a
    table (tab_cell_detail_<n>). {protocol} stands for the cell's protocol id.

    `conditions` are roles, not names: base+{protocol} is each cell's own teacher
    condition, so cells on different protocols share a column. When every cell of a table
    has one protocol, the headers name it; otherwise they say "P" and each cell's block
    row names its protocol. `leave_out` works as in CrossTable."""
    per_table: int = Field(gt=0)
    blocks: list[RegistryId] = Field(default_factory=list)
    leave_out: list[CellId] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=lambda: ["base", "base+{protocol}", "lora", "lora+{protocol}"],
                                  min_length=1)
    contrasts: list[str] = Field(min_length=1)
    metrics: list[str] = Field(min_length=1)


class Count(_Record):
    """The cells of a group where the listed metrics move in `direction` in a contrast,
    each with p below alpha: the seven dimensions under Holm within the contrast (test
    holm), or under the paired t (test p). Test bootstrap: the 95% interval of the pooled
    difference from the prompt-clustered bootstrap lies on that side of zero, which
    covers prompt sampling where every seed evaluated the same prompts. any: at least one
    metric, not all. The report also counts the base models and benchmarks of those cells."""
    token: MacroToken
    group: RegistryId
    contrast: str
    metrics: list[str] = Field(min_length=1)
    direction: Literal["up", "down"]
    test: Literal["holm", "p", "bootstrap"] = "holm"
    any: bool = False


class Compare(_Record):
    """The cells of a group where condition a's mean of a metric stands in `relation` to
    condition b's, both scaled and, with decimals, rounded first (to compare at a posted
    precision). {protocol} stands for the cell's protocol id."""
    token: MacroToken
    group: RegistryId
    metric: str
    a: str
    b: str
    relation: Literal[">", ">=", "<", "<="]
    scale: float = 1.0
    decimals: Optional[int] = None


class Range(_Record):
    """The smallest and largest value of one stats.json path over the cells of a group."""
    token: MacroToken
    group: RegistryId
    value: str
    scale: float = 1.0
    decimals: int = Field(ge=0)


class Display(_Record):
    """Names the tables print. A key without one prints as its id. `benchmarks` shortens
    the benchmark part of a cell label in a table with label_columns. `module_order` is
    the order in which the report names LoRA target modules (without _proj)."""
    weights: dict[str, str] = Field(default_factory=dict)
    protocols: dict[RegistryId, str] = Field(default_factory=dict)
    metrics: dict[str, str] = Field(default_factory=dict)
    labels: dict[RegistryId, str] = Field(default_factory=dict)
    benchmarks: dict[str, str] = Field(default_factory=dict)
    module_order: list[str] = Field(default_factory=list)


class ReportConfig(_Record):
    statuses: list[Literal["complete", "partial", "pilot", "planned"]]
    bootstrap: Bootstrap
    alpha: float = Field(gt=0, lt=1)
    discrimination: list[Discrimination]
    subsets: list[Subset] = Field(default_factory=list)
    length_reference: str
    reply_health: ReplyHealth = Field(default_factory=ReplyHealth)
    cross_cells: list[CellId]
    detail_cells: list[CellId] = Field(default_factory=list)
    groups: dict[RegistryId, Group]
    group_contrasts: list[GroupContrast]
    counts: list[Count] = Field(default_factory=list)
    compares: list[Compare] = Field(default_factory=list)
    ranges: list[Range] = Field(default_factory=list)
    grader_comparisons: list[GraderComparison] = Field(default_factory=list)
    posted: Optional[Posted] = None
    display: Display = Field(default_factory=Display)
    families: list[Family] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    cross_tables: dict[str, CrossTable] = Field(default_factory=dict)
    cell_detail_tables: Optional[CellDetailTables] = None

    @model_validator(mode="after")
    def _refs(self) -> "ReportConfig":
        parse_condition(self.length_reference)
        responses = [d.response for d in self.discrimination]
        if len(set(responses)) != len(responses):
            raise ValueError("a discrimination response repeats; list all its labels in one entry")
        tokens = ([g.token for g in self.groups.values()] + [g.token for g in self.grader_comparisons]
                  + [f.token for f in self.families] + [h.token for h in self.hypotheses])
        if len(set(tokens)) != len(tokens):
            raise ValueError("group, comparison and family tokens repeat")
        left_out = [c for t in self.cross_tables.values() for c in t.leave_out]
        left_out += self.cell_detail_tables.leave_out if self.cell_detail_tables else []
        unknown_cells = sorted(set(left_out) - set(self.cross_cells))
        if unknown_cells:
            raise ValueError(f"leave_out names cells that are not in cross_cells: {unknown_cells}")
        for gid, g in self.groups.items():
            # A listed cell outside cross_cells would drop out of its group without a word.
            outside = sorted(set(g.cells) - set(self.cross_cells))
            if outside:
                raise ValueError(f"group {gid} lists cells that are not in cross_cells: {outside}")
        named = [x.group for x in (*self.counts, *self.compares, *self.ranges, *self.families)]
        named += [b for t in self.cross_tables.values() for b in t.blocks]
        named += self.cell_detail_tables.blocks if self.cell_detail_tables else []
        unknown = sorted(set(named) - set(self.groups))
        if unknown:
            raise ValueError(f"unknown groups {unknown}")
        for name in self.cross_tables:
            if not name.startswith("tab_"):
                raise ValueError(f"cross table {name!r}: a table name starts with tab_")
        return self

    def labels_for(self, cell) -> set[str]:
        """The labels whose analysis runs in this cell, so its prompts table gets a column
        for each; every other label stays unseen (docs/ANALYSIS_PLAN.md section 12,
        2026-10-10). A drawn_only label and a subset count only where the cell drew by them."""
        drawn = drawn_groups(cell)
        out = {lab for spec in self.discrimination for lab in spec.labels
               if lab not in spec.drawn_only or lab in drawn}
        return out | {s.label for s in self.subsets if s.group in drawn.get(s.label, set())}


def load_report_config(path: Path) -> ReportConfig:
    return ReportConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
