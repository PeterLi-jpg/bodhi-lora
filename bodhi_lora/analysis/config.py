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


class Discrimination(_Record):
    """P(response | label positive) - P(response | label negative) for each label.

    `response` is a 0/1 column of the response table: an evaluator rate (scored replies
    only) or a rate of every completion, such as the harm proxy neg_any."""
    response: str
    labels: list[RegistryId] = Field(min_length=1)


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


class GroupContrast(_Record):
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
        parse_condition(a)
        parse_condition(b)
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
    With skip_missing, a cell without a value in the first column has no row."""
    columns: list[CrossColumn] = Field(min_length=1)
    label_columns: list[str] = Field(default_factory=list, max_length=2)
    blocks: list[RegistryId] = Field(default_factory=list)
    skip_missing: bool = False


class CellDetailTables(_Record):
    """Every cell's metrics in each condition and a few contrasts, `per_table` cells to a
    table (tab_cell_detail_<n>). {protocol} stands for the cell's protocol id."""
    per_table: int = Field(gt=0)
    blocks: list[RegistryId] = Field(default_factory=list)
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
    length_reference: str
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
    cross_tables: dict[str, CrossTable] = Field(default_factory=dict)
    cell_detail_tables: Optional[CellDetailTables] = None

    @model_validator(mode="after")
    def _refs(self) -> "ReportConfig":
        parse_condition(self.length_reference)
        responses = [d.response for d in self.discrimination]
        if len(set(responses)) != len(responses):
            raise ValueError("a discrimination response repeats; list all its labels in one entry")
        tokens = [g.token for g in self.groups.values()] + [g.token for g in self.grader_comparisons]
        if len(set(tokens)) != len(tokens):
            raise ValueError("group and comparison tokens repeat")
        for gid, g in self.groups.items():
            # A listed cell outside cross_cells would drop out of its group without a word.
            outside = sorted(set(g.cells) - set(self.cross_cells))
            if outside:
                raise ValueError(f"group {gid} lists cells that are not in cross_cells: {outside}")
        named = [x.group for x in (*self.counts, *self.compares, *self.ranges)]
        named += [b for t in self.cross_tables.values() for b in t.blocks]
        named += self.cell_detail_tables.blocks if self.cell_detail_tables else []
        unknown = sorted(set(named) - set(self.groups))
        if unknown:
            raise ValueError(f"unknown groups {unknown}")
        for name in self.cross_tables:
            if not name.startswith("tab_"):
                raise ValueError(f"cross table {name!r}: a table name starts with tab_")
        return self


def load_report_config(path: Path) -> ReportConfig:
    return ReportConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
