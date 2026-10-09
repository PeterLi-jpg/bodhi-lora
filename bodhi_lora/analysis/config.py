"""The report config, configs/report.yaml, as typed records.

pydantic checks the file the way the registry checks its records: unknown keys are
errors, so a misspelled key fails at load time instead of being ignored.
"""
from __future__ import annotations

import re
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


class TextFeature(_Record):
    kind: Literal["question"]
    description: str
    pattern: str

    @model_validator(mode="after")
    def _shape(self) -> "TextFeature":
        re.compile(self.pattern)
        return self


class Discrimination(_Record):
    response: str
    labels: list[RegistryId]


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


class Posted(_Record):
    file: str
    conditions: dict[str, str]
    metrics: dict[str, PostedMetric]
    cells: dict[str, CellId]


class ReportConfig(_Record):
    statuses: list[Literal["complete", "partial", "pilot", "planned"]]
    bootstrap: Bootstrap
    text_features: dict[str, TextFeature]
    discrimination: Discrimination
    length_reference: str
    cross_cells: list[CellId]
    groups: dict[RegistryId, Group]
    group_contrasts: list[GroupContrast]
    grader_comparisons: list[GraderComparison] = Field(default_factory=list)
    posted: Optional[Posted] = None

    @model_validator(mode="after")
    def _refs(self) -> "ReportConfig":
        parse_condition(self.length_reference)
        tokens = [g.token for g in self.groups.values()] + [g.token for g in self.grader_comparisons]
        if len(set(tokens)) != len(tokens):
            raise ValueError("group and comparison tokens repeat")
        return self


def load_report_config(path: Path) -> ReportConfig:
    return ReportConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
