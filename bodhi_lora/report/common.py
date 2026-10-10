"""What the parts of the report share: printed names, paths into stats.json, and the
number formats of each kind of metric."""
from __future__ import annotations

from typing import Optional

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.registry import CellConfig, parse_condition
from bodhi_lora.report.latex import escape, finite, token

# Scale and decimals of a condition mean, by metric kind (bodhi_lora.analysis.cell.Metric).
KIND_FORMAT = {"score02": (1.0, 2), "rate": (100.0, 1), "count": (1.0, 2), "score": (1.0, 3),
               "chars": (1.0, 0), "tokens": (1.0, 0)}
# Decimals of a difference. A difference of 0-2 grades gets one more decimal than the
# grades: the paper's contrasts are often a tenth of a point or less.
DIFF_DECIMALS = {"score02": 3, "rate": 1, "count": 2, "score": 3, "chars": 0, "tokens": 0}


def get(obj, path: str):
    """A dotted path into stats.json; None where any key is missing."""
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def x(value, scale: float = 1.0):
    """value * scale, or None when the value is missing (a 0.0 stays 0.0)."""
    return value * scale if finite(value) else None


def scaled(ci, scale: float):
    return [v * scale if finite(v) else None for v in ci] if ci else None


def with_protocol(text: str, cell: CellConfig) -> str:
    """A config string with {protocol} standing for the cell's protocol id."""
    return text.replace("{protocol}", cell.protocol)


def contrast_pair(contrast: str) -> tuple[str, str]:
    a, _, b = contrast.partition(" - ")
    return a.strip(), b.strip()


class Names:
    """Tokens and printed names of conditions, contrasts, metrics and labels."""

    def __init__(self, cfg: ReportConfig):
        self.d = cfg.display

    def cond_token(self, condition: str) -> str:
        weights, proto = parse_condition(condition)
        return token(self.d.weights.get(weights, weights)) + (token(proto) if proto else "")

    def ctr_token(self, contrast: str) -> str:
        a, b = contrast_pair(contrast)
        return f"{self.cond_token(a)}vs{self.cond_token(b)}"

    def cond(self, condition: str) -> str:
        weights, proto = parse_condition(condition)
        name = self.d.weights.get(weights, weights)
        return f"{name}+{self.d.protocols.get(proto, escape(proto))}" if proto else name

    def ctr(self, contrast: str) -> str:
        a, b = contrast_pair(contrast)
        return f"{self.cond(a)} $-$ {self.cond(b)}"

    def metric(self, key: str) -> str:
        return self.d.metrics.get(key, escape(key))

    def metric_short(self, key: str) -> str:
        """The metric's name without its scale, for tables whose caption gives it."""
        return self.metric(key).replace(" (0--2)", "").replace(" (0--1)", "")

    def label(self, key: str) -> str:
        return self.d.labels.get(key, escape(key))

    def cell_parts(self, label: str) -> list[str]:
        """A cell label "<benchmark>, <base model>" as two table cells, the benchmark part
        shortened by display.benchmarks."""
        head, sep, tail = label.rpartition(", ")
        if not sep:
            return [escape(label), ""]
        return [escape(self.d.benchmarks.get(head, head)), escape(tail)]


def marks(cell: CellConfig) -> str:
    """dagger: the base model shares the evaluator's family; double dagger: set aside."""
    out = ""
    if cell.grading.same_family_waiver:
        out += "$^{\\dagger}$"
    if cell.report.set_aside:
        out += "$^{\\ddagger}$"
    return out


def star(text: str, p: Optional[float], alpha: float) -> str:
    """Append a star inside a number's \\ensuremath when p is below alpha."""
    if finite(p) and p < alpha and text.endswith("}"):
        return text[:-1] + "^{*}}"
    return text
