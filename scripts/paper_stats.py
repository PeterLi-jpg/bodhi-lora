"""Compute every number the paper reports for the submitted MedGemma cell.

One script, one pass, no hand-typed numbers. It reads the committed eval
outputs and writes three kinds of file into ``paper/generated/``:

  numbers.json   every statistic, nested, full precision (rounded to 6 dp)
  numbers.tex    one ``\\newcommand`` per number the prose cites, with a
                 comment on each that names its source
  tab_*.tex      complete ``tabular`` environments; the paper keeps its own
                 ``table`` float, caption and label and ``\\input``s these

Inputs
------
Required (tracked in git):
  results_modal/seed_<N>/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json
      HealthBench grader output, one row per completion.
  results_modal/seed_<N>/epistemic_scores.json
      Epistemic grader output, one row per graded completion.

Optional (each table that needs one is skipped with a message if it is absent):
  data/raw/healthbench_hard.jsonl          theme tags (theme table)
  data/sft_qwen_graded/seed_7_train.jsonl  filter scores (tau table)
  results_modal/irr/kappa_results.json     physician kappa (kappa table)
  rebuttal/results/*.json                  rebuttal analyses (leakage, odds ratios)
  data/rebuttal_cells/posted_values.json   rebuttal cells (generality table)

Statistical units
-----------------
A seed trains one adapter and evaluates it on its own 200-prompt draw, and
all four conditions in a seed share that draw. So the seed is the unit of
inference: every test and CI here is over the 5 per-seed means (paired across
conditions within a seed, t with n_seeds - 1 df). Pooled response-level
numbers (Cohen's d, the length and theme tables) are descriptive. Their CIs
come from a bootstrap that resamples prompt ids, because the 5 draws overlap
(667 unique prompts in 1,000 per condition) and responses to one prompt are
not independent.

Determinism: no timestamps, no absolute paths, fixed bootstrap seed, sorted
iteration. Running the script twice on the same inputs gives identical bytes.

Usage:
    python3 scripts/paper_stats.py [--healthbench PATH] [--out-dir paper/generated]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")
# Macro-name token and table label for each condition.
COND_TOKEN = {
    "base_no_wrapper": "Base",
    "base_bodhi": "BaseCoT",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRACoT",
}
COND_LABEL = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Base+\\bodhi{}",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+\\bodhi{}",
}

# (token, condition a, condition b): the contrast is a minus b.
CONTRASTS = (
    ("LoRAvsBase", "lora_no_wrapper", "base_no_wrapper"),
    ("BaseCoTvsBase", "base_bodhi", "base_no_wrapper"),
    ("LoRACoTvsBase", "lora_bodhi", "base_no_wrapper"),
    ("LoRAvsBaseCoT", "lora_no_wrapper", "base_bodhi"),
    ("LoRACoTvsLoRA", "lora_bodhi", "lora_no_wrapper"),
)

# Every metric the paper reports. Fields:
#   key       column name in the response table
#   token     macro-name token
#   label     row label in tables
#   kind      "score02" (0-2 grade), "rate" (0/1, shown in %), "count",
#             "hb" (0-1 rubric score), "chars"
#   source    "epistemic" (scored responses only) or "hb" (all completions)
#   worst     value a failed or missing grade takes in the worst-case check:
#             the undesirable end of the scale for that metric
METRICS = (
    dict(key="uncertainty_ack", token="UncertaintyAck", label="Uncertainty ack.\\ (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="active_inquiry", token="ActiveInquiry", label="Active inquiry (\\%)",
         kind="rate", source="epistemic", worst=0.0),
    dict(key="context_seeking", token="ContextSeeking", label="Context-seeking (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="red_flag", token="RedFlag", label="Red-flag identification (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="scope_bounding", token="ScopeBounding", label="Scope bounding (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="specificity", token="Specificity", label="Specificity (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="hedging_quality", token="HedgingQuality", label="Hedging quality (0--2)",
         kind="score02", source="epistemic", worst=0.0),
    dict(key="questions", token="Questions", label="Questions per response",
         kind="count", source="epistemic", worst=0.0),
    dict(key="red_flag_rate", token="RedFlagRate", label="Red-flag rate (\\%)",
         kind="rate", source="epistemic", worst=0.0),
    dict(key="scope_bounded_rate", token="ScopeBoundedRate", label="Scope-bounded rate (\\%)",
         kind="rate", source="epistemic", worst=0.0),
    dict(key="specificity_rate", token="SpecificityRate", label="Specificity rate (\\%)",
         kind="rate", source="epistemic", worst=0.0),
    # Lower is better, so the worst case counts a failed grade as a disclaimer.
    dict(key="blanket_disclaimer", token="BlanketDisclaimer", label="Blanket disclaimer (\\%)",
         kind="rate", source="epistemic", worst=1.0),
    dict(key="appropriate_hedging", token="AppropriateHedging", label="Appropriate hedging (\\%)",
         kind="rate", source="epistemic", worst=0.0),
    dict(key="healthbench", token="HealthBench", label="HealthBench score (0--1)",
         kind="hb", source="hb", worst=None),
    dict(key="chars", token="ResponseChars", label="Response length (chars)",
         kind="chars", source="hb", worst=None),
)
METRIC = {m["key"]: m for m in METRICS}
# The paper's seven behavioral dimensions. This is the Holm family within a contrast.
SEVEN_DIMS = ("uncertainty_ack", "active_inquiry", "context_seeking", "red_flag",
              "scope_bounding", "hedging_quality", "specificity")

AXES = ("accuracy", "communication_quality", "completeness", "context_awareness",
        "instruction_following")

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
TAU_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# The tau table's pool column needs the share of the 4,000 generated traces that
# passed the filter. That count is not in the repo (PROV-11). The filter grades
# greedily, so every seed keeps the same traces; the paper's score-distribution
# caption gives 3,129 of 4,000 for seed 42. Used only for that one column.
FILTER_KEPT, FILTER_POOL = 3129, 4000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_seeds(results_dir: Path) -> list[int]:
    seeds = sorted(int(p.name.removeprefix("seed_")) for p in results_dir.glob("seed_*")
                   if p.is_dir() and p.name.removeprefix("seed_").isdigit())
    if len(seeds) < 2:
        raise SystemExit(f"paper_stats: need >=2 seed dirs under {results_dir}, found {seeds}")
    return seeds


def _hb_score(criteria: list[dict], flip_failed_negatives: bool = False) -> float:
    """Normalized rubric score, same formula as filter_traces.grade_trace.

    With ``flip_failed_negatives`` a negative criterion whose grade failed to
    parse counts as met. The grader records a failed parse as not met, which is
    the conservative reading for a positive criterion but the generous one for
    a negative criterion, so this gives the worst case.
    """
    pos = sum(c["points"] for c in criteria if c["points"] > 0)
    neg = sum(c["points"] for c in criteria if c["points"] < 0)
    earned = 0.0
    for c in criteria:
        met = c["criteria_met"]
        if flip_failed_negatives and c.get("parse_failed") and c["points"] < 0:
            met = True
        if met:
            earned += c["points"]
    rng = pos - neg
    return (earned - neg) / rng if rng > 0 else 0.0


def _epistemic_columns(scores: dict) -> dict:
    """Flatten one epistemic grade into the metric columns."""
    return {
        "uncertainty_ack": float(scores["uncertainty_acknowledgment"]),
        "active_inquiry": float(bool(scores["active_inquiry"])),
        "context_seeking": float(scores["context_seeking"]),
        "red_flag": float(scores["red_flag_identification"]),
        "scope_bounding": float(scores["scope_bounding"]),
        "specificity": float(scores["specificity"]),
        "hedging_quality": float(scores["hedging"]),
        "questions": float(scores["n_questions"] or 0),
        # Rate definitions match eval_epistemic.aggregate.
        "red_flag_rate": float(scores["red_flag_identification"] >= 2),
        "scope_bounded_rate": float(scores["scope_bounding"] >= 2),
        "specificity_rate": float(bool(scores["has_concrete_numbers"])),
        "blanket_disclaimer": float(bool(scores["is_blanket_disclaimer"])),
        "appropriate_hedging": float(bool(scores["appropriate_hedging"])),
    }


def load_responses(results_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """One row per HealthBench completion, joined to its epistemic grade.

    ``ep_status`` is "scored", "parse_failure" (the grader output did not
    parse) or "missing" (eval_epistemic hit a grade error and wrote no row).
    Epistemic metric columns are NaN unless the row is scored.
    """
    rows = []
    for seed in seeds:
        seed_dir = results_dir / f"seed_{seed}"
        with open(seed_dir / "epistemic_scores.json") as f:
            ep = {c["name"]: {e["prompt_id"]: e for e in c["examples"]}
                  for c in json.load(f)["configs"]}
        for cfg in CONDITIONS:
            with open(seed_dir / f"{cfg}.json") as f:
                hb = json.load(f)
            hb_ids = [r["prompt_id"] for r in hb["results"]]
            if len(hb_ids) != len(set(hb_ids)):
                raise SystemExit(f"duplicate prompt_id in {seed_dir}/{cfg}.json")
            extra = set(ep.get(cfg, {})) - set(hb_ids)
            if extra:
                raise SystemExit(f"{len(extra)} epistemic rows without a completion in seed {seed} {cfg}")
            for r in hb["results"]:
                criteria = r["criteria_results"]
                score = _hb_score(criteria)
                # Guard: our recompute must equal the stored score, or the
                # worst-case score below is not comparable.
                if not math.isclose(score, r["score"], abs_tol=1e-9):
                    raise SystemExit(f"score recompute mismatch seed {seed} {cfg} {r['prompt_id']}")
                row = {
                    "seed": seed,
                    "config": cfg,
                    "prompt_id": r["prompt_id"],
                    "healthbench": float(r["score"]),
                    "healthbench_worst": _hb_score(criteria, flip_failed_negatives=True),
                    "chars": float(len(r["response"])),
                    "tokens": float(r["response_token_count"]) if r.get("response_token_count") else np.nan,
                    "hb_items": len(criteria),
                    "hb_item_parse_failures": sum(bool(c.get("parse_failed")) for c in criteria),
                }
                for axis in AXES:
                    row[f"axis_{axis}"] = r["tag_scores"].get(f"axis:{axis}", np.nan)
                e = ep.get(cfg, {}).get(r["prompt_id"])
                if e is None:
                    row["ep_status"] = "missing"
                elif e.get("parse_failure") or e.get("scores") is None:
                    row["ep_status"] = "parse_failure"
                else:
                    row["ep_status"] = "scored"
                    row.update(_epistemic_columns(e["scores"]))
                rows.append(row)
    df = pd.DataFrame(rows)
    for m in METRICS:
        if m["key"] not in df:
            df[m["key"]] = np.nan
    return df


def load_themes(path: Path) -> dict[str, str]:
    """prompt_id -> HealthBench theme (each Hard prompt carries exactly one)."""
    themes = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            tags = [t.split(":", 1)[1] for t in ex.get("example_tags", []) if t.startswith("theme:")]
            if tags:
                themes[ex["prompt_id"]] = tags[0]
    return themes


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def metric_frame(df: pd.DataFrame, key: str, worst_case: bool = False) -> pd.DataFrame:
    """Rows that carry this metric, with the value in column ``v``.

    Normal: epistemic metrics use scored rows only; HealthBench metrics use
    every completion. Worst case: every completion counts, and a failed or
    missing epistemic grade takes the metric's worst value.
    """
    m = METRIC[key]
    if m["source"] == "hb":
        col = "healthbench_worst" if (worst_case and key == "healthbench") else key
        out = df[["seed", "config", "prompt_id", col]].rename(columns={col: "v"})
        return out
    if not worst_case:
        sub = df[df["ep_status"] == "scored"]
        return sub[["seed", "config", "prompt_id", key]].rename(columns={key: "v"})
    out = df[["seed", "config", "prompt_id", key]].rename(columns={key: "v"}).copy()
    out["v"] = out["v"].fillna(m["worst"])
    return out


def seed_means(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Seed x condition table of means (rows: seeds in order, cols: CONDITIONS)."""
    t = frame.groupby(["seed", "config"])["v"].mean().unstack("config")
    return t.reindex(index=seeds, columns=list(CONDITIONS))


def t_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t interval with n - 1 df (scipy). Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(values.std(ddof=1)) / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se
    return mean, mean - half, mean + half


def paired_seed_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Seed-level paired comparison a - b: mean difference, t CIs, paired t."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, lo, hi = t_interval(diffs, 0.95)
    _, lo90, hi90 = t_interval(diffs, 0.90)
    if np.allclose(diffs, diffs[0]):
        # No spread across seeds (e.g. both conditions at 0%): t is undefined.
        t_stat, p = float("nan"), float("nan")
    else:
        res = stats.ttest_rel(a, b)
        t_stat, p = float(res.statistic), float(res.pvalue)
    return {"diff": mean, "ci95": [lo, hi], "ci90": [lo90, hi90], "t": t_stat,
            "df": len(diffs) - 1, "p": p, "per_seed_diff": diffs.tolist()}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for a - b with the pooled SD (same formula as plot_postprocessing)."""
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


class ClusterBootstrap:
    """Bootstrap over prompt ids, shared by every pooled comparison.

    Each resample draws prompt ids with replacement and keeps every response to
    a drawn prompt, in every seed and condition. One weight matrix (resamples x
    prompts) is drawn once from a fixed seed and reused, so all CIs come from
    the same resamples and the output is deterministic.
    """

    def __init__(self, prompt_ids: list[str], resamples: int, seed: int):
        self.prompt_ids = sorted(prompt_ids)
        self.index = {p: i for i, p in enumerate(self.prompt_ids)}
        n = len(self.prompt_ids)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(resamples, n))
        # Count how often each prompt is drawn in each resample.
        self.weights = np.zeros((resamples, n))
        np.add.at(self.weights, (np.repeat(np.arange(resamples), n), draws.ravel()), 1.0)

    def _sums(self, frame: pd.DataFrame):
        idx = frame["prompt_id"].map(self.index).to_numpy()
        v = frame["v"].to_numpy(dtype=float)
        size = len(self.prompt_ids)
        cnt = np.bincount(idx, minlength=size).astype(float)
        s = np.bincount(idx, weights=v, minlength=size)
        q = np.bincount(idx, weights=v * v, minlength=size)
        # einsum rather than matmul: same result, but it does not go through
        # BLAS, which on macOS Accelerate raises spurious FP warnings and can
        # change the last bits between machines.
        sums = np.einsum("bp,pk->bk", self.weights, np.stack([cnt, s, q], axis=1))
        return sums[:, 0], sums[:, 1], sums[:, 2]

    def diff_and_d(self, fa: pd.DataFrame, fb: pd.DataFrame) -> dict:
        """Percentile 95% CIs for the pooled mean difference and Cohen's d (a - b)."""
        na, sa, qa = self._sums(fa)
        nb, sb, qb = self._sums(fb)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = sa / na, sb / nb
            va = (qa - sa * sa / na) / (na - 1)
            vb = (qb - sb * sb / nb) / (nb - 1)
            sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
            diff = ma - mb
            d = diff / sp
        d = np.where(sp > 0, d, np.nan)

        def pct(x):
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return [float("nan"), float("nan")]
            lo, hi = np.percentile(x, [2.5, 97.5])
            return [float(lo), float(hi)]

        return {"diff_ci95": pct(diff), "d_ci95": pct(d)}


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (statsmodels); NaN entries stay NaN and are left out."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fmt(x, dp: int, sign: bool = False) -> str:
    """Fixed-point number wrapped for text or math mode; "n/a" if undefined."""
    if not _finite(x):
        return "n/a"
    s = f"{x:+.{dp}f}" if sign else f"{x:.{dp}f}"
    # "-0.00" reads as a real negative; drop the sign when the value rounds to zero.
    if float(s) == 0:
        s = f"{0:.{dp}f}"
    return "\\ensuremath{" + s + "}"


def fmt_int(n) -> str:
    """Integer with a thin-space thousands separator, as the paper writes 1{,}000."""
    return f"{int(n):,}".replace(",", "{,}")


def fmt_p(p) -> str:
    """Two significant digits; scientific below 0.001."""
    if not _finite(p):
        return "n/a"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"
    return "\\ensuremath{" + f"{p:.2g}" + "}"


def fmt_ci(lo, hi, dp: int, sign: bool = True) -> str:
    if not (_finite(lo) and _finite(hi)):
        return "n/a"
    a = f"{lo:+.{dp}f}" if sign else f"{lo:.{dp}f}"
    b = f"{hi:+.{dp}f}" if sign else f"{hi:.{dp}f}"
    return "\\ensuremath{[" + a + ",\\," + b + "]}"


def scale_of(kind: str) -> tuple[float, int]:
    """(multiplier, decimals) used to print a metric of this kind."""
    return {"score02": (1.0, 2), "rate": (100.0, 1), "count": (1.0, 2),
            "hb": (1.0, 3), "chars": (1.0, 0)}[kind]


def round6(obj):
    """Round floats for numbers.json so the file is stable across platforms."""
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: round6(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round6(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round6(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


class Macros:
    """Collects \\newcommand definitions in emission order."""

    def __init__(self):
        self.items: list[tuple[str, str, str]] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str, source: str) -> None:
        name = "stat" + name
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        self.items.append((name, value, source))

    def render(self) -> str:
        lines = [
            "% Generated by scripts/paper_stats.py. Do not edit by hand; run `make numbers`.",
            "% Each macro carries a comment naming the data it comes from.",
            "",
        ]
        for name, value, source in self.items:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}} % {source}")
        return "\n".join(lines) + "\n"


def tabular(colspec: str, header: list[str], blocks: list[list[list[str]]],
            source: str) -> str:
    """A booktabs tabular. ``blocks`` are row groups separated by \\midrule."""
    lines = [f"% Generated by scripts/paper_stats.py from {source}. Do not edit by hand.",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\",
             "\\midrule"]
    for i, block in enumerate(blocks):
        if i:
            lines.append("\\midrule")
        for row in block:
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

SRC_EP = "results_modal/seed_*/epistemic_scores.json"
SRC_HB = "results_modal/seed_*/<condition>.json"


def describe_conditions(df, seeds, out, macros):
    """Per-condition means: seed-level mean of seed means, cross-seed SD, pooled mean."""
    res = {}
    for m in METRICS:
        frame = metric_frame(df, m["key"])
        sm = seed_means(frame, seeds)
        mult, dp = scale_of(m["kind"])
        src = SRC_EP if m["source"] == "epistemic" else SRC_HB
        res[m["key"]] = {}
        for cfg in CONDITIONS:
            vals = sm[cfg].to_numpy(dtype=float)
            pooled = frame.loc[frame["config"] == cfg, "v"]
            mean, lo, hi = t_interval(vals)
            res[m["key"]][cfg] = {
                "seed_mean": mean,
                "seed_sd": float(np.std(vals, ddof=1)),
                "seed_ci95_t": [lo, hi],
                "per_seed": {str(s): float(v) for s, v in zip(seeds, vals)},
                "pooled_mean": float(pooled.mean()),
                "n_pooled": int(pooled.count()),
            }
            r = res[m["key"]][cfg]
            tok = m["token"] + COND_TOKEN[cfg]
            unit = " (percent)" if m["kind"] == "rate" else ""
            macros.add(tok, fmt(r["seed_mean"] * mult, dp),
                       f"{src}: mean of {len(seeds)} seed means{unit}")
            macros.add(tok + "SD", fmt(r["seed_sd"] * mult, dp),
                       f"{src}: SD of {len(seeds)} seed means{unit}")
            macros.add(tok + "Pooled", fmt(r["pooled_mean"] * mult, dp),
                       f"{src}: mean over {r['n_pooled']} pooled responses{unit}")
            if m["key"] == "healthbench":
                macros.add(tok + "CI", fmt_ci(lo, hi, 3, sign=False),
                           f"{src}: 95% t interval over {len(seeds)} seed means")
    out["conditions"] = res
    return res


def compare_conditions(df, seeds, boot, out, macros):
    """Seed-level paired differences with Holm, and pooled d with a clustered bootstrap CI."""
    res = {}
    for ctok, a, b in CONTRASTS:
        res[ctok] = {"a": a, "b": b, "metrics": {}}
        for m in METRICS:
            frame = metric_frame(df, m["key"])
            sm = seed_means(frame, seeds)
            r = paired_seed_test(sm[a].to_numpy(dtype=float), sm[b].to_numpy(dtype=float))
            fa = frame[frame["config"] == a]
            fb = frame[frame["config"] == b]
            r["pooled_d"] = cohens_d(fa["v"].to_numpy(dtype=float), fb["v"].to_numpy(dtype=float))
            r["pooled_diff"] = float(fa["v"].mean() - fb["v"].mean())
            # Descriptive only: the paper's original test, which treats the
            # pooled responses as independent. Kept so old text can be checked.
            r["pooled_welch_p"] = float(stats.ttest_ind(fa["v"], fb["v"], equal_var=False).pvalue)
            r.update(boot.diff_and_d(fa, fb))
            res[ctok]["metrics"][m["key"]] = r
        family = [res[ctok]["metrics"][k]["p"] for k in SEVEN_DIMS]
        for k, ph in zip(SEVEN_DIMS, holm(family)):
            res[ctok]["metrics"][k]["p_holm_within_contrast"] = ph
    # Holm over the 7 dimensions x 5 contrasts, for the paper's correction statement.
    keys = [(c, k) for c, _, _ in CONTRASTS for k in SEVEN_DIMS]
    for (c, k), ph in zip(keys, holm([res[c]["metrics"][k]["p"] for c, k in keys])):
        res[c]["metrics"][k]["p_holm_all_35"] = ph

    for ctok, a, b in CONTRASTS:
        for m in METRICS:
            r = res[ctok]["metrics"][m["key"]]
            mult, dp = scale_of(m["kind"])
            dp_diff = 3 if m["kind"] in ("hb", "score02") else dp
            tok = m["token"] + ctok
            what = f"{COND_TOKEN[a]} minus {COND_TOKEN[b]}"
            unit = " (percentage points)" if m["kind"] == "rate" else ""
            macros.add(tok + "Diff", fmt(r["diff"] * mult, dp_diff, sign=True),
                       f"seed-level mean difference, {what}{unit}")
            macros.add(tok + "CI", fmt_ci(r["ci95"][0] * mult, r["ci95"][1] * mult, dp_diff),
                       f"95% t({r['df']}) interval of the seed-level difference, {what}{unit}")
            macros.add(tok + "CILow", fmt(r["ci95"][0] * mult, dp_diff, sign=True),
                       f"lower 95% t bound, {what}{unit}")
            macros.add(tok + "CIHigh", fmt(r["ci95"][1] * mult, dp_diff, sign=True),
                       f"upper 95% t bound, {what}{unit}")
            macros.add(tok + "T", fmt(r["t"], 2), f"paired t({r['df']}) over seeds, {what}")
            macros.add(tok + "P", fmt_p(r["p"]), f"two-sided paired t p over seeds, {what}")
            if m["key"] in SEVEN_DIMS:
                macros.add(tok + "PHolm", fmt_p(r["p_holm_within_contrast"]),
                           f"Holm p over the 7 dimensions within {what}")
                macros.add(tok + "PHolmAll", fmt_p(r["p_holm_all_35"]),
                           "Holm p over 7 dimensions x 5 contrasts")
            macros.add(tok + "DiffBootCI", fmt_ci(r["diff_ci95"][0] * mult, r["diff_ci95"][1] * mult, dp_diff),
                       f"prompt-clustered bootstrap 95% CI of the pooled difference{unit}")
            macros.add(tok + "D", fmt(r["pooled_d"], 2, sign=True),
                       f"pooled Cohen's d, {what} (descriptive)")
            macros.add(tok + "DCI", fmt_ci(r["d_ci95"][0], r["d_ci95"][1], 2),
                       f"prompt-clustered bootstrap 95% CI of d, {BOOTSTRAP_RESAMPLES} resamples")
            if m["key"] == "healthbench":
                # A one-sided 95% bound is what a non-inferiority margin is
                # checked against; the margin itself is the author's call.
                macros.add(tok + "CINinety", fmt_ci(r["ci90"][0], r["ci90"][1], 3),
                           f"90% t interval (TOST form), {what}")
    out["contrasts"] = res
    return res


def counts_and_failures(df, seeds, out, macros):
    """Response counts, prompt overlap across seeds, and parse-failure rates."""
    res = {"per_condition": {}}
    for cfg in CONDITIONS:
        sub = df[df["config"] == cfg]
        st = sub["ep_status"].value_counts()
        n_hb = len(sub)
        n_ep_rows = int(st.get("scored", 0) + st.get("parse_failure", 0))
        r = {
            "completions": n_hb,
            "completions_per_seed": {str(s): int((sub["seed"] == s).sum()) for s in seeds},
            "epistemic_rows": n_ep_rows,
            "epistemic_scored": int(st.get("scored", 0)),
            "epistemic_parse_failures": int(st.get("parse_failure", 0)),
            "epistemic_missing": int(st.get("missing", 0)),
            "epistemic_parse_failure_rate": st.get("parse_failure", 0) / n_ep_rows,
            "epistemic_unscored_rate": 1 - st.get("scored", 0) / n_hb,
            "hb_items": int(sub["hb_items"].sum()),
            "hb_item_parse_failures": int(sub["hb_item_parse_failures"].sum()),
        }
        r["hb_item_parse_failure_rate"] = r["hb_item_parse_failures"] / r["hb_items"]
        per_run = sub.groupby("seed")[["hb_item_parse_failures", "hb_items"]].sum()
        r["hb_item_parse_failure_rate_max_run"] = float(
            (per_run["hb_item_parse_failures"] / per_run["hb_items"]).max())
        res["per_condition"][cfg] = r
        t = COND_TOKEN[cfg]
        macros.add(f"NCompletions{t}", fmt_int(r["completions"]),
                   f"{SRC_HB}: completions pooled over seeds")
        macros.add(f"NCompletionsPerSeed{t}", fmt(r["completions"] / len(seeds), 1),
                   "mean completions per seed (out of 200)")
        macros.add(f"NScored{t}", fmt_int(r["epistemic_scored"]),
                   f"{SRC_EP}: responses with a parsed epistemic grade, pooled")
        macros.add(f"NEpistemicParseFail{t}", fmt_int(r["epistemic_parse_failures"]),
                   f"{SRC_EP}: epistemic grades that failed to parse")
        macros.add(f"NEpistemicMissing{t}", fmt_int(r["epistemic_missing"]),
                   "completions with no epistemic row (grade error, dropped by eval_epistemic)")
        macros.add(f"EpistemicParseFailPct{t}", fmt(100 * r["epistemic_parse_failure_rate"], 2),
                   "epistemic parse failures / epistemic rows (percent)")
        macros.add(f"EpistemicUnscoredPct{t}", fmt(100 * r["epistemic_unscored_rate"], 2),
                   "(parse failures + missing) / completions (percent)")
        macros.add(f"HealthBenchParseFailPct{t}", fmt(100 * r["hb_item_parse_failure_rate"], 3),
                   f"{SRC_HB}: criterion grades that failed to parse / criteria (percent)")

    base_ids = df[df["config"] == "base_no_wrapper"]
    per_prompt = base_ids.groupby("prompt_id")["seed"].nunique()
    mult = Counter(per_prompt.tolist())
    res["unique_prompts"] = int(per_prompt.size)
    res["prompt_seed_multiplicity"] = {str(k): int(v) for k, v in sorted(mult.items())}
    res["prompts_in_more_than_one_seed"] = int((per_prompt > 1).sum())
    res["prompt_evaluations_per_condition"] = int(len(base_ids))
    macros.add("UniquePrompts", fmt_int(res["unique_prompts"]),
               "unique prompt ids over the 5 seed draws (Base)")
    macros.add("RepeatedPrompts", fmt_int(res["prompts_in_more_than_one_seed"]),
               "prompt ids drawn by more than one seed")
    macros.add("PromptEvaluations", fmt_int(res["prompt_evaluations_per_condition"]),
               "prompt draws per condition (5 seeds x 200)")
    macros.add("NSeeds", str(len(seeds)), "results_modal/seed_* directories")
    worst_ep = max(r["epistemic_parse_failure_rate"] for r in res["per_condition"].values())
    worst_hb = max(r["hb_item_parse_failure_rate_max_run"] for r in res["per_condition"].values())
    macros.add("EpistemicParseFailPctMax", fmt(100 * worst_ep, 2),
               "highest epistemic parse-failure rate of the 4 conditions (percent)")
    macros.add("HealthBenchParseFailPctMaxRun", fmt(100 * worst_hb, 3),
               "highest criterion parse-failure rate of any single run (percent)")
    out["counts"] = res
    return res


def worst_case(df, seeds, out, macros):
    """Sensitivity: every failed or missing grade takes the metric's worst value."""
    res = {}
    for m in METRICS:
        if m["key"] == "chars":
            continue
        frame = metric_frame(df, m["key"], worst_case=True)
        sm = seed_means(frame, seeds)
        mult, dp = scale_of(m["kind"])
        res[m["key"]] = {"conditions": {cfg: float(sm[cfg].mean()) for cfg in CONDITIONS},
                         "contrasts": {}}
        for ctok, a, b in CONTRASTS:
            r = paired_seed_test(sm[a].to_numpy(dtype=float), sm[b].to_numpy(dtype=float))
            res[m["key"]]["contrasts"][ctok] = r
        for cfg in CONDITIONS:
            macros.add(f"{m['token']}{COND_TOKEN[cfg]}WorstCase",
                       fmt(res[m["key"]]["conditions"][cfg] * mult, dp),
                       "worst case: failed or missing grades take the worst value")
        r = res[m["key"]]["contrasts"]["LoRAvsBase"]
        dp_diff = 3 if m["kind"] in ("hb", "score02") else dp
        macros.add(f"{m['token']}LoRAvsBaseWorstCaseDiff", fmt(r["diff"] * mult, dp_diff, sign=True),
                   "worst case: seed-level LoRA minus Base")
        macros.add(f"{m['token']}LoRAvsBaseWorstCaseP", fmt_p(r["p"]),
                   "worst case: paired t p, LoRA minus Base")
    out["worst_case"] = res
    return res


def blanket_disclaimers(df, out, macros):
    """Blanket-disclaimer counts and two-sided Fisher exact p against Base."""
    res = {}
    scored = df[df["ep_status"] == "scored"]
    counts = {cfg: (int(scored.loc[scored["config"] == cfg, "blanket_disclaimer"].sum()),
                    int((scored["config"] == cfg).sum())) for cfg in CONDITIONS}
    kb, nb = counts["base_no_wrapper"]
    for cfg in CONDITIONS:
        k, n = counts[cfg]
        res[cfg] = {"count": k, "n": n}
        macros.add(f"BlanketDisclaimerCount{COND_TOKEN[cfg]}", fmt_int(k),
                   f"{SRC_EP}: responses graded as a blanket disclaimer")
        if cfg == "base_no_wrapper":
            continue
        p = float(stats.fisher_exact([[k, n - k], [kb, nb - kb]]).pvalue)
        res[cfg]["fisher_p_vs_base"] = p
        macros.add(f"BlanketDisclaimerFisherP{COND_TOKEN[cfg]}", fmt_p(p),
                   f"two-sided Fisher exact p, {k}/{n} vs Base {kb}/{nb}")
    out["blanket_disclaimers"] = res
    return res


def length_quartiles(df, out, macros):
    """Active inquiry by response-length quartile, boundaries from Base."""
    scored = df[df["ep_status"] == "scored"]
    base_len = scored.loc[scored["config"] == "base_no_wrapper", "chars"].to_numpy()
    # numpy's default (linear) quantile; the paper's 2,106 / 3,574 / 4,316
    # came from list indexing and differ by a few characters.
    q = np.quantile(base_len, [0.25, 0.5, 0.75])
    res = {"boundaries_chars": q.tolist(), "conditions": {}}
    for cfg in CONDITIONS:
        sub = scored[scored["config"] == cfg]
        # Right-open bins: Q1 is len < q1, Q4 is len >= q3.
        bins = np.digitize(sub["chars"].to_numpy(), q, right=False)
        rows = []
        for i in range(4):
            sel = sub["active_inquiry"].to_numpy()[bins == i]
            rows.append({"n": int(len(sel)), "rate": float(sel.mean()) if len(sel) else float("nan")})
        res["conditions"][cfg] = {"quartiles": rows, "overall": float(sub["active_inquiry"].mean()),
                                  "n": int(len(sub))}
    for i, v in enumerate(q, 1):
        macros.add(f"LengthQuartileBound{'ABC'[i - 1]}", fmt_int(round(v)),
                   "Base response length quartile boundary (chars, linear quantile)")
    out["length_quartiles"] = res
    return res


def response_size(df, out, macros):
    """Mean response length (chars) and tokens per condition, all completions."""
    res = {}
    for cfg in CONDITIONS:
        sub = df[df["config"] == cfg]
        tok = sub["tokens"].dropna()
        res[cfg] = {"chars_mean": float(sub["chars"].mean()),
                    "chars_mean_scored": float(sub.loc[sub["ep_status"] == "scored", "chars"].mean()),
                    "tokens_mean": float(tok.mean()) if len(tok) else None,
                    "n_with_tokens": int(len(tok))}
        if len(tok):
            macros.add(f"ResponseTokens{COND_TOKEN[cfg]}", fmt(res[cfg]["tokens_mean"], 0),
                       f"{SRC_HB}: mean response_token_count over completions")
    b, l = res["base_no_wrapper"]["chars_mean"], res["lora_no_wrapper"]["chars_mean"]
    res["lora_minus_base_chars"] = l - b
    res["lora_vs_base_chars_pct"] = 100 * (l - b) / b
    macros.add("ResponseCharsAllBase", fmt(b, 0), f"{SRC_HB}: mean chars over all completions")
    macros.add("ResponseCharsAllLoRA", fmt(l, 0), f"{SRC_HB}: mean chars over all completions")
    macros.add("ResponseCharsLoRAvsBaseAbs", fmt(l - b, 0, sign=True), "LoRA minus Base mean chars, all completions")
    macros.add("ResponseCharsLoRAvsBasePct", fmt(res["lora_vs_base_chars_pct"], 1, sign=True),
               "LoRA vs Base mean chars, percent change")
    out["response_size"] = res
    return res


def dimension_correlations(df, out, macros):
    """Per-response correlation of the LoRA - Base shift in active inquiry with other shifts.

    Pairs are (seed, prompt) where both Base and LoRA have a parsed grade.
    """
    dims = ("active_inquiry", "context_seeking", "red_flag", "scope_bounding", "specificity")
    scored = df[df["ep_status"] == "scored"]
    a = scored[scored["config"] == "lora_no_wrapper"].set_index(["seed", "prompt_id"])[list(dims)]
    b = scored[scored["config"] == "base_no_wrapper"].set_index(["seed", "prompt_id"])[list(dims)]
    delta = (a - b).dropna()
    res = {"n_pairs": int(len(delta)), "pearson_r_with_active_inquiry": {}}
    macros.add("ShiftCorrN", fmt_int(len(delta)), "(seed, prompt) pairs with Base and LoRA graded")
    for d in dims[1:]:
        # numpy 2.2 on macOS Accelerate raises spurious overflow warnings inside
        # the BLAS dot product here; the inputs are 0/1/2 differences and r was
        # checked against a pure-Python Pearson (0.2297 for context-seeking).
        with np.errstate(all="ignore"):
            r = float(stats.pearsonr(delta["active_inquiry"], delta[d]).statistic)
        if not math.isfinite(r):
            raise SystemExit(f"paper_stats: Pearson r for {d} is not finite")
        res["pearson_r_with_active_inquiry"][d] = r
        macros.add(f"ShiftCorr{METRIC[d]['token']}", fmt(r, 2, sign=True),
                   f"Pearson r of LoRA-Base shift in active inquiry and in {d}")
    out["shift_correlations"] = res
    return res


def axis_scores(df, out, macros):
    """Mean HealthBench rubric-axis tag score over pooled completions (prompts that carry the axis)."""
    res = {}
    for axis in AXES:
        res[axis] = {cfg: float(df.loc[df["config"] == cfg, f"axis_{axis}"].mean()) for cfg in CONDITIONS}
        tok = "".join(w.capitalize() for w in axis.split("_"))
        for cfg in CONDITIONS:
            macros.add(f"Axis{tok}{COND_TOKEN[cfg]}", fmt(res[axis][cfg], 3),
                       f"{SRC_HB}: mean axis:{axis} tag score over pooled completions")
    out["axis_scores"] = res
    return res


def theme_rates(df, themes, out, macros):
    """Active inquiry by HealthBench theme, the five themes with most evaluated prompts."""
    scored = df[df["ep_status"] == "scored"].copy()
    scored["theme"] = scored["prompt_id"].map(themes)
    if scored["theme"].isna().any():
        raise SystemExit("paper_stats: some evaluated prompt ids have no theme in the HealthBench file")
    uniq = df[df["config"] == "base_no_wrapper"].drop_duplicates("prompt_id").copy()
    uniq["theme"] = uniq["prompt_id"].map(themes)
    n_prompts = uniq["theme"].value_counts()
    hard_counts = Counter(themes.values())
    # Order by evaluated prompts, ties broken by name so the order is stable.
    top = sorted(n_prompts.index, key=lambda t: (-n_prompts[t], t))[:5]
    res = {}
    for t in top:
        res[t] = {"n_unique_prompts_evaluated": int(n_prompts[t]),
                  "n_prompts_in_hard_set": int(hard_counts[t]), "conditions": {}}
        for cfg in CONDITIONS:
            sub = scored[(scored["config"] == cfg) & (scored["theme"] == t)]
            res[t]["conditions"][cfg] = {"rate": float(sub["active_inquiry"].mean()),
                                         "n_responses": int(len(sub))}
    out["themes"] = res
    return res


def tau_calibration(path: Path, out):
    """Share of the seed-7 training split at or above each tau."""
    scores = []
    with open(path) as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line)["grade"]["normalized_score"])
    s = np.asarray(scores, dtype=float)
    res = {"n_train_split": int(len(s)), "rows": []}
    for tau in TAU_GRID:
        share = float((s >= tau - 1e-12).mean())
        res["rows"].append({"tau": tau, "share_of_train_split": share,
                            "share_of_pool_estimate": share * FILTER_KEPT / FILTER_POOL})
    out["tau_calibration"] = res
    return res


# ---------------------------------------------------------------------------
# Optional inputs written by other scripts
# ---------------------------------------------------------------------------

def kappa_section(path: Path, out, macros):
    with open(path) as f:
        k = json.load(f)
    out["kappa"] = k
    ev = k["llm_vs_physician"]["eval_llama"]
    subsets = [("All", ev["all"]), ("Positive", ev["positive"]), ("Negative", ev["negative"])]
    subsets += [(COND_TOKEN[c], ev["by_config"][c]) for c in CONDITIONS]
    for name, r in subsets:
        macros.add(f"Kappa{name}", fmt(r["kappa"], 2),
                   "results_modal/irr/kappa_results.json: Cohen's kappa, physician vs Llama grader")
        macros.add(f"Kappa{name}N", fmt_int(r["n"]), "criterion rows with a pass/fail grade from both")
        macros.add(f"Kappa{name}CI", fmt_ci(r["ci95"][0], r["ci95"][1], 2),
                   "prompt-clustered bootstrap 95% CI for kappa")
    macros.add("KappaPrompts", fmt_int(ev["all"]["n_prompts"]), "distinct prompts behind the kappa rows")
    macros.add("KappaRaters", fmt_int(k["n_raters"]), "physicians whose grades were returned")
    macros.add("KappaRowsTotal", fmt_int(k["rows"]["total"]), "criterion rows in the grading sheet")
    macros.add("KappaRowsEval", fmt_int(k["rows"]["eval_llama"]), "criterion rows from Llama-graded eval responses")
    macros.add("KappaUnsureEval", fmt_int(k["unsure_consensus"]["eval_llama"]),
               "eval rows graded unsure, excluded from kappa")


def rebuttal_section(rdir: Path, out, macros):
    """Numbers from the committed rebuttal analysis outputs, where present."""
    res = {}
    im = rdir / "interference_mechanism.json"
    if im.is_file():
        d = json.load(open(im))
        res["interference_mechanism"] = d
        for cfg in CONDITIONS:
            macros.add(f"LeakagePct{COND_TOKEN[cfg]}", fmt(100 * d["configs"][cfg]["leaked_rate"], 1),
                       "rebuttal/results/interference_mechanism.json: share of scored responses with >=2 Pass-1 headers")
        dec = d["decisive"]
        macros.add("LeakageNLoRACoT", fmt_int(dec["n_leaked"]), "leaked LoRA+CoT responses")
        macros.add("LeakageNCleanLoRACoT", fmt_int(dec["n_clean"]), "non-leaked LoRA+CoT responses")
        macros.add("RedFlagLeaked", fmt(dec["lora_bodhi_leaked_red_flag"], 2), "red-flag (0-2), leaked LoRA+CoT")
        macros.add("RedFlagClean", fmt(dec["lora_bodhi_clean_red_flag"], 2), "red-flag (0-2), non-leaked LoRA+CoT")
    ms = rdir / "mimicry_split.json"
    if ms.is_file():
        d = json.load(open(ms))
        res["mimicry_split"] = d
        for label, tok in (("primary_theme_or_rubric", "Primary"), ("robustness_theme_only", "ThemeOnly")):
            sec = d[label]
            macros.add(f"Discrimination{tok}LoRAvsBase", fmt(100 * sec["lora_minus_base_disc"], 1, sign=True),
                       f"rebuttal/results/mimicry_split.json [{label}]: LoRA minus Base discrimination (pp)")
            lo, hi = sec["lora_minus_base_disc_ci95"]
            macros.add(f"Discrimination{tok}LoRAvsBaseCI", fmt_ci(100 * lo, 100 * hi, 1),
                       "prompt-clustered bootstrap 95% CI (pp)")
            for cfg in CONDITIONS:
                c = sec["configs"][cfg]
                if "odds_ratio" in c:
                    macros.add(f"AskOddsRatio{tok}{COND_TOKEN[cfg]}", fmt(c["odds_ratio"], 2),
                               f"[{label}] odds ratio of asking, context-dependent vs self-contained")
    sp = rdir / "second_person_questions.json"
    if sp.is_file():
        d = json.load(open(sp))
        res["second_person_questions"] = d
        for cfg in CONDITIONS:
            macros.add(f"SecondPersonPct{COND_TOKEN[cfg]}", fmt(100 * d["conditions"][cfg]["rate"], 1),
                       "rebuttal/results/second_person_questions.json: completions with a second-person question")
    out["rebuttal_analyses"] = res


def load_posted_values(path: Path) -> dict:
    """Read data/rebuttal_cells/posted_values.json and check its shape."""
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d.get("cells"), list):
        raise SystemExit(f"{path}: expected a 'cells' list")
    for c in d["cells"]:
        for key in ("id", "benchmark", "base_model", "base_family", "same_family_as_evaluator",
                    "n_seeds", "set_aside", "conditions"):
            if key not in c:
                raise SystemExit(f"{path}: cell {c.get('id')!r} lacks {key!r}")
        for cond in ("base", "lora"):
            if cond not in c["conditions"]:
                raise SystemExit(f"{path}: cell {c['id']!r} lacks condition {cond!r}")
    return d


def posted_aggregate_diff(cell: dict) -> dict:
    """LoRA minus Base aggregate score from posted means.

    The difference of the two seed-mean means equals the mean of the paired
    per-seed differences, so the point estimate is valid whenever both means
    were posted. A CI or test is not: it needs the SD of the per-seed
    differences, and the posted SDs are per condition. The posted paired t
    (rebuttal/analyses/FINDINGS_quality_drop.md) stays the test.
    """
    base = cell["conditions"]["base"].get("aggregate")
    lora = cell["conditions"]["lora"].get("aggregate")
    if base is None or lora is None or base.get("mean") is None or lora.get("mean") is None:
        return {"diff": None, "note": "aggregate mean not posted for both conditions"}
    return {"diff": lora["mean"] - base["mean"],
            "note": "point estimate only; inference is the posted paired t over seeds"}


def generality_section(posted: dict, conditions_res: dict, out):
    res = {"provenance": posted.get("_provenance"), "cells": []}
    for c in posted["cells"]:
        res["cells"].append({"id": c["id"], "set_aside": c["set_aside"],
                             "same_family_as_evaluator": c["same_family_as_evaluator"],
                             "aggregate_lora_minus_base": posted_aggregate_diff(c)})
    out["generality"] = res
    return res


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _cond_cell(conditions_res, key, cfg, with_sd=True):
    m = METRIC[key]
    mult, dp = scale_of(m["kind"])
    r = conditions_res[key][cfg]
    if with_sd:
        return f"${r['seed_mean'] * mult:.{dp}f}_{{\\pm{r['seed_sd'] * mult:.{dp}f}}}$"
    return f"{r['seed_mean'] * mult:.{dp}f}"


def build_tables(out, seeds) -> dict[str, str]:
    t = {}
    cond = out["conditions"]
    counts = out["counts"]["per_condition"]
    ns = len(seeds)

    # Table 1: aggregate HealthBench score.
    rows = []
    for cfg in CONDITIONS:
        r = cond["healthbench"][cfg]
        c = counts[cfg]
        per = list(c["completions_per_seed"].values())
        rows.append([COND_LABEL[cfg], f"{r['seed_mean']:.3f}",
                     f"[{r['seed_ci95_t'][0]:.3f},\\;{r['seed_ci95_t'][1]:.3f}]",
                     fmt_int(c["completions"]),
                     str(min(per)) if min(per) == max(per) else f"{min(per)}--{max(per)}"])
    t["tab_main_healthbench"] = tabular(
        "lcccc", ["Config", "Score (mean)", f"95\\% CI ($t_{{{ns - 1}}}$)", "$n$ (pooled)", "$n$/seed"],
        [rows], "results_modal/seed_*/<condition>.json (CI: t over seed means)")

    # Per-seed HealthBench table.
    rows = [[COND_LABEL[cfg]] + [f"{cond['healthbench'][cfg]['per_seed'][str(s)]:.3f}" for s in seeds]
            for cfg in CONDITIONS]
    t["tab_healthbench_per_seed"] = tabular(
        "l" + "c" * ns, ["Config"] + [f"Seed {s}" for s in seeds], [rows],
        "results_modal/seed_*/<condition>.json")

    # Table 2: epistemic scores, seed mean with cross-seed SD.
    top = ["uncertainty_ack", "context_seeking", "red_flag", "scope_bounding", "hedging_quality", "specificity"]
    bottom = ["questions", "active_inquiry", "red_flag_rate", "scope_bounded_rate", "specificity_rate",
              "blanket_disclaimer", "appropriate_hedging"]
    arrow = {"blanket_disclaimer": "$\\downarrow$"}
    blocks = []
    for keys in (top, bottom):
        blocks.append([[METRIC[k]["label"] + " " + arrow.get(k, "$\\uparrow$")]
                       + [_cond_cell(cond, k, cfg) for cfg in CONDITIONS] for k in keys])
    blocks.append([["Scored responses (pooled)"] + [fmt_int(counts[cfg]["epistemic_scored"]) for cfg in CONDITIONS]])
    t["tab_epistemic"] = tabular("lcccc", ["Dimension"] + [COND_LABEL[c] for c in CONDITIONS], blocks,
                                 SRC_EP + " (mean of seed means, SD across seeds)")

    # Seed-level contrasts for the seven dimensions.
    blocks = []
    for ctok, a, b in CONTRASTS:
        block = [[f"\\multicolumn{{7}}{{l}}{{\\emph{{{COND_LABEL[a]} $-$ {COND_LABEL[b]}}}}}"]]
        for k in SEVEN_DIMS + ("healthbench",):
            m = METRIC[k]
            r = out["contrasts"][ctok]["metrics"][k]
            mult, dp = scale_of(m["kind"])
            dpd = 3 if m["kind"] in ("hb", "score02") else dp
            block.append([
                m["label"], fmt(r["diff"] * mult, dpd, sign=True),
                fmt_ci(r["ci95"][0] * mult, r["ci95"][1] * mult, dpd),
                fmt(r["t"], 2), fmt_p(r["p"]),
                fmt_p(r.get("p_holm_within_contrast")) if k in SEVEN_DIMS else "--",
                fmt(r["pooled_d"], 2, sign=True) + " " + fmt_ci(r["d_ci95"][0], r["d_ci95"][1], 2),
            ])
        blocks.append(block)
    t["tab_seed_contrasts"] = tabular(
        "lcccccc", ["Dimension", "Diff.", f"95\\% CI ($t_{{{ns - 1}}}$)", f"$t_{{{ns - 1}}}$", "$p$",
                    "$p_{\\text{Holm}}$", "Pooled $d$ [bootstrap 95\\% CI]"],
        blocks, "results_modal (seed-level paired t; Holm over the 7 dimensions within each contrast; "
                f"d CI from {BOOTSTRAP_RESAMPLES} prompt-clustered bootstrap resamples)")

    # Headline comparisons: seed-level t CI next to the clustered bootstrap CI.
    heads = [("LoRAvsBase", "active_inquiry"), ("LoRAvsBaseCoT", "active_inquiry"),
             ("LoRACoTvsLoRA", "red_flag"), ("LoRACoTvsLoRA", "red_flag_rate"), ("LoRAvsBase", "healthbench")]
    rows = []
    for ctok, k in heads:
        _, a, b = next(c for c in CONTRASTS if c[0] == ctok)
        r = out["contrasts"][ctok]["metrics"][k]
        m = METRIC[k]
        mult, dp = scale_of(m["kind"])
        dpd = 3 if m["kind"] in ("hb", "score02") else dp
        rows.append([f"{COND_LABEL[a]} $-$ {COND_LABEL[b]}, {m['label']}",
                     fmt(r["diff"] * mult, dpd, sign=True),
                     fmt_ci(r["ci95"][0] * mult, r["ci95"][1] * mult, dpd),
                     fmt_ci(r["diff_ci95"][0] * mult, r["diff_ci95"][1] * mult, dpd)])
    t["tab_bootstrap"] = tabular(
        "lrrr", ["Comparison", "Effect", f"Seed-level 95\\% CI ($t_{{{ns - 1}}}$)",
                 "Prompt-clustered bootstrap 95\\% CI"], [rows],
        "results_modal (effect = mean of seed-level differences; bootstrap CI is for the pooled difference)")

    # Length quartiles.
    lq = out["length_quartiles"]
    rate_rows, n_rows = [], []
    for cfg in CONDITIONS:
        c = lq["conditions"][cfg]
        rate_rows.append([COND_LABEL[cfg]] + [f"{100 * q['rate']:.1f}" for q in c["quartiles"]]
                         + [f"{100 * c['overall']:.1f}"])
        n_rows.append([f"$n$, {COND_LABEL[cfg]}"] + [fmt_int(q["n"]) for q in c["quartiles"]]
                      + [fmt_int(c["n"])])
    t["tab_length_quartiles"] = tabular("lrrrrr", ["Condition", "Q1", "Q2", "Q3", "Q4", "Overall"],
                                        [rate_rows, n_rows],
                                        SRC_EP + " (active-inquiry %, pooled scored responses; Base quartiles)")

    # Axis scores.
    rows = [[a.replace("_", "\\_")] + [f"{out['axis_scores'][a][cfg]:.3f}" for cfg in CONDITIONS] for a in AXES]
    t["tab_axis"] = tabular("lrrrr", ["Axis"] + [COND_LABEL[c] for c in CONDITIONS], [rows],
                            "results_modal/seed_*/<condition>.json tag_scores (mean over pooled completions)")

    # Internalization gap.
    gap_keys = ["active_inquiry", "red_flag", "context_seeking", "scope_bounding", "specificity",
                "uncertainty_ack", "appropriate_hedging"]
    rows = []
    for k in gap_keys:
        mean = {cfg: cond[k][cfg]["seed_mean"] for cfg in CONDITIONS}
        rows.append([METRIC[k]["label"].replace("(\\%)", "(rate)")]
                    + [f"{mean[cfg]:.3f}" for cfg in CONDITIONS]
                    + [f"{abs(mean['lora_no_wrapper'] - mean['base_bodhi']):.3f}",
                       fmt(mean["lora_no_wrapper"] - mean["base_no_wrapper"], 3, sign=True)])
    t["tab_gap"] = tabular("lrrrrrr", ["Dimension"] + [COND_LABEL[c] for c in CONDITIONS]
                           + ["$|$LoRA$-$Base+\\bodhi{}$|$", "LoRA$-$Base"], [rows],
                           SRC_EP + " (mean of seed means; rates as proportions)")

    # Counts and parse failures.
    rows = []
    for cfg in CONDITIONS:
        c = counts[cfg]
        rows.append([COND_LABEL[cfg], fmt_int(c["completions"]), fmt_int(c["epistemic_scored"]),
                     f"{fmt_int(c['epistemic_parse_failures'])} ({100 * c['epistemic_parse_failure_rate']:.2f}\\%)",
                     fmt_int(c["epistemic_missing"]),
                     f"{c['hb_item_parse_failures']}/{fmt_int(c['hb_items'])}"])
    t["tab_counts"] = tabular(
        "lrrrrr", ["Condition", "Completions", "Graded", "Epistemic parse fail.", "Grade errors",
                   "HB criterion parse fail."], [rows],
        "results_modal (completions pooled over seeds; grade errors have no epistemic row)")

    # Blanket disclaimers.
    bd = out["blanket_disclaimers"]
    rows = []
    for cfg in CONDITIONS:
        r = bd[cfg]
        rows.append([COND_LABEL[cfg], f"{r['count']}/{fmt_int(r['n'])}",
                     f"{100 * r['count'] / r['n']:.1f}", fmt_p(r.get("fisher_p_vs_base")) if cfg != "base_no_wrapper" else "--"])
    t["tab_disclaimers"] = tabular("lrrr", ["Condition", "Blanket disclaimers", "\\%",
                                            "Fisher $p$ vs Base"], [rows], SRC_EP)

    # Worst-case sensitivity for the main contrast.
    rows = []
    for k in SEVEN_DIMS + ("blanket_disclaimer", "healthbench"):
        m = METRIC[k]
        mult, dp = scale_of(m["kind"])
        dpd = 3 if m["kind"] in ("hb", "score02") else dp
        rep = out["contrasts"]["LoRAvsBase"]["metrics"][k]
        wc = out["worst_case"][k]["contrasts"]["LoRAvsBase"]
        rows.append([m["label"], fmt(rep["diff"] * mult, dpd, sign=True), fmt_p(rep["p"]),
                     fmt(wc["diff"] * mult, dpd, sign=True), fmt_p(wc["p"])])
    t["tab_worst_case"] = tabular("lrrrr", ["Dimension", "LoRA$-$Base", "$p$", "Worst case", "$p$"], [rows],
                                  "results_modal (worst case: failed or missing grades take the worst value)")

    if "themes" in out:
        rows = []
        for th, r in out["themes"].items():
            rows.append([f"\\texttt{{{th.replace('_', chr(92) + '_')}}}", fmt_int(r["n_unique_prompts_evaluated"])]
                        + [f"{100 * r['conditions'][cfg]['rate']:.1f}" for cfg in CONDITIONS])
        t["tab_theme"] = tabular("lrrrrr", ["Theme", "$n$ prompts"] + [COND_LABEL[c] for c in CONDITIONS], [rows],
                                 "results_modal + data/raw/healthbench_hard.jsonl (n = unique evaluated prompts)")

    if "tau_calibration" in out:
        rows = [[f"{r['tau']:.1f}", f"{100 * r['share_of_train_split']:.1f}", f"{100 * r['share_of_pool_estimate']:.1f}"]
                for r in out["tau_calibration"]["rows"]]
        t["tab_tau_calibration"] = tabular(
            "ccc", ["$\\tau$", "\\% of seed-7 train split", "\\% of pool (est.)"], [rows],
            f"data/sft_qwen_graded/seed_7_train.jsonl (pool column assumes {FILTER_KEPT}/{FILTER_POOL} kept, PROV-11)")

    if "kappa" in out:
        ev = out["kappa"]["llm_vs_physician"]["eval_llama"]

        def krow(name, r):
            return [name, fmt(r["kappa"], 2), fmt_ci(r["ci95"][0], r["ci95"][1], 2),
                    f"{100 * r['agree']:.0f}\\%", fmt_int(r["n"]),
                    f"{100 * r['physician_pass']:.0f}\\%", f"{100 * r['llm_pass']:.0f}\\%"]

        blocks = [[krow("All criteria", ev["all"]), krow("Positive (meet)", ev["positive"]),
                   krow("Negative (avoid)", ev["negative"])],
                  [krow(COND_LABEL[c], ev["by_config"][c]) for c in CONDITIONS]]
        t["tab_kappa"] = tabular("lcccccc", ["Subset", "$\\kappa$", "95\\% CI", "Agree", "$n$",
                                             "Phys.\\ pass", "LLM pass"], blocks,
                                 "results_modal/irr/kappa_results.json (CI: prompt-clustered bootstrap)")

    if "generality" in out and "posted" in out:
        t["tab_generality"] = generality_table(out["posted"], out)
    return t


def generality_table(posted: dict, out) -> str:
    """Cross-cell table: main cell from results_modal, other cells from posted values."""
    cond = out["conditions"]

    def pct(x):
        return f"{x:.1f}\\%" if x is not None else "--"

    def arrow(b, l, dp=2):
        if b is None or l is None:
            return "--"
        return f"{b:.{dp}f} $\\to$ {l:.{dp}f}"

    main = [["HealthBench", "MedGemma-27B (Gemma)",
             pct(100 * cond["active_inquiry"]["base_no_wrapper"]["seed_mean"]),
             pct(100 * cond["active_inquiry"]["lora_no_wrapper"]["seed_mean"]),
             arrow(cond["context_seeking"]["base_no_wrapper"]["seed_mean"], cond["context_seeking"]["lora_no_wrapper"]["seed_mean"]),
             arrow(cond["red_flag"]["base_no_wrapper"]["seed_mean"], cond["red_flag"]["lora_no_wrapper"]["seed_mean"]),
             fmt(out["contrasts"]["LoRAvsBase"]["metrics"]["healthbench"]["diff"], 3, sign=True)]]
    kept, aside = [], []
    for c in posted["cells"]:
        g = c["conditions"]

        def val(cond_name, metric):
            v = g.get(cond_name, {}).get(metric)
            return None if v is None else v.get("mean")

        diff = posted_aggregate_diff(c)["diff"]
        model = f"{c['base_model']} ({c['base_family']})"
        if c["same_family_as_evaluator"]:
            # Same family as the Llama evaluator, against the paper's own rule (EVAL-7).
            model += "$^{\\dagger}$"
        row = [c["benchmark"], model, pct(val("base", "active_inquiry_pct")), pct(val("lora", "active_inquiry_pct")),
               arrow(val("base", "context_seeking"), val("lora", "context_seeking")),
               arrow(val("base", "red_flag"), val("lora", "red_flag")),
               fmt(diff, 3, sign=True) if diff is not None else "--"]
        (aside if c["set_aside"] else kept).append(row)
    blocks = [main, kept] + ([aside] if aside else [])
    return tabular("llrrrrr", ["Benchmark", "Base model (family)", "AI Base", "AI LoRA", "Context-seek",
                               "Red-flag", "Agg.\\ $\\Delta$"], blocks,
                   "results_modal (first row) and data/rebuttal_cells/posted_values.json (other rows; "
                   "Agg. Delta is LoRA minus Base from posted means, no CI)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative path, or the bare file name for inputs outside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def run(results_dir: Path, out_dir: Path, healthbench: Path | None, filter_train: Path | None,
        kappa: Path | None, rebuttal_results: Path | None, posted_values: Path | None) -> dict:
    seeds = discover_seeds(results_dir)
    df = load_responses(results_dir, seeds)
    out: dict = {"meta": {"seeds": seeds, "conditions": list(CONDITIONS),
                          "seven_dimensions": list(SEVEN_DIMS),
                          "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                                        "cluster": "prompt_id", "interval": "percentile"},
                          "inputs": {}}}
    for s in seeds:
        for name in [*(f"{c}.json" for c in CONDITIONS), "epistemic_scores.json"]:
            p = results_dir / f"seed_{s}" / name
            out["meta"]["inputs"][f"results_modal/seed_{s}/{name}"] = _sha256(p)
    macros = Macros()
    boot = ClusterBootstrap(df["prompt_id"].unique().tolist(), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)

    cond = describe_conditions(df, seeds, out, macros)
    compare_conditions(df, seeds, boot, out, macros)
    counts_and_failures(df, seeds, out, macros)
    worst_case(df, seeds, out, macros)
    blanket_disclaimers(df, out, macros)
    length_quartiles(df, out, macros)
    response_size(df, out, macros)
    dimension_correlations(df, out, macros)
    axis_scores(df, out, macros)

    def optional(path, what, logical):
        if path is not None and path.is_file():
            out["meta"]["inputs"][logical] = _sha256(path)
            return True
        print(f"SKIP {what}: {path} not found")
        return False

    if optional(healthbench, "theme table", "data/raw/healthbench_hard.jsonl"):
        theme_rates(df, load_themes(healthbench), out, macros)
    if optional(filter_train, "tau table", "data/sft_qwen_graded/seed_7_train.jsonl"):
        tau_calibration(filter_train, out)
    if optional(kappa, "kappa table", "results_modal/irr/kappa_results.json"):
        kappa_section(kappa, out, macros)
    if rebuttal_results is not None and rebuttal_results.is_dir():
        for p in sorted(rebuttal_results.glob("*.json")):
            out["meta"]["inputs"][f"rebuttal/results/{p.name}"] = _sha256(p)
        rebuttal_section(rebuttal_results, out, macros)
    else:
        print(f"SKIP rebuttal analyses: {rebuttal_results} not found")
    if optional(posted_values, "generality table (rebuttal cells)", "data/rebuttal_cells/posted_values.json"):
        out["posted"] = load_posted_values(posted_values)
        generality_section(out["posted"], cond, out)

    tables = build_tables(out, seeds)
    out.pop("posted", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    # A table whose input went missing must not survive from an earlier run.
    for stale in sorted(out_dir.glob("tab_*.tex")):
        if stale.stem not in tables:
            stale.unlink()
            print(f"removed stale {stale.name}")
    (out_dir / "numbers.json").write_text(json.dumps(round6(out), indent=2) + "\n")
    (out_dir / "numbers.tex").write_text(macros.render())
    for name, text in sorted(tables.items()):
        (out_dir / f"{name}.tex").write_text(text)
    print(f"wrote {_rel(out_dir)}/numbers.json, numbers.tex ({len(macros.items)} macros), "
          f"{len(tables)} tables: {', '.join(sorted(tables))}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_modal")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "generated")
    ap.add_argument("--healthbench", type=Path, default=ROOT / "data" / "raw" / "healthbench_hard.jsonl")
    ap.add_argument("--filter-train", type=Path,
                    default=ROOT / "data" / "sft_qwen_graded" / "seed_7_train.jsonl")
    ap.add_argument("--kappa", type=Path, default=ROOT / "results_modal" / "irr" / "kappa_results.json")
    ap.add_argument("--rebuttal-results", type=Path, default=ROOT / "rebuttal" / "results")
    ap.add_argument("--posted-values", type=Path,
                    default=ROOT / "data" / "rebuttal_cells" / "posted_values.json")
    a = ap.parse_args()
    run(a.results_dir, a.out_dir, a.healthbench, a.filter_train, a.kappa, a.rebuttal_results,
        a.posted_values)


if __name__ == "__main__":
    main()
