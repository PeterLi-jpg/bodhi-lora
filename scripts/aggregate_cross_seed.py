"""Aggregate Stage 4 + Stage 5 metrics across all 5 seeds.

Reads ``results_modal/seed_<N>/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi,epistemic_scores}.json``
for N in {7, 13, 42, 99, 101} and writes:

  results_modal/cross_seed_aggregate.json
    Per-config means and stds across the 5 seeds, both for HealthBench score
    (Stage 4) and the six epistemic dimensions (Stage 5). One JSON file the
    paper can quote directly.

  results_modal/cross_seed_aggregate.md
    Human-readable Markdown table for quick eyeballing / pasting into the
    results section.

Why ``mean over seeds, not pooled means`` — each seed evaluates a different
200-prompt subset (deterministic by seed, see make_bootstrap_eval_ids.py),
so a pooled-prompt mean would mix subsets and hide between-seed variance.
We average per-seed means and report the across-seed std as the uncertainty
bar.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_modal"
SEEDS = (7, 13, 42, 99, 101)
STAGE4_CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")
EPISTEMIC_AGG_KEYS = (
    "uncertainty_acknowledgment_mean",
    "context_seeking_mean",
    "red_flag_identification_mean",
    "scope_bounding_mean",
    "specificity_mean",
    "hedging_mean",
    "questions_asked_mean",
    "active_inquiry_rate",
    "red_flag_rate",
    "specificity_rate",
    "blanket_disclaimer_rate",
    "scope_bounded_rate",
    "appropriate_hedging_rate",
)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Return (mean, sample-std). Returns (mean, 0.0) if only one value."""
    vs = [v for v in values if v is not None]
    if not vs:
        return (float("nan"), float("nan"))
    if len(vs) == 1:
        return (float(vs[0]), 0.0)
    return (statistics.mean(vs), statistics.stdev(vs))


def aggregate_stage4(per_seed_data: dict[int, dict]) -> dict:
    """Average HealthBench score + parse-failure rate across seeds, per config."""
    out = {}
    for cfg in STAGE4_CONFIGS:
        means = [per_seed_data[s][cfg]["mean"] for s in SEEDS]
        ns    = [per_seed_data[s][cfg].get("n_examples", 0) for s in SEEDS]
        results_n = [len(per_seed_data[s][cfg].get("results", [])) for s in SEEDS]
        parse_fr  = [per_seed_data[s][cfg].get("grader_parse_failure_rate", 0.0) for s in SEEDS]
        m, sd = _mean_std(means)
        out[cfg] = {
            "score_mean_across_seeds": m,
            "score_std_across_seeds": sd,
            "score_per_seed": dict(zip([f"seed_{s}" for s in SEEDS], means)),
            "n_examples_per_seed": dict(zip([f"seed_{s}" for s in SEEDS], ns)),
            "n_results_per_seed": dict(zip([f"seed_{s}" for s in SEEDS], results_n)),
            "grader_parse_failure_rate_per_seed": dict(zip([f"seed_{s}" for s in SEEDS], parse_fr)),
        }
    return out


def aggregate_stage5(per_seed_eps: dict[int, dict]) -> dict:
    """Average epistemic dimensions across seeds, per Stage-4 config."""
    out = {}
    for cfg in STAGE4_CONFIGS:
        per_cfg = {}
        # Pull each seed's aggregates dict for this config.
        per_seed_aggs = {s: per_seed_eps[s][cfg]["aggregates"] for s in SEEDS}
        # Track scoring counts.
        n_total = [per_seed_aggs[s]["n"] for s in SEEDS]
        n_scored = [per_seed_aggs[s]["n_scored"] for s in SEEDS]
        n_pf = [per_seed_aggs[s]["n_parse_failures"] for s in SEEDS]
        per_cfg["n_per_seed"] = dict(zip([f"seed_{s}" for s in SEEDS], n_total))
        per_cfg["n_scored_per_seed"] = dict(zip([f"seed_{s}" for s in SEEDS], n_scored))
        per_cfg["n_parse_failures_per_seed"] = dict(zip([f"seed_{s}" for s in SEEDS], n_pf))
        for k in EPISTEMIC_AGG_KEYS:
            vals = [per_seed_aggs[s].get(k) for s in SEEDS]
            m, sd = _mean_std(vals)
            per_cfg[k] = {
                "mean": m,
                "std": sd,
                "per_seed": dict(zip([f"seed_{s}" for s in SEEDS], vals)),
            }
        out[cfg] = per_cfg
    return out


def load_per_seed_stage4(seed: int) -> dict[str, dict]:
    """Load the 4 Stage-4 JSONs for one seed."""
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    out = {}
    for cfg in STAGE4_CONFIGS:
        with open(seed_dir / f"{cfg}.json") as f:
            out[cfg] = json.load(f)
    return out


def load_per_seed_stage5(seed: int) -> dict[str, dict]:
    """Load epistemic_scores.json and index by config name."""
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    with open(seed_dir / "epistemic_scores.json") as f:
        d = json.load(f)
    # epistemic_scores.json has list of {name, aggregates, ...} configs.
    by_name = {c["name"]: c for c in d["configs"]}
    return by_name


def render_md(stage4_agg: dict, stage5_agg: dict) -> str:
    """Markdown report — paper-ready table for both stages."""
    lines = []
    lines.append("# Cross-seed aggregate (5 seeds: 7, 13, 42, 99, 101)\n")
    lines.append("Mean ± std across 5 random seeds. Each seed evaluates its own deterministic 200-prompt subset of HealthBench Hard.\n")

    # Stage 4 table.
    lines.append("## Stage 4 — HealthBench Hard score\n")
    lines.append("| Config | Mean | Std (across seeds) | Per-seed mean (7/13/42/99/101) |")
    lines.append("|---|---|---|---|")
    for cfg in STAGE4_CONFIGS:
        a = stage4_agg[cfg]
        per = a["score_per_seed"]
        per_str = " / ".join(f"{per[f'seed_{s}']:.4f}" for s in SEEDS)
        lines.append(f"| {cfg} | {a['score_mean_across_seeds']:.4f} | {a['score_std_across_seeds']:.4f} | {per_str} |")
    lines.append("")

    # Stage 5 epistemic — main rows.
    lines.append("## Stage 5 — Epistemic dimensions (mean ± std across seeds)\n")
    head_dims = (
        ("uncertainty_acknowledgment_mean", "Uncertainty"),
        ("context_seeking_mean",            "Context"),
        ("scope_bounding_mean",             "Scope"),
        ("specificity_mean",                "Specificity"),
        ("hedging_mean",                    "Hedging"),
        ("questions_asked_mean",            "Q/resp"),
        ("active_inquiry_rate",             "Active inq %"),
        ("red_flag_rate",                   "Red-flag %"),
        ("specificity_rate",                "Specific %"),
        ("scope_bounded_rate",              "Scope-bounded %"),
    )
    lines.append("| Config | " + " | ".join(label for _, label in head_dims) + " |")
    lines.append("|---|" + "|".join(["---"] * len(head_dims)) + "|")
    for cfg in STAGE4_CONFIGS:
        per_cfg = stage5_agg[cfg]
        cells = []
        for key, _ in head_dims:
            d = per_cfg[key]
            m, s = d["mean"], d["std"]
            if "rate" in key:
                cells.append(f"{m * 100:.1f} ± {s * 100:.1f}")
            else:
                cells.append(f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    # Sample sizes.
    lines.append("## Sample sizes\n")
    lines.append("| Config | seed_7 n_results | seed_13 | seed_42 | seed_99 | seed_101 |")
    lines.append("|---|---|---|---|---|---|")
    for cfg in STAGE4_CONFIGS:
        per = stage4_agg[cfg]["n_results_per_seed"]
        cells = " | ".join(str(per[f"seed_{s}"]) for s in SEEDS)
        lines.append(f"| {cfg} | {cells} |")
    lines.append("")
    lines.append("(200 = no inference failures. <200 = bodhi 2-pass requests that exceeded the 4096-token context cap; ~4-5%% per bodhi config.)")

    return "\n".join(lines)


def main() -> None:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"results dir not found: {RESULTS_DIR}")
    s4_data = {s: load_per_seed_stage4(s) for s in SEEDS}
    s5_data = {s: load_per_seed_stage5(s) for s in SEEDS}

    s4_agg = aggregate_stage4(s4_data)
    s5_agg = aggregate_stage5(s5_data)

    out = {
        "seeds": list(SEEDS),
        "stage4_healthbench": s4_agg,
        "stage5_epistemic": s5_agg,
    }
    json_path = RESULTS_DIR / "cross_seed_aggregate.json"
    md_path = RESULTS_DIR / "cross_seed_aggregate.md"
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(render_md(s4_agg, s5_agg))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    # Echo headline numbers.
    print()
    print("=== HEADLINE: Stage 4 HealthBench score (mean ± across-seed std) ===")
    for cfg in STAGE4_CONFIGS:
        a = s4_agg[cfg]
        print(f"  {cfg:<22} {a['score_mean_across_seeds']:.4f} ± {a['score_std_across_seeds']:.4f}")
    print()
    print("=== HEADLINE: Stage 5 active_inquiry_rate (the one that moves most) ===")
    for cfg in STAGE4_CONFIGS:
        d = s5_agg[cfg]["active_inquiry_rate"]
        print(f"  {cfg:<22} {d['mean'] * 100:5.1f}% ± {d['std'] * 100:.1f}%")


if __name__ == "__main__":
    main()
