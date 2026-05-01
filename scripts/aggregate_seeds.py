"""Aggregate per-seed eval outputs into across-seed means, stds, and 95% CIs.

Expects a directory layout like::

    eval/seed_42/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json
    eval/seed_7/{...}
    eval/seed_13/{...}

for N >= 2 seed directories. Each JSON is the output of
scripts/eval_healthbench.py. For each (config, metric), we compute mean
and std across seeds, plus a percentile-based 95% CI if N is large
enough (>= 5). Stratified U-shape numbers are re-aggregated per seed
via eval_ushape.py and then combined across seeds.

Usage:
    python scripts/aggregate_seeds.py \\
        --seed-dirs eval/seed_* \\
        --healthbench data/raw/healthbench_hard.jsonl \\
        --output eval/multi_seed_summary.json
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the per-seed aggregation logic from eval_ushape so the two scripts stay
# in sync. If eval_ushape ever changes its tier math, aggregate_seeds benefits
# automatically.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_ushape import (
    load_healthbench_meta,
    compute_tertile_cutoffs,
    aggregate_by_tier,
    aggregate_by_theme,
)


CONFIG_NAMES = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")


def _expand_seed_dirs(raw_patterns):
    """Expand shell globs (in case the shell didn't) and filter to directories
    that actually contain eval JSONs."""
    dirs = []
    for pat in raw_patterns:
        matches = glob.glob(pat) if any(c in pat for c in "*?[") else [pat]
        for m in matches:
            if Path(m).is_dir():
                dirs.append(m)
    dirs = sorted(set(dirs))
    return dirs


def _seed_label(path):
    """Recover the seed number from a directory name like eval/seed_42/."""
    base = Path(path).name
    if base.startswith("seed_"):
        try:
            return int(base.removeprefix("seed_"))
        except ValueError:
            pass
    return base


def _collect_per_seed_results(seed_dirs):
    """For each seed-directory, load the four config JSONs."""
    per_seed = {}
    for d in seed_dirs:
        label = _seed_label(d)
        configs = {}
        for cfg in CONFIG_NAMES:
            p = Path(d) / f"{cfg}.json"
            if not p.exists():
                continue
            with open(p) as f:
                configs[cfg] = json.load(f)
        if configs:
            per_seed[label] = configs
    return per_seed


def _aggregate_metric_across_seeds(values, ci=0.95):
    """Given a list of numbers (one per seed), return mean/std/min/max/CI.

    Honesty contract for downstream tools (audit N1, N2):
      * std is None when N<2 (no second sample to estimate spread from);
        previously we emitted 0.0 which read as "zero variance" instead of
        "undefined."
      * ci_low/ci_high are ALWAYS present; None when N<5. Callers can
        distinguish "skipped" from "computed and was 0-width."
      * note / ci_method strings document why each field is what it is.
    """
    values = [v for v in values if v is not None]
    if not values:
        return {"n_seeds": 0}

    n = len(values)
    if n >= 2:
        std_val = float(statistics.stdev(values))
        std_note = None
    else:
        std_val = None
        std_note = "stdev undefined for n_seeds<2"

    if n >= 5:
        lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
        ci_low = float(np.percentile(values, lo))
        ci_high = float(np.percentile(values, hi))
        ci_method = f"percentile-{int(ci * 100)} (n>=5)"
    else:
        ci_low = None
        ci_high = None
        ci_method = "skipped (n<5)"

    out = {
        "n_seeds": n,
        "mean": float(statistics.mean(values)),
        "std": std_val,
        "min": float(min(values)),
        "max": float(max(values)),
        "values": [float(v) for v in values],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": ci_method,
    }
    if std_note is not None:
        out["note"] = std_note
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dirs", nargs="+", required=True,
                        help="per-seed eval dirs (e.g. eval/seed_*); shell "
                             "globs are expanded if the shell doesn't.")
    parser.add_argument("--healthbench", nargs="+", required=True,
                        help="HealthBench JSONL files with metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-threshold", type=float, default=0.4)
    args = parser.parse_args()

    seed_dirs = _expand_seed_dirs(args.seed_dirs)
    if len(seed_dirs) < 2:
        raise SystemExit(
            f"need at least 2 seed directories, got {len(seed_dirs)}: {seed_dirs}"
        )
    print(f"Aggregating across {len(seed_dirs)} seeds: "
          f"{[_seed_label(d) for d in seed_dirs]}")

    meta = load_healthbench_meta(args.healthbench)
    q1, q2 = compute_tertile_cutoffs(meta)

    per_seed = _collect_per_seed_results(seed_dirs)

    # For each config, collect lists of metrics across seeds.
    # Structure: metrics_per_config[cfg_name][metric_key] = [value_per_seed, ...]
    metrics_per_config = {cfg: defaultdict(list) for cfg in CONFIG_NAMES}
    tier_per_config = {cfg: defaultdict(lambda: defaultdict(list))
                       for cfg in CONFIG_NAMES}
    theme_per_config = {cfg: defaultdict(lambda: defaultdict(list))
                        for cfg in CONFIG_NAMES}

    for seed, configs in per_seed.items():
        for cfg_name in CONFIG_NAMES:
            ev = configs.get(cfg_name)
            if ev is None:
                continue
            metrics_per_config[cfg_name]["overall_mean"].append(ev.get("mean"))
            results = ev.get("results", [])
            by_tier = aggregate_by_tier(results, meta, q1, q2, args.fail_threshold)
            for tier in ("easy", "medium", "hard"):
                t = by_tier.get(tier, {})
                tier_per_config[cfg_name][tier]["mean"].append(t.get("mean"))
                tier_per_config[cfg_name][tier]["fail_rate"].append(t.get("fail_rate"))
            by_theme = aggregate_by_theme(results, meta, args.fail_threshold)
            for theme, s in by_theme.items():
                theme_per_config[cfg_name][theme]["mean"].append(s.get("mean"))
                theme_per_config[cfg_name][theme]["fail_rate"].append(s.get("fail_rate"))

    summary = {
        "n_seeds": len(per_seed),
        "seeds": sorted(per_seed.keys()),
        "thresholds": {"q1": q1, "q2": q2, "fail_below": args.fail_threshold},
        "configs": {},
    }
    for cfg in CONFIG_NAMES:
        summary["configs"][cfg] = {
            "overall_mean": _aggregate_metric_across_seeds(
                metrics_per_config[cfg]["overall_mean"]
            ),
            "by_tier": {
                tier: {
                    "mean": _aggregate_metric_across_seeds(d["mean"]),
                    "fail_rate": _aggregate_metric_across_seeds(d["fail_rate"]),
                }
                for tier, d in tier_per_config[cfg].items()
            },
            "by_theme": {
                theme: {
                    "mean": _aggregate_metric_across_seeds(d["mean"]),
                    "fail_rate": _aggregate_metric_across_seeds(d["fail_rate"]),
                }
                for theme, d in theme_per_config[cfg].items()
            },
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Quick console readout for cluster logs. std may be None (n_seeds<2);
    # render as "n/a" rather than 0.000 so readers don't mistake "undefined"
    # for "zero variance."
    def _fmt_std(s, precision):
        v = s.get("std")
        return f"{v:.{precision}f}" if v is not None else "n/a"

    print("\n=== Across-seed headline (overall mean ± std) ===")
    for cfg in CONFIG_NAMES:
        s = summary["configs"][cfg]["overall_mean"]
        if s.get("n_seeds"):
            print(f"  {cfg:<20} {s['mean']:.3f} ± {_fmt_std(s, 3)}  "
                  f"(n={s['n_seeds']})")

    print("\n=== Across-seed by tier (fail rate ± std) ===")
    print(f"  {'config':<20} {'easy':>16} {'medium':>16} {'hard':>16}")
    for cfg in CONFIG_NAMES:
        cells = []
        for tier in ("easy", "medium", "hard"):
            t = summary["configs"][cfg]["by_tier"].get(tier, {}).get("fail_rate", {})
            if t.get("n_seeds"):
                cells.append(f"{t['mean']:.2f} ± {_fmt_std(t, 2)}")
            else:
                cells.append("-")
        print(f"  {cfg:<20} " + " ".join(f"{c:>16}" for c in cells))

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
