"""Aggregate per-seed eval outputs into across-seed means, stds, and 95% CIs.

Inputs
------
For each seed there is one directory matching the ``--seed-dirs`` glob (e.g.
``eval/seed_42``). Each directory holds the per-config outputs of
``scripts/eval_healthbench.py``::

    eval/seed_42/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json
    eval/seed_7/{...}
    eval/seed_13/{...}

The HealthBench JSONL passed via ``--healthbench`` is consulted for tier
metadata (``pos_points`` per prompt) so the across-seed numbers can be
re-stratified by difficulty tier and by example tag (theme).

Outputs
-------
A single aggregate summary JSON written to ``--output``. For every (config,
metric) it records ``mean`` / ``std`` / ``min`` / ``max`` / ``values`` plus a
percentile-based 95 % CI when there are at least 5 seeds. Stratified U-shape
numbers (by_tier, by_theme) are re-aggregated per seed via ``eval_ushape.py``
helpers and then combined across seeds, so ``aggregate_seeds`` benefits
automatically when the U-shape math changes.

How ``--seed-dirs`` is resolved
-------------------------------
Each pattern is passed to ``glob.glob`` if it contains shell wildcards
(``*?[``); otherwise it is treated as a literal path. Resulting paths are
filtered to those that are actually directories, deduplicated, and sorted.
The script aborts if fewer than 2 seed directories are found, since a single
seed cannot meaningfully estimate run-to-run spread.

Honesty contract for downstream tools
-------------------------------------
``_aggregate_metric_across_seeds`` is the single source of truth for how the
across-seed numbers are reported. It guarantees:

* ``std`` is ``None`` when ``n_seeds < 2`` (no second sample to estimate
  spread from). A ``"note"`` field carries the reason
  (``"stdev undefined for n_seeds<2"``). Older versions returned 0.0, which
  read as "zero variance" and silently understated uncertainty.
* ``ci_low`` / ``ci_high`` are ALWAYS present: they are ``None`` when
  ``n_seeds < 5``. ``ci_method`` documents whether the CI was computed
  (``"percentile-95 (n>=5)"``) or skipped (``"skipped (n<5)"``), so a
  consumer can tell "we didn't compute it" from "we computed it and the
  width was zero."
* Non-finite values (``NaN`` / ``Inf``) in the per-seed inputs are filtered
  out before any statistic is computed. ``n_excluded_nan`` reports how many
  values were dropped, so a partial corruption is visible rather than
  silently averaging garbage.

Usage
-----
::

    python scripts/aggregate_seeds.py \\
        --seed-dirs eval/seed_* \\
        --healthbench data/raw/healthbench_hard.jsonl \\
        --output eval/multi_seed_summary.json
"""

import argparse
import glob
import json
import math
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


def _load_eval_json(path):
    """Load a per-seed eval JSON with file context on parse failure.

    The aggregate run touches dozens of JSONs; if any one of them is
    truncated (e.g. the eval job was killed mid-write), the cluster log
    needs the path to debug. A bare json.JSONDecodeError stack trace from
    deep inside ``json.load`` is not enough to tell which file broke.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"aggregate_seeds: malformed JSON in {path}: {e}")


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
            configs[cfg] = _load_eval_json(p)
        if configs:
            per_seed[label] = configs
    return per_seed


def _aggregate_metric_across_seeds(values, ci=0.95):
    """Given a list of numbers (one per seed), return mean/std/min/max/CI.

    Honesty contract for downstream tools (audit N1, N2, N3):
      * std is None when N<2 (no second sample to estimate spread from);
        previously we emitted 0.0 which read as "zero variance" instead of
        "undefined."
      * ci_low/ci_high are ALWAYS present; None when N<5. Callers can
        distinguish "skipped" from "computed and was 0-width."
      * note / ci_method strings document why each field is what it is.
      * NaN / Inf are filtered before any statistic is computed and the
        count is reported as ``n_excluded_nan``, so partial corruption is
        visible rather than silently averaging garbage.
    """
    raw = list(values)
    # Drop None first (missing per-seed value), then non-finite (NaN/Inf).
    # Tracking n_excluded_nan separately from None lets downstream readers
    # distinguish "this seed didn't report the metric" from "this seed
    # reported a corrupt number." Both should be visible.
    non_none = [v for v in raw if v is not None]
    finite = [v for v in non_none if math.isfinite(v)]
    n_excluded_nan = len(non_none) - len(finite)

    if not finite:
        out = {"n_seeds": 0}
        if n_excluded_nan:
            out["n_excluded_nan"] = n_excluded_nan
        return out

    n = len(finite)
    if n >= 2:
        std_val = float(statistics.stdev(finite))
        std_note = None
    else:
        std_val = None
        std_note = "stdev undefined for n_seeds<2"

    if n >= 5:
        lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
        ci_low = float(np.percentile(finite, lo))
        ci_high = float(np.percentile(finite, hi))
        ci_method = f"percentile-{int(ci * 100)} (n>=5)"
    else:
        ci_low = None
        ci_high = None
        ci_method = "skipped (n<5)"

    out = {
        "n_seeds": n,
        "mean": float(statistics.mean(finite)),
        "std": std_val,
        "min": float(min(finite)),
        "max": float(max(finite)),
        "values": [float(v) for v in finite],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": ci_method,
        "n_excluded_nan": n_excluded_nan,
    }
    if std_note is not None:
        out["note"] = std_note
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-seed HealthBench eval outputs into across-seed "
            "means, stds, and 95% CIs (when n>=5)."
        ),
    )
    parser.add_argument(
        "--seed-dirs",
        nargs="+",
        required=True,
        help=(
            "Glob pattern matching per-seed eval directories "
            "(e.g., 'eval/seed_*'). Shell globs are re-expanded inside the "
            "script for the case where the caller's shell didn't expand them."
        ),
    )
    parser.add_argument(
        "--healthbench",
        nargs="+",
        required=True,
        help=(
            "Path(s) to HealthBench JSONL files used for tier metadata "
            "(pos_points per prompt and example_tags). Multiple paths are "
            "merged."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the aggregate summary JSON.",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.4,
        help=(
            "Per-config failure-rate threshold (e.g. 0.4 = 40%%); per-prompt "
            "scores below this count as failures when computing fail-rate "
            "by tier and by theme."
        ),
    )
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
