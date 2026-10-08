"""Render the U-shape eval into paper-ready figures.

Two plotting modes:

1. From ushape.json only (fast, coarse):
     python scripts/plot_ushape.py --input eval/ushape.json --out-dir eval/figures
   Produces u_curve.png / u_fail.png / theme_fail.png with the 3 tertiles
   (easy/medium/hard) eval_ushape.py aggregates.

2. From raw eval JSONs + HealthBench (slower, gives a smooth U):
     python scripts/plot_ushape.py \\
         --eval-jsons eval/base_no_wrapper.json ... eval/lora_bodhi.json \\
         --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \\
         --out-dir eval/figures --n-bins 10
   Additionally produces u_curve_smooth.png (10-bin curve with quadratic
   overlay) and u_scatter.png (per-example scatter + smoother). These show
   the actual curve rather than 3 line-segments pretending to be one.

No model inference happens here. Purely post-hoc plotting.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — works on slurm/GCP
import matplotlib.pyplot as plt
import numpy as np


TIER_ORDER = ["easy", "medium", "hard"]

# Stable config order for consistent colors across figures
CONFIG_ORDER = [
    "base_no_wrapper",
    "base_bodhi",
    "lora_no_wrapper",
    "lora_bodhi",
]
CONFIG_COLORS = {
    "base_no_wrapper": "#c0392b",   # red — worst expected
    "base_bodhi":      "#e67e22",   # orange
    "lora_no_wrapper": "#2980b9",   # blue — headline config
    "lora_bodhi":      "#27ae60",   # green — best expected
}
CONFIG_LABELS = {
    "base_no_wrapper": "Base (no wrapper)",
    "base_bodhi":      "Base + BODHI wrapper",
    "lora_no_wrapper": "LoRA (no wrapper)",
    "lora_bodhi":      "LoRA + BODHI wrapper",
}


def ordered_configs_from_summary(summary):
    present = list(summary["configs"].keys())
    ordered = [c for c in CONFIG_ORDER if c in present]
    extras = [c for c in present if c not in CONFIG_ORDER]
    return ordered + extras


def ordered_configs(names):
    ordered = [c for c in CONFIG_ORDER if c in names]
    extras = [c for c in names if c not in CONFIG_ORDER]
    return ordered + extras


# -------- Tier plots (from summary json) --------

def _ci_bounds(stats, metric):
    """Return (low, high) for the CI key associated with metric, or (None, None)."""
    ci_key = f"{metric}_ci"
    ci = stats.get(ci_key)
    if ci and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def plot_u_curve(summary, out_path, metric="mean", ylabel=None, title=None):
    """3-tier line plot with 95% CI band (if bootstrap was run).

    With bootstrap CIs: shaded band between mean_ci lower and upper bounds.
    Without CIs: fall back to plain line + markers (same as before).
    """
    configs = ordered_configs_from_summary(summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        tier_stats = summary["configs"][cfg]["by_tier"]
        ys, ci_lo, ci_hi = [], [], []
        for t in TIER_ORDER:
            s = tier_stats.get(t, {})
            ys.append(s.get(metric))
            lo, hi = _ci_bounds(s, metric)
            ci_lo.append(lo)
            ci_hi.append(hi)
        if all(y is None for y in ys):
            continue

        ys_plot = np.array([y if y is not None else np.nan for y in ys])
        color = CONFIG_COLORS.get(cfg, None)
        label = CONFIG_LABELS.get(cfg, cfg)

        # Main line
        ax.plot(TIER_ORDER, ys_plot, marker="o", linewidth=2.5, markersize=9,
                color=color, label=label)

        # CI band (only if every tier has CIs; partial bands are misleading)
        if all(lo is not None and hi is not None for lo, hi in zip(ci_lo, ci_hi)):
            ax.fill_between(
                range(len(TIER_ORDER)),
                np.array(ci_lo), np.array(ci_hi),
                color=color, alpha=0.18, linewidth=0,
            )

    ax.set_xlabel("Difficulty tier (rubric positive-point tertiles)", fontsize=11)
    ax.set_ylabel(ylabel or metric, fontsize=11)
    ax.set_title(title or f"{metric} across difficulty tiers", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if metric == "fail_rate":
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_theme_fails(summary, out_path,
                     priority_themes=("emergency_referrals", "hedging",
                                      "context_seeking", "complex_responses")):
    configs = ordered_configs_from_summary(summary)
    all_themes = set()
    for cfg in configs:
        all_themes.update(summary["configs"][cfg]["by_theme"].keys())
    themes = [t for t in priority_themes if t in all_themes]
    themes += sorted(t for t in all_themes if t not in themes)
    if not themes:
        print(f"  (no theme data; skipping {out_path})")
        return

    n_configs = len(configs)
    n_themes = len(themes)
    width = 0.8 / max(n_configs, 1)

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_themes), 5.5))
    xs = range(n_themes)
    for i, cfg in enumerate(configs):
        ys, errs_lo, errs_hi = [], [], []
        for theme in themes:
            s = summary["configs"][cfg]["by_theme"].get(theme, {})
            if s.get("n"):
                ys.append(s["fail_rate"])
                lo, hi = _ci_bounds(s, "fail_rate")
                if lo is not None and hi is not None:
                    errs_lo.append(s["fail_rate"] - lo)
                    errs_hi.append(hi - s["fail_rate"])
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            else:
                ys.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
        offsets = [x + (i - (n_configs - 1) / 2) * width for x in xs]
        has_ci = any(l > 0 or h > 0 for l, h in zip(errs_lo, errs_hi))
        ax.bar(offsets, ys, width,
               color=CONFIG_COLORS.get(cfg, None),
               label=CONFIG_LABELS.get(cfg, cfg),
               yerr=[errs_lo, errs_hi] if has_ci else None,
               error_kw={"ecolor": "black", "capsize": 2, "elinewidth": 0.8,
                         "alpha": 0.6})

    ax.set_xticks(list(xs))
    ax.set_xticklabels(themes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Failure rate (score < threshold)", fontsize=11)
    thr = summary["thresholds"].get("fail_below")
    ax.set_title(
        "Failure rate by HealthBench theme"
        + (f" (threshold = {thr})" if thr is not None else ""),
        fontsize=12,
    )
    for j, theme in enumerate(themes):
        if theme in priority_themes[:2]:
            ax.axvspan(j - 0.45, j + 0.45, color="yellow", alpha=0.08, zorder=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


# -------- Smooth-curve plots (from raw eval JSONs) --------

def load_healthbench_pos_points(paths):
    """Map prompt_id -> sum of positive rubric points (the difficulty axis)."""
    pp = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                ex = json.loads(line)
                pp[ex["prompt_id"]] = sum(r["points"] for r in ex["rubrics"] if r["points"] > 0)
    return pp


def load_per_example(eval_jsons, pos_points):
    """Return {config: [(pos_points, score), ...]}."""
    out = {}
    for ep in eval_jsons:
        with open(ep) as f:
            ev = json.load(f)
        name = ev.get("config") or Path(ep).stem
        rows = []
        for r in ev.get("results", []):
            pid = r["prompt_id"]
            if pid in pos_points:
                rows.append((pos_points[pid], r["score"]))
        out[name] = rows
    return out


def equal_frequency_bin_means(xs, ys, n_bins):
    """Bin by equal-frequency on xs, return (bin_centers, bin_means, bin_counts).

    Equal-frequency is more honest than equal-width when the x distribution
    is skewed (HealthBench pos_points is heavily right-skewed).
    """
    if len(xs) < n_bins:
        n_bins = max(1, len(xs))
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    ys = np.asarray(ys)[order]
    # Splits into n_bins roughly equal buckets
    edges = np.array_split(np.arange(len(xs)), n_bins)
    centers = []
    means = []
    counts = []
    for idx in edges:
        if len(idx) == 0:
            continue
        centers.append(float(np.median(xs[idx])))
        means.append(float(np.mean(ys[idx])))
        counts.append(int(len(idx)))
    return np.array(centers), np.array(means), np.array(counts)


def smooth_quadratic(centers, means, n_eval=200):
    """Fit y = a + b*x + c*x^2 on (centers, means). Returns dense (x, y)."""
    if len(centers) < 3:
        return centers, means
    coeffs = np.polyfit(centers, means, deg=2)
    xs = np.linspace(centers.min(), centers.max(), n_eval)
    ys = np.polyval(coeffs, xs)
    return xs, ys


def plot_u_curve_smooth(per_example, out_path, n_bins=10,
                        metric="mean", fail_threshold=0.4):
    """Multi-bin smoothed U-curve.

    For each config: equal-frequency bin into n_bins, compute per-bin mean
    score (or fail rate), plot the bin centers as markers and overlay a
    quadratic fit. Produces a visually curvy line instead of 3 segments.
    """
    configs = ordered_configs(list(per_example.keys()))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in configs:
        rows = per_example[cfg]
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys_raw = [r[1] for r in rows]
        if metric == "fail_rate":
            ys = [1.0 if s < fail_threshold else 0.0 for s in ys_raw]
        else:
            ys = ys_raw
        centers, means, counts = equal_frequency_bin_means(xs, ys, n_bins)
        color = CONFIG_COLORS.get(cfg)
        label = CONFIG_LABELS.get(cfg, cfg)
        # markers at the bin means
        ax.plot(centers, means, "o", markersize=7, color=color, alpha=0.7)
        # smooth quadratic overlay
        sx, sy = smooth_quadratic(centers, means)
        ax.plot(sx, sy, linewidth=2.5, color=color, label=label)

    ax.set_xlabel("Difficulty (rubric positive-point sum)", fontsize=11)
    if metric == "fail_rate":
        ax.set_ylabel(f"Fail rate (score < {fail_threshold})", fontsize=11)
        ax.set_title(f"U-shape (smoothed, {n_bins} bins): fail rate vs difficulty",
                     fontsize=12)
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("Mean score (higher = better)", fontsize=11)
        ax.set_title(f"U-shape (smoothed, {n_bins} bins): score vs difficulty",
                     fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_u_scatter(per_example, out_path, fail_threshold=0.4):
    """Per-example scatter with a quadratic-on-binned-means overlay.

    Shows the raw data distribution — if the U is real you see it, if it
    isn't you see that too. One subplot per config for clarity.
    """
    configs = ordered_configs(list(per_example.keys()))
    n = len(configs)
    if n == 0:
        return
    cols = 2 if n >= 2 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows),
                             sharex=True, sharey=True, squeeze=False)

    for i, cfg in enumerate(configs):
        ax = axes[i // cols][i % cols]
        data = per_example[cfg]
        if not data:
            ax.set_visible(False)
            continue
        xs = np.array([r[0] for r in data])
        ys = np.array([r[1] for r in data])
        color = CONFIG_COLORS.get(cfg, "#444")
        ax.scatter(xs, ys, s=18, alpha=0.35, color=color, edgecolors="none")

        # quadratic fit on equal-freq bin means
        n_bins = min(10, max(3, len(xs) // 10))
        centers, means, _ = equal_frequency_bin_means(xs, ys, n_bins)
        if len(centers) >= 3:
            sx, sy = smooth_quadratic(centers, means)
            ax.plot(sx, sy, linewidth=2.5, color=color)
            ax.plot(centers, means, "o", markersize=7,
                    color="black", markerfacecolor=color)

        ax.axhline(fail_threshold, color="red", linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.set_title(CONFIG_LABELS.get(cfg, cfg), fontsize=11, color=color)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    for ax in axes[-1]:
        ax.set_xlabel("Difficulty (rubric positive-point sum)", fontsize=10)
    for row in axes:
        row[0].set_ylabel("Score", fontsize=10)

    # hide unused panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"Per-example score vs difficulty (red dashed = fail threshold {fail_threshold})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval/ushape.json",
                        help="ushape.json produced by eval_ushape.py "
                             "(used for the 3-tier plots + theme bar chart)")
    parser.add_argument("--eval-jsons", nargs="*", default=None,
                        help="per-config eval JSONs from eval_healthbench.py. "
                             "If provided along with --healthbench, renders "
                             "smoothed multi-bin and scatter plots.")
    parser.add_argument("--healthbench", nargs="*", default=None,
                        help="HealthBench JSONL files (for per-example pos-points lookup)")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="bin count for the smoothed U-curve (default 10)")
    parser.add_argument("--out-dir", default="eval/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always do the tier plots from ushape.json (quick, coarse reference)
    if Path(args.input).exists():
        summary = json.loads(Path(args.input).read_text())
        fail_threshold = summary["thresholds"].get("fail_below", 0.4)
        plot_u_curve(summary, out_dir / "u_curve.png",
                     metric="mean",
                     ylabel="Mean score (higher = better)",
                     title="U-shape: mean score across difficulty tiers")
        plot_u_curve(summary, out_dir / "u_fail.png",
                     metric="fail_rate",
                     ylabel=f"Fail rate (score < {fail_threshold})",
                     title="U-shape: failure rate across difficulty tiers")
        plot_theme_fails(summary, out_dir / "theme_fail.png")
    else:
        print(f"  (--input {args.input} not found; skipping 3-tier plots)")
        fail_threshold = 0.4

    # If raw eval jsons + HealthBench are provided, render smooth + scatter
    if args.eval_jsons and args.healthbench:
        pp = load_healthbench_pos_points(args.healthbench)
        per_example = load_per_example(args.eval_jsons, pp)
        total = sum(len(v) for v in per_example.values())
        print(f"  per-example rows: {total} across {len(per_example)} configs")
        if total >= 3:
            plot_u_curve_smooth(per_example, out_dir / "u_curve_smooth.png",
                                n_bins=args.n_bins, metric="mean",
                                fail_threshold=fail_threshold)
            plot_u_curve_smooth(per_example, out_dir / "u_fail_smooth.png",
                                n_bins=args.n_bins, metric="fail_rate",
                                fail_threshold=fail_threshold)
            plot_u_scatter(per_example, out_dir / "u_scatter.png",
                           fail_threshold=fail_threshold)
        else:
            print("  (not enough per-example data for smooth plots)")
    else:
        print("  (no --eval-jsons/--healthbench; skipping smooth + scatter plots)")

    print(f"\nWrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
