"""Generate all post-processing figures from the full 5-seed pipeline run.

Saves to paper/figures/:
  ushape_difficulty.pdf      -- failure rate vs. prompt difficulty (pooled 5 seeds)
  axis_breakdown.pdf         -- mean score by HealthBench rubric axis (pooled 5 seeds)
  score_distribution.pdf     -- per-prompt score violin plot (pooled 5 seeds)
  effect_size_forest.pdf     -- Cohen's d forest plot (reads the cell's analysis/stats.json)

Usage:
    python scripts/plot_postprocessing.py [--data-dir <cell folder>] [--hb-hard data/raw/healthbench/hard.jsonl] [--out-dir paper/figures]

The default cell folder is results/healthbench__medgemma-27b-text__bodhi__2026-05.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bodhi_lora.registry import condition_from_legacy  # noqa: E402

# ── Palette and config ordering (match epistemic figures) ────────────────────

CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]
CONFIG_LABELS = {
    "base_no_wrapper": "Base",
    "base_bodhi":      "Base + BODHI",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi":      "LoRA + BODHI",
}
CONFIG_COLORS = {
    "base_no_wrapper": "#CC3311",
    "base_bodhi":      "#EE7733",
    "lora_no_wrapper": "#0077BB",
    "lora_bodhi":      "#009988",
}

SEEDS = [7, 13, 42, 99, 101]

AXIS_LABELS = {
    "axis:accuracy":             "Accuracy",
    "axis:completeness":         "Completeness",
    "axis:communication_quality":"Communication",
    "axis:context_awareness":    "Context\nawareness",
    "axis:instruction_following":"Instruction\nfollowing",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_healthbench_difficulty(hb_path):
    """Return dict: prompt_id -> total positive rubric points (difficulty proxy)."""
    difficulty = {}
    with open(hb_path) as f:
        for line in f:
            p = json.loads(line)
            diff = sum(rb["points"] for rb in p["rubrics"] if rb["points"] > 0)
            difficulty[p["prompt_id"]] = diff
    return difficulty


def load_health_results(data_dir, seeds=SEEDS, configs=None):
    """
    Load per-prompt HealthBench scores for all seeds and configs.
    Returns dict: config -> list of {prompt_id, score, tag_scores, ...}
    """
    if configs is None:
        configs = CONFIG_ORDER
    pooled = {cfg: [] for cfg in configs}
    for seed in seeds:
        for cfg in configs:
            # A missing seed would silently shrink the pooled figures, so it is an error.
            path = Path(data_dir) / f"seed_{seed}" / "eval" / f"{condition_from_legacy(cfg)}.json"
            if not path.exists():
                raise SystemExit(f"plot_postprocessing: missing {path}")
            d = json.load(open(path))
            for r in d["results"]:
                pooled[cfg].append(r)
    return pooled


# ── Figure 1: U-shape (failure rate vs. difficulty) ──────────────────────────

def plot_ushape(health_results, difficulty, out_path, n_bins=10):
    """
    Bin prompts by difficulty (equal-frequency), plot mean failure rate
    (1 - score) and 95% bootstrap CI per bin for all 4 conditions.
    """
    # Build per-config (prompt_id, score) lists
    all_prompts = set()
    for cfg in CONFIG_ORDER:
        for r in health_results[cfg]:
            all_prompts.add(r["prompt_id"])

    # Compute difficulty for prompts that appear in results
    diffs_list = sorted(
        [(pid, difficulty.get(pid, 0)) for pid in all_prompts],
        key=lambda x: x[1]
    )
    # Equal-frequency bins
    bin_size = len(diffs_list) // n_bins
    bins = []
    for i in range(n_bins):
        chunk = diffs_list[i * bin_size: (i + 1) * bin_size]
        if chunk:
            bins.append({
                "pids": {c[0] for c in chunk},
                "diff_mid": np.median([c[1] for c in chunk]),
                "diff_range": (chunk[0][1], chunk[-1][1]),
            })

    # Score lookup per config
    score_lookup = {}
    for cfg in CONFIG_ORDER:
        score_lookup[cfg] = {r["prompt_id"]: r["score"] for r in health_results[cfg]}

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for cfg in CONFIG_ORDER:
        xs, ys, errs = [], [], []
        for bn in bins:
            vals = [
                1.0 - score_lookup[cfg][pid]
                for pid in bn["pids"]
                if pid in score_lookup[cfg]
            ]
            if not vals:
                continue
            # Bootstrap 95% CI on the mean
            boots = [np.mean(random.choices(vals, k=len(vals))) for _ in range(500)]
            xs.append(bn["diff_mid"])
            ys.append(np.mean(vals))
            errs.append(1.96 * np.std(boots))

        ax.plot(xs, ys, "-o", color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg],
                linewidth=2, markersize=5, zorder=4)
        ax.fill_between(xs,
                         [y - e for y, e in zip(ys, errs)],
                         [y + e for y, e in zip(ys, errs)],
                         color=CONFIG_COLORS[cfg], alpha=0.12, zorder=2)

    ax.set_xlabel("Prompt difficulty (total rubric points, binned)", fontsize=11)
    ax.set_ylabel("Failure rate (1 − score)", fontsize=11)
    ax.set_title("Failure rate vs. prompt difficulty\n"
                 "(pooled 5 seeds, 200 prompts/seed; shading = 95% bootstrap CI)",
                 fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
        for c in CONFIG_ORDER
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Figure 2: Axis breakdown ─────────────────────────────────────────────────

def plot_axis_breakdown(health_results, out_path):
    """
    Grouped bar chart: mean score per HealthBench rubric axis, 4 conditions.
    """
    axes_keys = sorted(AXIS_LABELS.keys())
    n_axes = len(axes_keys)
    n_cfgs = len(CONFIG_ORDER)
    bar_w = 0.18
    group_gap = 0.9

    # Compute mean axis scores
    means = {}
    for cfg in CONFIG_ORDER:
        means[cfg] = {}
        for ax_key in axes_keys:
            vals = []
            for r in health_results[cfg]:
                ts = r.get("tag_scores", {})
                if ax_key in ts:
                    vals.append(ts[ax_key])
            means[cfg][ax_key] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, cfg in enumerate(CONFIG_ORDER):
        for j, ax_key in enumerate(axes_keys):
            x = j * group_gap + (i - (n_cfgs - 1) / 2) * bar_w
            ax.bar(x, means[cfg][ax_key], bar_w,
                   color=CONFIG_COLORS[cfg], zorder=3, alpha=0.9)

    xtick_positions = [j * group_gap for j in range(n_axes)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([AXIS_LABELS[k] for k in axes_keys], fontsize=10)
    ax.set_ylabel("Mean axis score (0–1)", fontsize=11)

    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
        for c in CONFIG_ORDER
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Figure 3: Score distribution violin ──────────────────────────────────────

def plot_score_distribution(health_results, out_path):
    """Violin plot of per-prompt HealthBench score for each condition."""
    scores_by_cfg = [
        [r["score"] for r in health_results[cfg]]
        for cfg in CONFIG_ORDER
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    parts = ax.violinplot(scores_by_cfg, positions=range(len(CONFIG_ORDER)),
                          showmedians=True, showextrema=True, widths=0.7)

    for i, (pc, cfg) in enumerate(zip(parts["bodies"], CONFIG_ORDER)):
        pc.set_facecolor(CONFIG_COLORS[cfg])
        pc.set_alpha(0.6)
        pc.set_edgecolor(CONFIG_COLORS[cfg])

    # Median bar color
    parts["cmedians"].set_color("#333")
    parts["cbars"].set_color("#555")
    parts["cmins"].set_color("#555")
    parts["cmaxes"].set_color("#555")

    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=10)
    ax.set_ylabel("HealthBench score (0–1)", fontsize=11)
    ax.set_title("Per-prompt score distribution by evaluation condition\n"
                 "(pooled 5 seeds; line = median)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Figure 5: Effect size forest plot ────────────────────────────────────────

def plot_effect_sizes(stats_path, out_path):
    """
    Cohen's d forest plot for each epistemic dimension.

    d and its 95% CI come from the cell's analysis/stats.json (python -m bodhi_lora
    analyze): pooled d over the scored responses, CI from a bootstrap that resamples
    prompt ids.
    The 5 seed draws overlap (667 unique prompts per 1,000 draws), so a normal-
    approximation CI that treats responses as independent would be too narrow.
    The seed-level paired tests are in the paper's tables; this figure is descriptive.
    """
    with open(stats_path) as f:
        contrasts = json.load(f)["contrasts"]

    dim_keys = [
        ("active_inquiry",  "Active inquiry"),
        ("context_seeking", "Context-seeking (0–2)"),
        ("red_flag",        "Red-flag ID (0–2)"),
        ("scope_bounding",  "Scope bounding (0–2)"),
        ("uncertainty_ack", "Uncertainty ack. (0–2)"),
        ("hedging_quality", "Hedging quality (0–2)"),
        ("specificity",     "Specificity (0–2)"),
    ]

    comparisons = [
        ("lora - base",       "LoRA vs Base",                CONFIG_COLORS["lora_no_wrapper"]),
        ("base+bodhi - base", "Base + BODHI vs Base",        "#EE7733"),
        ("lora+bodhi - lora", "LoRA + BODHI vs LoRA",        "#009988"),
    ]

    n_dims = len(dim_keys)
    group_gap = 2.8
    offsets = [-0.8, 0.0, 0.8]

    fig, ax = plt.subplots(figsize=(9, max(5, n_dims * 0.9 + 1)))
    group_centers = [i * group_gap for i in range(n_dims)]

    for ci, (ctok, label, color) in enumerate(comparisons):
        for gi, (dim_key, _) in enumerate(dim_keys):
            r = contrasts[ctok]["metrics"][dim_key]
            d = r["pooled_d"]
            lo, hi = r["d_ci95"]
            y = group_centers[gi] + offsets[ci]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, alpha=0.8, zorder=3)
            ax.scatter([d], [y], color=color, s=55, zorder=5, edgecolors="white",
                       linewidths=0.5)

    ax.axvline(0, color="#333", linewidth=1, linestyle="--", zorder=2, alpha=0.6)
    ax.set_yticks(group_centers)
    ax.set_yticklabels([dk[1] for dk in dim_keys], fontsize=10)
    ax.set_xlabel("Pooled Cohen's $d$ (first condition minus second)", fontsize=11)
    ax.set_title("Effect sizes for the seven epistemic dimensions\n"
                 "(95% CI: bootstrap over prompt ids; descriptive)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Shaded band drawn before the legend so it can supply a handle.
    ax.axvspan(-0.2, 0.2, alpha=0.04, color="#aaa", zorder=1)

    # Legend sits below the x-label, not in the axes, so it hides no point.
    handles = [mpatches.Patch(color=c, label=lbl) for (_, lbl, c) in comparisons]
    handles.append(mpatches.Patch(color="#aaa", alpha=0.25,
                                  label="$|d| < 0.2$"))
    ax.legend(handles=handles, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="results/healthbench__medgemma-27b-text__bodhi__2026-05")
    parser.add_argument("--hb-hard",   default="data/raw/healthbench/hard.jsonl")
    parser.add_argument("--out-dir",   default="paper/figures")
    # The forest plot reads d and its CI from the cell's uniform analysis (make analyze).
    parser.add_argument("--stats",     default=None,
                        help="the cell's analysis/stats.json (default: <data-dir>/analysis/stats.json)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    np.random.seed(0)

    print("Loading HealthBench Hard difficulty scores …")
    difficulty = load_healthbench_difficulty(args.hb_hard)

    print("Loading per-prompt HealthBench results (5 seeds × 4 configs) …")
    health_results = load_health_results(args.data_dir)

    print("\nGenerating figures …")

    plot_ushape(
        health_results, difficulty,
        out_dir / "ushape_difficulty.pdf"
    )

    plot_axis_breakdown(
        health_results,
        out_dir / "axis_breakdown.pdf"
    )

    plot_score_distribution(
        health_results,
        out_dir / "score_distribution.pdf"
    )

    plot_effect_sizes(
        args.stats or Path(args.data_dir) / "analysis" / "stats.json",
        out_dir / "effect_size_forest.pdf"
    )

    print("\nDone. All figures saved to", out_dir)


if __name__ == "__main__":
    main()
