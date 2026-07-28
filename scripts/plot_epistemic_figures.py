"""Generate paper-ready epistemic virtue figures from cross-seed aggregate results.

Produces:
  figures/epistemic_bar.pdf   -- grouped bar chart: 4 configs × key dimensions
  figures/epistemic_seeds.pdf -- per-seed dot plot for the primary metrics

Usage:
    python scripts/plot_epistemic_figures.py \
        --aggregate results_modal/cross_seed_aggregate.json \
        --out-dir paper/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Stable config ordering and palette ───────────────────────────────────────

CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]
CONFIG_LABELS = {
    "base_no_wrapper": "Base",
    "base_bodhi":      "Base + BODHI",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi":      "LoRA + BODHI",
}
# Colorblind-friendly palette (Paul Tol's bright)
CONFIG_COLORS = {
    "base_no_wrapper": "#CC3311",   # red
    "base_bodhi":      "#EE7733",   # orange
    "lora_no_wrapper": "#0077BB",   # blue  (headline)
    "lora_bodhi":      "#009988",   # teal
}

# ── Metrics to include in the bar chart ─────────────────────────────────────

# (key in stage5_epistemic, display label, y-axis label, is_pct, multiplier)
# Panel A: dimensions graded 0-2.  Panel B: dimensions reported as rates.
# Split because a 0-80% bar and a 0-2 bar on one axis is unreadable.
PANEL_A_METRICS = [
    ("uncertainty_acknowledgment_mean", "Uncertainty"),
    ("context_seeking_mean",            "Context-seeking"),
    ("red_flag_identification_mean",    "Red-flag ID"),
    ("scope_bounding_mean",             "Scope bounding"),
    ("hedging_mean",                    "Hedging"),
    ("specificity_mean",                "Specificity"),
]
PANEL_B_METRICS = [
    ("active_inquiry_rate",     "Active inquiry"),
    ("red_flag_rate",           "Red-flag"),
    ("scope_bounded_rate",      "Scope-bounded"),
    ("blanket_disclaimer_rate", "Blanket disclaimer"),
]

# Per-seed keys for the seed-consistency figure
SEED_METRICS = [
    ("active_inquiry_rate",         "Active inquiry (%)",    True,  100),
    ("red_flag_identification_mean","Red-flag ID (0–2)",     False, 1),
]


def load_aggregate(path):
    with open(path) as f:
        return json.load(f)


# ── Bar chart ────────────────────────────────────────────────────────────────

def plot_epistemic_bar(ep, out_path):
    """Two stacked panels: 0-2 scored dimensions above, percentage rates below."""
    bar_w, group_gap = 0.18, 0.9
    n_cfgs = len(CONFIG_ORDER)

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"hspace": 0.42})

    def draw(ax, metrics, scale, ylabel, ymax):
        for i, cfg in enumerate(CONFIG_ORDER):
            for j, (key, _) in enumerate(metrics):
                rec = ep[cfg][key]
                x = j * group_gap + (i - (n_cfgs - 1) / 2) * bar_w
                ax.bar(x, rec["mean"] * scale, bar_w,
                       color=CONFIG_COLORS[cfg],
                       yerr=rec["std"] * scale, capsize=3,
                       error_kw={"elinewidth": 1, "ecolor": "#333", "capthick": 1},
                       zorder=3)
        ax.set_xticks([j * group_gap for j in range(len(metrics))])
        ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    draw(axA, PANEL_A_METRICS, 1.0, "Score (0\u20132)", 2.15)
    draw(axB, PANEL_B_METRICS, 100.0, "Rate (%)", 105)
    axA.set_title("A. Graded dimensions", fontsize=11, loc="left")
    axB.set_title("B. Behavioral rates", fontsize=11, loc="left")

    handles = [mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
               for c in CONFIG_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.99))

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Per-seed dot plot ────────────────────────────────────────────────────────

def plot_seed_consistency(ep, seeds, out_path):
    """
    Per-seed dot plot for the primary metrics.
    Each config gets a column; each seed is a dot; the cross-seed mean is a
    horizontal tick. Shows that the between-seed variance is small relative
    to the between-config effect.
    """
    n_metrics = len(SEED_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5),
                             sharey=False)
    if n_metrics == 1:
        axes = [axes]

    seed_keys = [f"seed_{s}" for s in seeds]

    for ax, (key, ylabel, is_pct, mult) in zip(axes, SEED_METRICS):
        x_pos = {cfg: i for i, cfg in enumerate(CONFIG_ORDER)}
        jitter = np.linspace(-0.1, 0.1, len(seed_keys))

        for cfg in CONFIG_ORDER:
            if cfg not in ep:
                continue
            data = ep[cfg]
            metric_data = data.get(key, {})
            per_seed = metric_data.get("per_seed", {})
            mean = metric_data.get("mean", 0) * mult

            xi = x_pos[cfg]
            # Per-seed dots
            for ji, sk in zip(jitter, seed_keys):
                val = per_seed.get(sk, None)
                if val is None:
                    continue
                ax.scatter(xi + ji, val * mult,
                           color=CONFIG_COLORS[cfg], s=55, zorder=4,
                           alpha=0.85, edgecolors="white", linewidths=0.5)
            # Cross-seed mean tick
            ax.plot([xi - 0.22, xi + 0.22], [mean, mean],
                    color=CONFIG_COLORS[cfg], linewidth=2.5, zorder=5)

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split(" (")[0], fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Per-seed consistency (dots = individual seeds, bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default="results_modal/cross_seed_aggregate.json",
    )
    parser.add_argument(
        "--out-dir",
        default="paper/figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_aggregate(args.aggregate)
    ep = agg.get("stage5_epistemic", {})
    seeds = agg.get("seeds", [7, 13, 42, 99, 101])

    plot_epistemic_bar(ep, out_dir / "epistemic_bar.pdf")
    plot_seed_consistency(ep, seeds, out_dir / "epistemic_seeds.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
