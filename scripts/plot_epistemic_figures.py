"""Figure E.1 and the per-seed figure, from one cell's analysis/stats.json.

Produces:
  figures/epistemic_bar.pdf   -- grouped bars: the four conditions on the dimensions
  figures/epistemic_seeds.pdf -- per-seed dots for active inquiry and red-flag identification

The values are the seed means and the SD across seeds that the uniform analysis writes
(conditions.<metric>.<condition>.seed_mean, seed_sd, per_seed), the same numbers as the
tables. The cell is the first detail cell of configs/report.yaml unless --cell names
another, so a new main cell gets its figures from the config, not from code. Until
2026-10-11 the figures read the 2026-05 cell's cross-seed summary, which pooled replies
in its own way; that file stays as a moved record (results/PROVENANCE.md).

Usage:
    python scripts/plot_epistemic_figures.py [--cell <cell id>] --out-dir paper/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Conditions by role, labels and palette ───────────────────────────────────

# The four roles of the 2x2 design; {protocol} is the cell's own protocol id.
ROLES = ["base", "base+{protocol}", "lora", "lora+{protocol}"]
# Colorblind-friendly palette (Paul Tol's bright), one color per role.
ROLE_COLORS = ["#CC3311", "#EE7733", "#0077BB", "#009988"]

# ── Metrics to include in the bar chart ─────────────────────────────────────

# (metric of stats.json, display label). Panel A: dimensions graded 0-2. Panel B: rates.
# Split because a 0-80% bar and a 0-2 bar on one axis is unreadable.
PANEL_A_METRICS = [
    ("uncertainty_ack", "Uncertainty"),
    ("context_seeking", "Context-seeking"),
    ("red_flag",        "Red-flag ID"),
    ("scope_bounding",  "Scope bounding"),
    ("hedging_quality", "Hedging"),
    ("specificity",     "Specificity"),
]
PANEL_B_METRICS = [
    ("active_inquiry",     "Active inquiry"),
    ("red_flag_rate",      "Red-flag"),
    ("scope_bounded_rate", "Scope-bounded"),
    ("blanket_disclaimer", "Blanket disclaimer"),
]

# Per-seed metrics: (metric, y label, multiplier)
SEED_METRICS = [
    ("active_inquiry", "Active inquiry (%)", 100),
    ("red_flag",       "Red-flag ID (0–2)",  1),
]


def default_cell() -> str:
    """The first detail cell of configs/report.yaml: the cell whose detail the paper prints."""
    sys.path.insert(0, str(ROOT))
    from bodhi_lora.analysis import report_config

    cells = report_config().detail_cells
    if not cells:
        raise SystemExit("configs/report.yaml names no detail cell; pass --cell")
    return cells[0]


def conditions_of(stats: dict, protocol: str) -> list[tuple[str, str, str]]:
    """(condition, label, color) for each role the cell has."""
    name = protocol.upper()
    labels = {"base": "Base", "lora": "LoRA"}
    out = []
    for role, color in zip(ROLES, ROLE_COLORS):
        cond = role.replace("{protocol}", protocol)
        if cond in stats["conditions_order"]:
            weights, _, proto = cond.partition("+")
            out.append((cond, labels[weights] + (f" + {name}" if proto else ""), color))
    return out


def load_stats(path):
    with open(path) as f:
        return json.load(f)


# ── Bar chart ────────────────────────────────────────────────────────────────

def plot_epistemic_bar(stats, conds, out_path):
    """Two stacked panels: 0-2 scored dimensions above, percentage rates below."""
    bar_w, group_gap = 0.18, 0.9
    n_cfgs = len(conds)

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"hspace": 0.42})

    def draw(ax, metrics, scale, ylabel, ymax):
        for i, (cond, _, color) in enumerate(conds):
            for j, (key, _) in enumerate(metrics):
                rec = stats["conditions"][key][cond]
                x = j * group_gap + (i - (n_cfgs - 1) / 2) * bar_w
                ax.bar(x, rec["seed_mean"] * scale, bar_w,
                       color=color,
                       yerr=(rec["seed_sd"] or 0) * scale, capsize=3,
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

    handles = [mpatches.Patch(color=color, label=label) for _, label, color in conds]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.99))

    # No creation date in the PDF, so a rebuild from the same data is byte-identical.
    fig.savefig(out_path, dpi=200, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return fig


# ── Per-seed dot plot ────────────────────────────────────────────────────────

def plot_seed_consistency(stats, conds, out_path):
    """
    Per-seed dot plot for the primary metrics.
    Each condition gets a column; each seed is a dot; the mean of the seed means is a
    horizontal tick. Shows that the between-seed variance is small relative
    to the between-condition effect.
    """
    n_metrics = len(SEED_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5),
                             sharey=False)
    if n_metrics == 1:
        axes = [axes]

    seeds = [str(s) for s in stats["seeds"]]
    jitter = np.linspace(-0.1, 0.1, len(seeds))

    for ax, (key, ylabel, mult) in zip(axes, SEED_METRICS):
        for xi, (cond, _, color) in enumerate(conds):
            rec = stats["conditions"][key][cond]
            for ji, s in zip(jitter, seeds):
                val = rec["per_seed"].get(s)
                if val is None:
                    continue
                ax.scatter(xi + ji, val * mult,
                           color=color, s=55, zorder=4,
                           alpha=0.85, edgecolors="white", linewidths=0.5)
            mean = rec["seed_mean"] * mult
            ax.plot([xi - 0.22, xi + 0.22], [mean, mean],
                    color=color, linewidth=2.5, zorder=5)

        ax.set_xticks(list(range(len(conds))))
        ax.set_xticklabels([label for _, label, _ in conds],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split(" (")[0], fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Per-seed consistency (dots = individual seeds, bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=None,
                        help="the cell to plot; default the first detail cell of configs/report.yaml")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    cell = args.cell or default_cell()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(Path(args.results_root) / cell / "analysis" / "stats.json")
    # The cell id is <benchmark>__<model>__<protocol>[__<tag>].
    conds = conditions_of(stats, cell.split("__")[2])

    plot_epistemic_bar(stats, conds, out_dir / "epistemic_bar.pdf")
    plot_seed_consistency(stats, conds, out_dir / "epistemic_seeds.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
