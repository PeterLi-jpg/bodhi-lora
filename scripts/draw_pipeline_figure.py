"""Draw the paper's pipeline overview figure as a PDF."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOXES = [
    (
        0.03,
        0.38,
        0.16,
        0.24,
        "1. HealthBench prompts\nFull + Hard\nminus 200 eval holdout",
        "#e8f1ff",
    ),
    (
        0.22,
        0.38,
        0.16,
        0.24,
        "2. BODHI wrapper\n+ base model\n+ greedy decoding",
        "#eef7e6",
    ),
    (
        0.41,
        0.38,
        0.16,
        0.24,
        "3. Per-criterion\nrubric grading\nQwen-14B grader",
        "#fff5db",
    ),
    (
        0.60,
        0.38,
        0.16,
        0.24,
        "4. Score filter\n>= 0.4\n90/10 train/val split",
        "#f8e8f2",
    ),
    (
        0.79,
        0.30,
        0.18,
        0.40,
        "5. LoRA SFT\nthen 200-prompt holdout eval\nbase / base+BODHI /\nLoRA / LoRA+BODHI\n\nNo wrapper at serve time",
        "#ffe7d6",
    ),
]


def add_box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6,
            edgecolor="#24364b",
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        family="DejaVu Sans",
        weight="bold" if "No wrapper at serve time" in text else "normal",
    )


def add_arrow(ax, start_x, end_x, y):
    ax.add_patch(
        FancyArrowPatch(
            (start_x, y),
            (end_x, y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="#24364b",
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for box in BOXES:
        add_box(ax, *box)

    add_arrow(ax, 0.19, 0.22, 0.50)
    add_arrow(ax, 0.38, 0.41, 0.50)
    add_arrow(ax, 0.57, 0.60, 0.50)
    add_arrow(ax, 0.76, 0.79, 0.50)

    ax.text(
        0.31,
        0.70,
        "Training-time only wrapper",
        fontsize=8,
        color="#2f6b2f",
        weight="bold",
        ha="center",
    )
    ax.text(
        0.88,
        0.73,
        "Serve-time LoRA model\nruns without BODHI wrapper",
        fontsize=8,
        color="#8b3d13",
        weight="bold",
        ha="center",
    )

    out = Path("paper/figures/pipeline.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
