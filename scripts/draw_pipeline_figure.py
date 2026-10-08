"""Draw the paper's pipeline overview figure as a polished PDF."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


STAGES = [
    {
        "num": "1",
        "x": 0.04,
        "w": 0.15,
        "accent": "#5B8FF9",
        "title": "Prompt Pool",
        "body": "HealthBench Full + Hard\nexclude 200-prompt\nholdout evaluation set",
    },
    {
        "num": "2",
        "x": 0.22,
        "w": 0.16,
        "accent": "#43AA8B",
        "title": "Trace Generation",
        "body": "BODHI wrapper + base model\ngreedy decoding\nproduce teaching traces",
    },
    {
        "num": "3",
        "x": 0.41,
        "w": 0.16,
        "accent": "#F6BD60",
        "title": "Rubric Grading",
        "body": "Qwen-14B grader\nscore each rubric criterion\nproduce quality signal",
    },
    {
        "num": "4",
        "x": 0.60,
        "w": 0.16,
        "accent": "#C77DFF",
        "title": "Filter + Split",
        "body": "keep traces with\nscore >= 0.4\n90/10 train/val",
    },
    {
        "num": "5",
        "x": 0.80,
        "w": 0.17,
        "accent": "#F9844A",
        "title": "LoRA Tune + Eval",
        "body": "train adapter, then compare\nBase | Base+BODHI\nLoRA | LoRA+BODHI",
    },
]

CARD_Y = 0.27
CARD_H = 0.44

EDGE = "#1F2A44"
TEXT = "#23314D"
MUTED = "#5D6B82"
BG = "#F6F7FB"


def add_rounded_panel(ax, x, y, w, h, facecolor, edgecolor="none", lw=1.0, alpha=1.0):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(panel)
    return panel


def add_lane_headers(ax):
    add_rounded_panel(ax, 0.05, 0.84, 0.71, 0.08, "#E8F6EF", edgecolor="#B7E4CC", lw=1.2)
    add_rounded_panel(ax, 0.81, 0.84, 0.14, 0.08, "#FFF1E8", edgecolor="#F4B183", lw=1.2)

    ax.text(
        0.065,
        0.88,
        "TRAINING-TIME DISTILLATION",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
        color="#236A4B",
    )
    ax.text(
        0.065,
        0.845,
        "Wrapper appears only while generating demonstrations.",
        ha="left",
        va="center",
        fontsize=8,
        color="#3F7D62",
    )

    ax.text(
        0.88,
        0.88,
        "EVAL / SERVE",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#B85C2E",
    )
    ax.text(
        0.88,
        0.845,
        "No wrapper",
        ha="center",
        va="center",
        fontsize=8,
        color="#C96A39",
    )


def add_stage(ax, stage):
    x = stage["x"]
    w = stage["w"]
    accent = stage["accent"]

    add_rounded_panel(ax, x + 0.008, CARD_Y - 0.015, w, CARD_H, "#D9DEE8", alpha=0.45)
    add_rounded_panel(ax, x, CARD_Y, w, CARD_H, "white", edgecolor=EDGE, lw=1.5)
    add_rounded_panel(ax, x, CARD_Y + CARD_H - 0.09, w, 0.09, accent)

    badge = Circle((x + 0.038, CARD_Y + CARD_H - 0.045), 0.027, facecolor="white", edgecolor=accent, linewidth=1.6)
    ax.add_patch(badge)
    ax.text(
        x + 0.038,
        CARD_Y + CARD_H - 0.045,
        stage["num"],
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=accent,
    )

    ax.text(
        x + 0.072,
        CARD_Y + CARD_H - 0.045,
        stage["title"],
        ha="left",
        va="center",
        fontsize=8.9,
        weight="bold",
        color="white",
    )

    ax.text(
        x + 0.02,
        CARD_Y + CARD_H - 0.13,
        stage["body"],
        ha="left",
        va="top",
        fontsize=8.4,
        color=TEXT,
        linespacing=1.35,
    )


def add_connectors(ax):
    y = CARD_Y + CARD_H / 2
    for left, right in zip(STAGES, STAGES[1:]):
        start = left["x"] + left["w"] + 0.01
        end = right["x"] - 0.01
        ax.add_patch(
            FancyArrowPatch(
                (start, y),
                (end, y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.0,
                color="#55637D",
            )
        )


def add_serve_note(ax):
    add_rounded_panel(ax, 0.80, 0.14, 0.17, 0.07, "#FFF4EC", edgecolor="#F4B183", lw=1.2)
    ax.text(
        0.885,
        0.175,
        "Adapter is evaluated and served\nwithout the BODHI wrapper.",
        ha="center",
        va="center",
        fontsize=8,
        color="#A65427",
        weight="bold",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="paper/figures/pipeline.pdf",
        help="where to write the PDF",
    )
    args = parser.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_lane_headers(ax)
    for stage in STAGES:
        add_stage(ax, stage)
    add_connectors(ax)
    add_serve_note(ax)

    ax.text(
        0.04,
        0.06,
        "Pipeline summary: generate wrapper-conditioned demonstrations, grade them, filter for quality, "
        "then distill the behavior into a LoRA adapter.",
        ha="left",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
