"""Analyze graded BOHDI traces and render a small summary bundle."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def response_lengths(rows):
    return [len(row.get("response", "").split()) for row in rows]


def theme_counts(rows):
    counts = Counter()
    for row in rows:
        grade = row.get("grade", {})
        theme_keys = [
            tag.split(":", 1)[1]
            for tag in grade.get("tag_scores", {})
            if tag.startswith("theme:")
        ]
        for theme in theme_keys:
            counts[theme] += 1
    return counts


def grader_retry_disagreement(rows):
    total = 0
    disagreed = 0
    for row in rows:
        for item in row.get("grade", {}).get("criteria_results", []):
            raw_outputs = item.get("raw_grader_outputs", [])
            if len(raw_outputs) < 2:
                continue
            parsed = []
            for raw in raw_outputs:
                if '"criteria_met": true' in raw.lower():
                    parsed.append(True)
                elif '"criteria_met": false' in raw.lower():
                    parsed.append(False)
            if len(parsed) < 2:
                continue
            total += 1
            if len(set(parsed)) > 1:
                disagreed += 1
    return {
        "criteria_items_with_multiple_attempts": total,
        "criteria_items_with_disagreement": disagreed,
        "disagreement_rate": (disagreed / total) if total else 0.0,
    }


def plot_hist(values, title, xlabel, out_path, bins=20):
    plt.figure(figsize=(7, 4.2))
    plt.hist(values, bins=bins, color="#457b9d", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_theme_counts(counts, out_path):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    plt.figure(figsize=(8, max(4, len(labels) * 0.3)))
    plt.barh(labels, values, color="#e76f51")
    plt.gca().invert_yaxis()
    plt.title("Theme Distribution After Filtering")
    plt.xlabel("Trace count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sft/all_graded.jsonl")
    parser.add_argument("--output", default="data/sft/trace_analysis.json")
    parser.add_argument("--figures-dir", default="data/sft/figures")
    parser.add_argument("--min-theme-count", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(args.input)
    scores = [row.get("grade", {}).get("overall_score") for row in rows if row.get("grade")]
    lengths = response_lengths(rows)
    counts = theme_counts(rows)
    disagreement = grader_retry_disagreement(rows)
    sparse_themes = sorted([theme for theme, count in counts.items() if count < args.min_theme_count])

    out = Path(args.output)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    score_fig = fig_dir / "score_histogram.png"
    length_fig = fig_dir / "response_length_histogram.png"
    theme_fig = fig_dir / "theme_distribution.png"

    if scores:
        plot_hist(scores, "Trace Score Distribution", "overall_score", score_fig)
    if lengths:
        plot_hist(lengths, "Response Length Distribution", "response length (whitespace tokens)", length_fig)
    if counts:
        plot_theme_counts(counts, theme_fig)

    summary = {
        "n_traces": len(rows),
        "score_summary": {
            "min": float(min(scores)) if scores else None,
            "max": float(max(scores)) if scores else None,
            "mean": float(np.mean(scores)) if scores else None,
            "median": float(np.median(scores)) if scores else None,
        },
        "response_length_summary": {
            "min": int(min(lengths)) if lengths else None,
            "max": int(max(lengths)) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
        },
        "theme_counts": dict(sorted(counts.items())),
        "sparse_themes_below_threshold": sparse_themes,
        "min_theme_count_threshold": args.min_theme_count,
        "grader_retry_disagreement": disagreement,
        "figures": {
            "score_histogram": str(score_fig),
            "response_length_histogram": str(length_fig),
            "theme_distribution": str(theme_fig),
        },
    }

    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
