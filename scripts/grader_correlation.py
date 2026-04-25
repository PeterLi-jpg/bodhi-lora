"""Compare per-example eval scores across two grader families."""

import argparse
import json
from pathlib import Path


def load_eval(path):
    with open(path) as handle:
        payload = json.load(handle)

    scores_by_prompt = {}
    for row in payload.get("results", []):
        prompt_id = row.get("prompt_id")
        if prompt_id is None:
            continue
        scores_by_prompt[prompt_id] = row.get("score")

    return {
        "path": str(path),
        "config": payload.get("config") or Path(path).stem,
        "scores": scores_by_prompt,
    }


def average_ranks(values):
    """Return average ranks (1-based) with tie handling."""
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and order[end + 1][1] == order[start][1]:
            end += 1

        avg_rank = (start + end + 2) / 2.0
        for idx in range(start, end + 1):
            ranks[order[idx][0]] = avg_rank
        start = end + 1

    return ranks


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]

    denom_x = sum(x * x for x in centered_x) ** 0.5
    denom_y = sum(y * y for y in centered_y) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None

    numer = sum(x * y for x, y in zip(centered_x, centered_y))
    return numer / (denom_x * denom_y)


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def compare(reference_eval, candidate_eval, threshold, max_examples):
    shared_prompt_ids = sorted(
        set(reference_eval["scores"]) & set(candidate_eval["scores"])
    )
    reference_scores = [reference_eval["scores"][pid] for pid in shared_prompt_ids]
    candidate_scores = [candidate_eval["scores"][pid] for pid in shared_prompt_ids]

    disagreements = []
    abs_diffs = []
    for prompt_id, reference_score, candidate_score in zip(
        shared_prompt_ids, reference_scores, candidate_scores
    ):
        abs_diff = abs(reference_score - candidate_score)
        abs_diffs.append(abs_diff)
        if abs_diff > threshold:
            disagreements.append(
                {
                    "prompt_id": prompt_id,
                    "reference_score": reference_score,
                    "candidate_score": candidate_score,
                    "abs_diff": abs_diff,
                }
            )

    disagreements.sort(key=lambda row: row["abs_diff"], reverse=True)

    return {
        "reference_config": reference_eval["config"],
        "candidate_config": candidate_eval["config"],
        "reference_path": reference_eval["path"],
        "candidate_path": candidate_eval["path"],
        "n_shared_examples": len(shared_prompt_ids),
        "spearman_rho": spearman(reference_scores, candidate_scores),
        "mean_abs_diff": (
            sum(abs_diffs) / len(abs_diffs) if abs_diffs else None
        ),
        "disagreement_threshold": threshold,
        "disagreements_above_threshold": len(disagreements),
        "largest_disagreements": disagreements[:max_examples],
    }


def print_summary(summary):
    rho = summary["spearman_rho"]
    rho_str = f"{rho:.4f}" if rho is not None else "n/a"
    mad = summary["mean_abs_diff"]
    mad_str = f"{mad:.4f}" if mad is not None else "n/a"

    print(
        f"{summary['reference_config']} vs {summary['candidate_config']}: "
        f"n={summary['n_shared_examples']}  "
        f"spearman={rho_str}  "
        f"mean_abs_diff={mad_str}  "
        f"disagreements>{summary['disagreement_threshold']:.2f}="
        f"{summary['disagreements_above_threshold']}"
    )
    for row in summary["largest_disagreements"]:
        print(
            "  "
            f"{row['prompt_id']}: "
            f"{row['reference_score']:.3f} vs {row['candidate_score']:.3f} "
            f"(diff={row['abs_diff']:.3f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-jsons", nargs="+", required=True)
    parser.add_argument("--candidate-jsons", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if len(args.reference_jsons) != len(args.candidate_jsons):
        raise SystemExit(
            "--reference-jsons and --candidate-jsons must have the same length"
        )

    comparisons = []
    for reference_path, candidate_path in zip(
        args.reference_jsons,
        args.candidate_jsons,
    ):
        summary = compare(
            load_eval(reference_path),
            load_eval(candidate_path),
            args.threshold,
            args.max_examples,
        )
        comparisons.append(summary)
        print_summary(summary)

    payload = {"comparisons": comparisons}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
