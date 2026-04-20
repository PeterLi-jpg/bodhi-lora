"""Compare per-criterion rubric outcomes between two eval JSON files."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_eval(path):
    with open(path) as handle:
        return json.load(handle)


def axis_tags(criteria_result):
    tags = criteria_result.get("tags", [])
    axes = [tag.split(":", 1)[1] for tag in tags if tag.startswith("axis:")]
    return axes or ["unlabeled"]


def index_results(payload):
    indexed = {}
    for result in payload.get("results", []):
        criteria = {}
        for item in result.get("criteria_results", []):
            criteria[item["criterion"]] = item
        indexed[result["prompt_id"]] = {
            "response": result.get("response", ""),
            "criteria": criteria,
        }
    return indexed


def compare(base_payload, candidate_payload):
    base_index = index_results(base_payload)
    candidate_index = index_results(candidate_payload)

    flips_up = Counter()
    flips_down = Counter()
    axis_summary = defaultdict(lambda: {"improved": 0, "regressed": 0})
    examples = defaultdict(lambda: {"improved": [], "regressed": []})

    for prompt_id, base_result in base_index.items():
        candidate_result = candidate_index.get(prompt_id)
        if candidate_result is None:
            continue

        for criterion, base_item in base_result["criteria"].items():
            candidate_item = candidate_result["criteria"].get(criterion)
            if candidate_item is None:
                continue

            base_met = bool(base_item.get("criteria_met"))
            candidate_met = bool(candidate_item.get("criteria_met"))
            if base_met == candidate_met:
                continue

            bucket = "improved" if candidate_met and not base_met else "regressed"
            if bucket == "improved":
                flips_up[criterion] += 1
            else:
                flips_down[criterion] += 1

            record = {
                "prompt_id": prompt_id,
                "base_response": base_result["response"],
                "candidate_response": candidate_result["response"],
            }
            current_examples = examples[criterion][bucket]
            if len(current_examples) < 5:
                current_examples.append(record)

            for axis in axis_tags(candidate_item):
                axis_summary[axis][bucket] += 1

    return {
        "improved": flips_up,
        "regressed": flips_down,
        "axis_summary": dict(axis_summary),
        "examples": dict(examples),
    }


def sorted_counter(counter):
    return [
        {"criterion": criterion, "count": count}
        for criterion, count in counter.most_common()
    ]


def print_table(label, rows):
    print(label)
    if not rows:
        print("  none")
        return
    for row in rows[:10]:
        print(f"  {row['count']:>4}  {row['criterion']}")


def print_axis_summary(axis_summary):
    print("\nAxis summary")
    if not axis_summary:
        print("  none")
        return
    for axis in sorted(axis_summary):
        stats = axis_summary[axis]
        print(
            f"  {axis}: improved={stats['improved']} regressed={stats['regressed']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_eval")
    parser.add_argument("candidate_eval")
    parser.add_argument("--output", default="eval/rubric_diff.json")
    args = parser.parse_args()

    comparison = compare(load_eval(args.base_eval), load_eval(args.candidate_eval))
    improved_rows = sorted_counter(comparison["improved"])
    regressed_rows = sorted_counter(comparison["regressed"])

    print_table("Improved criteria", improved_rows)
    print()
    print_table("Regressed criteria", regressed_rows)
    print_axis_summary(comparison["axis_summary"])

    payload = {
        "base_eval": args.base_eval,
        "candidate_eval": args.candidate_eval,
        "improved": improved_rows,
        "regressed": regressed_rows,
        "axis_summary": comparison["axis_summary"],
        "examples": comparison["examples"],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
