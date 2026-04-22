"""Print a compact comparison table across two eval directories."""

import argparse
import json
from pathlib import Path


FILES = ["base_no_wrapper.json", "base_bodhi.json", "lora_no_wrapper.json", "lora_bodhi.json"]


def load_metric(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir")
    parser.add_argument("candidate_dir")
    args = parser.parse_args()

    baseline = Path(args.baseline_dir)
    candidate = Path(args.candidate_dir)

    print(f"{'config':<20} {'baseline':>10} {'candidate':>10} {'delta':>10}")
    for name in FILES:
        base = load_metric(baseline / name)
        cand = load_metric(candidate / name)
        delta = None if base is None or cand is None else cand - base
        label = name.replace(".json", "")
        print(
            f"{label:<20} "
            f"{(f'{base:.4f}' if base is not None else 'n/a'):>10} "
            f"{(f'{cand:.4f}' if cand is not None else 'n/a'):>10} "
            f"{(f'{delta:+.4f}' if delta is not None else 'n/a'):>10}"
        )


if __name__ == "__main__":
    main()
