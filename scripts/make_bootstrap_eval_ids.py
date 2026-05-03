"""Generate a deterministic per-seed eval draw from HealthBench Hard.

For the multi-seed bootstrap eval protocol (issue #60, RESULTS.md §1): each
seed gets its own random 200-prompt subset of the 1000 HealthBench Hard
prompts. The draw is deterministic in the seed (Python's Random + sorted
output), so re-running this script with the same seed produces the same file.

This is the eval-set side of the leakage fix: training already excludes all
1000 Hard via --exclude-ids, so any 200 we draw here are genuinely held-out.

Usage:
    python scripts/make_bootstrap_eval_ids.py \\
        --healthbench-jsonl data/raw/healthbench_hard.jsonl \\
        --seed 42 \\
        --output data/raw/hard_seed_42.json
"""

import argparse
import json
import os
from pathlib import Path
from random import Random


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--healthbench-jsonl",
                   default="data/raw/healthbench_hard.jsonl",
                   help="Source pool to draw from.")
    p.add_argument("--seed", type=int, required=True,
                   help="Per-seed RNG; same seed → same draw.")
    p.add_argument("--size", type=int, default=200,
                   help="Number of prompts to draw (default 200).")
    p.add_argument("--output", required=True,
                   help="Where to write the JSON file of prompt_ids.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output even if it already exists.")
    args = p.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        # Idempotent: launchers call this every run; skip if cached.
        print(f"{out} already exists, skipping (pass --force to regenerate)")
        return

    ids = []
    with open(args.healthbench_jsonl) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"make_bootstrap_eval_ids: malformed JSON at "
                    f"{args.healthbench_jsonl}:{lineno}"
                ) from e
            ids.append(obj["prompt_id"])
    if len(ids) < args.size:
        raise SystemExit(
            f"only {len(ids)} prompts in {args.healthbench_jsonl}, "
            f"need {args.size}")

    # sample without replacement: 200 distinct prompts. sort the result so
    # the output file is order-stable across Python versions / hash seeds.
    rng = Random(args.seed)
    drawn = sorted(rng.sample(ids, args.size))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: launchers run multiple seeds in parallel, and two processes
    # racing on the same output path could otherwise interleave bytes. Write to
    # a sibling .tmp and os.replace it into place (POSIX rename is atomic), so
    # the loser of the race clobbers fully or not at all, never half-written.
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({
            "description": (f"Bootstrap eval draw: {args.size} of "
                            f"{len(ids)} HealthBench Hard prompts, "
                            f"seed={args.seed}"),
            "seed": args.seed,
            "size": args.size,
            "total_pool": len(ids),
            "source": str(args.healthbench_jsonl),
            "prompt_ids": drawn,
        }, f, indent=2)
    os.replace(tmp, out)
    print(f"wrote {len(drawn)} prompt_ids to {out}")


if __name__ == "__main__":
    main()
