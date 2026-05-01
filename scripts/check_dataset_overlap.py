"""Verify HealthBench splits have no data leakage before training or eval.

Checks (in order):

  1. prompt_id disjointness between HealthBench Full and Hard.
  2. The fixed 200-prompt eval holdout lives inside Hard (sanity check).
  3. Whether any eval IDs appear in Full (relevant for generalization runs).
  4. [--train-jsonl] Whether any eval IDs leaked into the SFT training file.
  5. [--seeds N]  Per-seed leakage: samples N independent 200-prompt subsets
     from the 1K Hard set (using seeds 0..N-1) and checks each against the
     training file.  This is the eval protocol for the main result — 5 draws
     of 200 from 1K Hard, each excluded from its own training pool.
  6. [--tag-overlap] Theme/tag distribution comparison between the training
     pool and each eval draw.  Flags any tag that appears >2x more in train
     than eval or vice versa.

Run (basic):
    python scripts/check_dataset_overlap.py

Run (full pre-training check):
    python scripts/check_dataset_overlap.py \
        --train-jsonl data/sft/train.jsonl \
        --seeds 5 \
        --tag-overlap
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_prompt_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.add(obj["prompt_id"])
    return ids


def load_hard_with_tags(path):
    """Return list of {prompt_id, tags} dicts from a HealthBench JSONL.

    Accepts both upstream HealthBench rows (``example_tags``) and our trace /
    train rows (``tags``, copied through by ``generate_traces.py``). Without
    this fallback, ``--tag-overlap`` against ``data/sft/train.jsonl`` or
    ``raw_traces.jsonl`` silently reports every tag as missing on the train
    side and the imbalance check is meaningless.
    """
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "prompt_id": obj["prompt_id"],
                "tags": obj.get("example_tags", obj.get("tags", [])),
            })
    return rows


def load_eval_ids(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    return set(data)


def tag_distribution(rows):
    c = Counter()
    for r in rows:
        for t in r["tags"]:
            c[t] += 1
    return c


def report_tag_overlap(train_rows, eval_rows, label="eval draw"):
    train_tags = tag_distribution(train_rows)
    eval_tags = tag_distribution(eval_rows)
    all_tags = set(train_tags) | set(eval_tags)
    n_train = len(train_rows)
    n_eval = len(eval_rows)
    flagged = []
    for tag in sorted(all_tags):
        tr = train_tags.get(tag, 0) / n_train if n_train else 0
        ev = eval_tags.get(tag, 0) / n_eval if n_eval else 0
        if ev > 0 and tr / ev > 2.0:
            flagged.append((tag, tr, ev, "train-heavy"))
        elif tr > 0 and ev / tr > 2.0:
            flagged.append((tag, tr, ev, "eval-heavy"))
    if flagged:
        print(f"  Tag imbalance in {label} (ratio >2x):")
        for tag, tr, ev, direction in flagged:
            print(f"    {tag}: train={tr:.2%} eval={ev:.2%} [{direction}]")
    else:
        print(f"  Tag distribution balanced in {label} (no tag >2x skew)")
    return flagged


def draw_seed_subset(hard_rows, n, seed):
    rng = random.Random(seed)
    return rng.sample(hard_rows, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthbench",
                        default="data/raw/healthbench.jsonl")
    parser.add_argument("--healthbench-hard",
                        default="data/raw/healthbench_hard.jsonl")
    parser.add_argument("--eval-ids",
                        default="data/raw/hard_200_sample_ids.json",
                        help="fixed 200-prompt holdout (legacy single-seed path)")
    parser.add_argument("--train-jsonl", default=None,
                        help="data/sft/train.jsonl — check for leakage of eval "
                             "IDs into the training file")
    parser.add_argument("--seeds", type=int, default=0,
                        help="number of independent 200-sample draws to check "
                             "(0 = skip per-seed checks, use --eval-ids only). "
                             "Set to 5 for the main eval protocol.")
    parser.add_argument("--draw-size", type=int, default=200,
                        help="samples per seed draw (default 200)")
    parser.add_argument("--tag-overlap", action="store_true",
                        help="report tag/theme distribution imbalance between "
                             "training pool and each eval draw")
    args = parser.parse_args()

    for p in (args.healthbench, args.healthbench_hard, args.eval_ids):
        if not Path(p).exists():
            raise SystemExit(f"missing file: {p}  (run scripts/download_data.py first)")

    full_ids = load_prompt_ids(args.healthbench)
    hard_rows = load_hard_with_tags(args.healthbench_hard)
    hard_ids = {r["prompt_id"] for r in hard_rows}
    eval_ids = load_eval_ids(args.eval_ids)

    train_ids = set()
    train_rows = []
    if args.train_jsonl and Path(args.train_jsonl).exists():
        train_rows = load_hard_with_tags(args.train_jsonl)
        train_ids = {r["prompt_id"] for r in train_rows}
        print(f"Training file:        {len(train_ids):>6} prompts  ({args.train_jsonl})")
    elif args.train_jsonl:
        print(f"Training file:        not found ({args.train_jsonl}) — skipping leakage check")

    full_hard_overlap = full_ids & hard_ids
    hard_is_subset = full_hard_overlap == hard_ids

    print(f"HealthBench Full:     {len(full_ids):>6} prompts")
    print(f"HealthBench Hard:     {len(hard_ids):>6} prompts")
    print(f"Fixed eval holdout:   {len(eval_ids):>6} prompts")
    print()

    # --- Check 1: Full vs Hard relationship ---
    if hard_is_subset:
        print("full >= hard:         Hard is a SUBSET of Full — generalization runs "
              "must pass --exclude-ids data/raw/healthbench_hard.jsonl")
    elif full_hard_overlap:
        print(f"full & hard:          {len(full_hard_overlap):>6} overlap "
              "(not a clean subset — inspect before running)")
    else:
        print("full & hard:          disjoint")

    # --- Check 2: Fixed eval holdout sanity ---
    eval_in_hard = eval_ids & hard_ids
    if eval_in_hard == eval_ids:
        print("eval in hard:         all 200 fixed eval IDs are in Hard (expected)")
    else:
        print(f"eval in hard:         WARNING — only {len(eval_in_hard)}/{len(eval_ids)} "
              "fixed eval IDs are in Hard")

    eval_in_full = eval_ids & full_ids
    if eval_in_full:
        print(f"eval in full:         {len(eval_in_full)} fixed eval IDs appear in Full "
              "— generalization runs must exclude these")
    else:
        print("eval in full:         no fixed eval IDs in Full (generalization run safe)")

    # --- Check 3: Fixed eval leakage into training file ---
    if train_ids:
        leaked = eval_ids & train_ids
        if leaked:
            print(f"\nDATA LEAKAGE: {len(leaked)} fixed eval IDs found in training file!")
            for pid in sorted(leaked):
                print(f"  {pid}")
            raise SystemExit("Abort — training data is contaminated with eval IDs. "
                             "Re-run generate_traces.py with --exclude-ids.")
        else:
            print(f"leakage (fixed eval): clean — 0/{len(eval_ids)} eval IDs in training file")

    # --- Check 3b: Full HealthBench Hard exclusion ---
    # Post-issue-#60: training pool excludes *all 1000* Hard prompts so any
    # bootstrap draw from Hard (any seed) is automatically held out. Catch
    # any Hard prompt in train, not just the fixed 200.
    if train_ids:
        hard_leak = hard_ids & train_ids
        if hard_leak:
            sample = sorted(hard_leak)[:5]
            print(f"\nDATA LEAKAGE: {len(hard_leak)} HealthBench Hard prompts in "
                  f"training file (post-#60 invariant: all 1K Hard must be excluded).")
            print(f"  first {len(sample)}: {sample}")
            raise SystemExit("Abort — training data contains HealthBench Hard. "
                             "Pass --exclude-ids data/raw/healthbench_hard.jsonl to "
                             "generate_traces.py + filter_traces.py and re-run.")
        else:
            print(f"leakage (full Hard): clean — 0/{len(hard_ids)} Hard prompts "
                  f"in training file")

    errors = []

    # --- Check 4: Per-seed draw leakage ---
    if args.seeds > 0:
        print(f"\n--- Per-seed eval draws ({args.seeds} seeds x {args.draw_size} samples "
              f"from {len(hard_ids)} Hard examples) ---")
        for seed in range(args.seeds):
            draw = draw_seed_subset(hard_rows, args.draw_size, seed)
            draw_ids = {r["prompt_id"] for r in draw}

            if train_ids:
                leaked = draw_ids & train_ids
                if leaked:
                    msg = (f"seed {seed}: DATA LEAKAGE — {len(leaked)} eval IDs "
                           f"in training file")
                    print(f"  {msg}")
                    errors.append(msg)
                else:
                    print(f"  seed {seed}: clean — 0/{args.draw_size} eval IDs "
                          f"in training file")
            else:
                print(f"  seed {seed}: {args.draw_size} IDs drawn (no training file "
                      f"to check against)")

            if args.tag_overlap and train_rows:
                report_tag_overlap(train_rows, draw, label=f"seed {seed} draw")

        # Verify all 5 draws are disjoint from each other (sanity check on the
        # sampling — not strictly required, overlap between draws is fine since
        # each draw has its own independently excluded training pool).
        all_draws = [
            {r["prompt_id"] for r in draw_seed_subset(hard_rows, args.draw_size, s)}
            for s in range(args.seeds)
        ]
        pairwise_overlaps = []
        for i in range(args.seeds):
            for j in range(i + 1, args.seeds):
                ov = len(all_draws[i] & all_draws[j])
                if ov > 0:
                    pairwise_overlaps.append((i, j, ov))
        if pairwise_overlaps:
            print(f"\nNote: seed draws overlap each other (expected for random subsets "
                  f"of 1K Hard):")
            for i, j, ov in pairwise_overlaps:
                print(f"  seed {i} & seed {j}: {ov} shared IDs "
                      f"({ov/args.draw_size:.0%} of draw size)")
            print("  This is fine — each seed uses its own --exclude-ids file.")
        else:
            print(f"\n  All {args.seeds} seed draws are fully disjoint.")

    if errors:
        print(f"\n{len(errors)} leakage error(s) found. Abort before training.")
        raise SystemExit("\n".join(errors))

    print("\noverlap check passed")


if __name__ == "__main__":
    main()
