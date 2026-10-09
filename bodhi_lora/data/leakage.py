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
    python -m bodhi_lora.data.leakage

Run (full pre-training check):
    python -m bodhi_lora.data.leakage \
        --train-jsonl data/sft/train.jsonl \
        --seeds 5 \
        --tag-overlap

Other benchmarks (--benchmark-jsonl): the check takes the held-out pool
(--holdout-ids) and fails if any training prompt_id is in it, or if any
training prompt has the same text as a held-out prompt after lowercasing and
collapsing whitespace (a near-duplicate under a different id). With --eval-ids
it also checks that each per-seed eval file lies inside the pool.

    python -m bodhi_lora.data.leakage --benchmark-jsonl data/raw/chatdoctor_s0.jsonl \
        --holdout-ids data/raw/chatdoctor_s0_holdout_ids.json \
        --train-jsonl <cell>/sft/train.jsonl --eval-ids <cell>/eval_ids/seed_*.json \
        --json-out <cell>/overlap_check.json

The HealthBench mode runs the same text check against all of HealthBench Hard.
--json-out writes the result, pass or fail, so a run keeps a record of it.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def normalize_text(text):
    """Lowercase and collapse whitespace, as in contamination_probe.py."""
    return " ".join(str(text).lower().split())


def last_user_text(messages):
    """The last user turn of a prompt, or '' when there is none."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content
    return ""


def load_prompt_texts(path, key):
    """{prompt_id: normalized last user turn} from a JSONL file.

    key is "prompt" for benchmark rows and "messages" for trace rows.
    """
    texts = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"check_dataset_overlap: malformed JSON at {path}:{lineno}") from e
            texts[obj["prompt_id"]] = normalize_text(last_user_text(obj.get(key)))
    return texts


def text_duplicates(train_texts, held_texts):
    """Training prompt_ids whose text equals a held-out prompt's text."""
    held = {t for t in held_texts.values() if t}
    return sorted(pid for pid, t in train_texts.items() if t and t in held)


def write_report(path, report):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"overlap report -> {path}")


def fail(report, json_out, message):
    """Record a failed check, then stop the run."""
    report["passed"] = False
    report.setdefault("errors", []).append(message)
    write_report(json_out, report)
    raise SystemExit(message)


def check_benchmark(args):
    """Leakage gate for a non-HealthBench cell."""
    for p in (args.benchmark_jsonl, args.holdout_ids, args.train_jsonl):
        if not p or not Path(p).exists():
            raise SystemExit(f"missing file: {p}  (benchmark mode needs "
                             "--benchmark-jsonl, --holdout-ids and --train-jsonl)")
    held_ids = load_eval_ids(args.holdout_ids)
    bench_texts = load_prompt_texts(args.benchmark_jsonl, "prompt")
    train_texts = load_prompt_texts(args.train_jsonl, "messages")
    held_texts = {pid: bench_texts.get(pid, "") for pid in held_ids}
    report = {
        "mode": "benchmark",
        "benchmark_jsonl": args.benchmark_jsonl,
        "holdout_ids": args.holdout_ids,
        "train_jsonl": args.train_jsonl,
        "n_holdout": len(held_ids),
        "n_holdout_missing_from_benchmark": len(held_ids - set(bench_texts)),
        "n_train": len(train_texts),
    }
    id_leak = sorted(held_ids & set(train_texts))
    dup = text_duplicates(train_texts, held_texts)
    report.update({"id_leak": len(id_leak), "id_leak_first": id_leak[:5],
                   "text_duplicates": len(dup), "text_duplicates_first": dup[:5]})
    print(f"holdout pool: {len(held_ids)}  train: {len(train_texts)}")
    print(f"id leakage: {len(id_leak)}  text duplicates: {len(dup)}")
    report["eval_files"] = []
    for path in args.eval_ids_files or []:
        ids = load_eval_ids(path)
        outside = sorted(ids - held_ids)
        report["eval_files"].append({"path": path, "n": len(ids), "outside_pool": len(outside)})
        if outside:
            fail(report, args.json_out, f"{path}: {len(outside)} eval ids are not in the holdout pool")
    if report["n_holdout_missing_from_benchmark"]:
        fail(report, args.json_out, "holdout ids missing from --benchmark-jsonl "
             f"({report['n_holdout_missing_from_benchmark']}); wrong benchmark file?")
    if id_leak:
        fail(report, args.json_out, f"DATA LEAKAGE: {len(id_leak)} holdout prompt_ids in training file")
    if dup:
        fail(report, args.json_out, f"DATA LEAKAGE: {len(dup)} training prompts repeat a holdout prompt's text")
    report["passed"] = True
    write_report(args.json_out, report)
    print("\noverlap check passed")


def load_prompt_ids(path):
    ids = set()
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            # mirror load_hard_with_tags: skip blank lines so a stray trailing
            # newline doesn't trip the JSON parser.
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"check_dataset_overlap: malformed JSON at {path}:{lineno}"
                ) from e
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
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"check_dataset_overlap: malformed JSON at {path}:{lineno}"
                ) from e
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
                        default="data/raw/healthbench.jsonl",
                        help="HealthBench Full prompts JSONL (one row per prompt).")
    parser.add_argument("--healthbench-hard",
                        default="data/raw/healthbench_hard.jsonl",
                        help="HealthBench Hard prompts JSONL (one row per prompt).")
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
    parser.add_argument("--benchmark-jsonl", default=None,
                        help="benchmark mode: the benchmark rows (prompt_id, prompt)")
    parser.add_argument("--holdout-ids", default=None,
                        help="benchmark mode: the held-out pool of prompt_ids")
    parser.add_argument("--eval-ids-files", nargs="*", default=None,
                        help="benchmark mode: per-seed eval id files that must lie "
                             "inside the pool")
    parser.add_argument("--json-out", default=None,
                        help="write the check result (pass or fail) to this JSON file")
    args = parser.parse_args()

    if args.benchmark_jsonl:
        check_benchmark(args)
        return

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
    report = {"mode": "healthbench", "train_jsonl": args.train_jsonl,
              "n_full": len(full_ids), "n_hard": len(hard_ids),
              "n_fixed_eval": len(eval_ids), "n_train": len(train_ids)}

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
            report["fixed_eval_leak"] = len(leaked)
            fail(report, args.json_out,
                 "Abort — training data is contaminated with eval IDs. "
                 "Re-run generate_traces.py with --exclude-ids.")
        else:
            print(f"leakage (fixed eval): clean — 0/{len(eval_ids)} eval IDs in training file")

    # --- Check 3b: Full HealthBench Hard exclusion ---
    # Training pool excludes all 1K HealthBench Hard prompts so per-seed
    # bootstrap eval is honestly held-out (see RESULTS.md). Catch any Hard
    # prompt in train, not just the fixed 200.
    if train_ids:
        hard_leak = hard_ids & train_ids
        if hard_leak:
            sample = sorted(hard_leak)[:5]
            print(f"\nDATA LEAKAGE: {len(hard_leak)} HealthBench Hard prompts in "
                  f"training file (training pool must exclude all 1K Hard).")
            print(f"  first {len(sample)}: {sample}")
            report["hard_id_leak"] = len(hard_leak)
            fail(report, args.json_out,
                 "Abort — training data contains HealthBench Hard. "
                 "Pass --exclude-ids data/raw/healthbench_hard.jsonl to "
                 "generate_traces.py + filter_traces.py and re-run.")
        else:
            print(f"leakage (full Hard): clean — 0/{len(hard_ids)} Hard prompts "
                  f"in training file")
        report["fixed_eval_leak"] = 0
        report["hard_id_leak"] = 0

        # Same prompt text under another id (a near-duplicate) also leaks.
        dup = text_duplicates(load_prompt_texts(args.train_jsonl, "messages"),
                              load_prompt_texts(args.healthbench_hard, "prompt"))
        report["text_duplicates"] = len(dup)
        report["text_duplicates_first"] = dup[:5]
        if dup:
            fail(report, args.json_out,
                 f"DATA LEAKAGE: {len(dup)} training prompts repeat a Hard prompt's text")
        print("leakage (Hard text):  clean, 0 training prompts repeat a Hard prompt's text")

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
        fail(report, args.json_out, "\n".join(errors))

    report["passed"] = True
    write_report(args.json_out, report)
    print("\noverlap check passed")


if __name__ == "__main__":
    main()
