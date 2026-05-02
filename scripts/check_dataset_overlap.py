"""Verify HealthBench splits don't overlap before running the HealthBench-only
generalization experiment.

Training on HealthBench alone (no HealthBench Hard in the training data) still
helps on the HealthBench Hard holdout. That claim only makes sense if the two
splits have disjoint prompt_ids. If the same prompt appears in both, "train on
HealthBench only" accidentally leaks Hard prompts into training.

Answers:
  1. Are HealthBench and HealthBench Hard disjoint? (expected: yes)
  2. Are the 200 eval IDs in HealthBench Hard? (expected: yes, they're drawn from it)
  3. Are any eval IDs in HealthBench full? (expected: no, relevant to the
     generalization experiment.)
  4. (--tag-overlap) Are any rubric themes over-represented in train vs eval?
  5. (--regen-holdout) Emit a new theme-stratified eval holdout from Hard."""

import argparse
import json
import math
import random as _random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Tags classified as safety-critical for HealthBench.
# An "S" marker appears in the overlap table; these tags are highlighted
# when they carry a per-theme score imbalance or high contamination risk.
# Override the default set via --safety-themes tag1 tag2 ...
SAFETY_THEMES: frozenset[str] = frozenset({
    "theme:emergency_referrals",
    "theme:hedging",
    "theme:medication_safety",
    "theme:palliative_care_euthanasia",
    "theme:mental_health",
    "theme:harmful_information",
    "theme:diagnostic_errors",
    "theme:patient_safety",
    "theme:contraindications",
    "theme:suicide_self_harm",
})


def load_prompt_ids(path):
    """Return a set of prompt_ids from a HealthBench JSONL file."""
    ids = set()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.add(obj["prompt_id"])
    return ids


def load_eval_ids(path):
    """Load the fixed eval holdout. Accepts both list and dict schema."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    return set(data)


def _tag_counts_from_jsonl(path, tag_field):
    """Return (Counter of tag -> count, total example count).

    Warns when more than 10 % of records are missing the tag field — silent
    zero-counts would make the overlap table misleading.
    """
    counts = Counter()
    total = 0
    missing = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            tags = obj.get(tag_field)
            if tags is None:
                missing += 1
            else:
                for tag in tags:
                    counts[tag] += 1
            total += 1
    if total > 0 and missing / total > 0.1:
        print(
            f"  WARNING: {missing}/{total} records in {path} are missing "
            f"'{tag_field}' — tag counts may be incomplete"
        )
    return counts, total


def load_eval_results(path: str) -> dict[str, float]:
    """Load eval JSON produced by eval_healthbench.py → {prompt_id: score}."""
    with open(path) as f:
        data = json.load(f)
    return {r["prompt_id"]: r["score"] for r in data["results"]}


def load_eval_tag_scores(path: str) -> dict[str, list[float]]:
    """Load per-tag scores from eval JSON → {tag: [per-example scores]}.

    Reads results[*].tag_scores (per-criterion tag breakdown from the grader).
    More precise than assigning the overall example score to every tag the
    example happens to carry, because different rubric criteria under the same
    example can score differently per theme.
    """
    with open(path) as f:
        data = json.load(f)
    tag_score_map: dict[str, list[float]] = defaultdict(list)
    for r in data.get("results", []):
        for tag, score in r.get("tag_scores", {}).items():
            if score is not None and not math.isnan(score):
                tag_score_map[tag].append(score)
    return dict(tag_score_map)


def per_theme_score_report(
    tag_score_map: dict[str, list[float]],
    flagged_tags: list[str],
    safety_themes: frozenset[str],
    tag_ratios: dict[str, float] | None = None,
) -> None:
    """Print per-theme mean scores with optional contamination-risk column.

    tag_score_map : {tag: [scores]} from load_eval_tag_scores.
    flagged_tags  : tags with >2x frequency skew from tag_overlap_report.
    tag_ratios    : {tag: train%/eval%}; when provided, adds a risk column
                    (ratio × mean_score) to surface potential rubric memorisation.
    """
    if not tag_score_map:
        print("  (no tag scores found in eval results)")
        return

    flagged_set = set(flagged_tags)
    show_risk = tag_ratios is not None
    col = 52

    if show_risk:
        header = (
            f"  {'tag':<{col}} {'n':>5} {'mean':>7} {'std':>6}  "
            f"{'risk':>6}  {'warn':<5} safety"
        )
    else:
        header = (
            f"  {'tag':<{col}} {'n':>5} {'mean':>7} {'std':>6}  {'warn':<5} safety"
        )
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Flagged+safety tags first, then flagged-only, then safety-only, then rest
    def _sort_key(tag):
        in_f = tag in flagged_set
        in_s = tag in safety_themes
        return (not (in_f and in_s), not in_f, not in_s, tag)

    high_risk: list[tuple[str, float]] = []

    for tag in sorted(tag_score_map, key=_sort_key):
        s = tag_score_map[tag]
        mean_s = sum(s) / len(s)
        std_s = statistics.stdev(s) if len(s) > 1 else 0.0
        warn = "WARN" if tag in flagged_set else ""
        safe = "S" if tag in safety_themes else ""

        if show_risk:
            ratio = tag_ratios.get(tag)
            if ratio is None:
                risk_str = "     ?"
            elif math.isinf(ratio):
                risk_str = "   inf"
                if tag in flagged_set and mean_s > 0.5:
                    high_risk.append((tag, float("inf")))
            else:
                risk_val = ratio * mean_s
                risk_str = f"{risk_val:6.2f}"
                if tag in flagged_set and mean_s > 0.5:
                    high_risk.append((tag, risk_val))
            print(
                f"  {tag:<{col}} {len(s):>5} {mean_s:>7.3f} {std_s:>6.3f}  "
                f"{risk_str}  {warn:<5} {safe}"
            )
        else:
            print(
                f"  {tag:<{col}} {len(s):>5} {mean_s:>7.3f} {std_s:>6.3f}  "
                f"{warn:<5} {safe}"
            )

    if high_risk:
        high_risk.sort(key=lambda x: -x[1])
        print()
        print(
            "  MEMORIZATION RISK — over-represented in train AND scoring high in eval:"
        )
        for tag, risk in high_risk:
            safe_marker = "  [SAFETY]" if tag in safety_themes else ""
            risk_label = "inf" if math.isinf(risk) else f"{risk:.2f}"
            print(f"    {tag}  risk={risk_label}{safe_marker}")


def tag_overlap_report(
    train_path: str,
    hard_path: str,
    eval_ids: set,
    safety_themes: frozenset[str] = SAFETY_THEMES,
    eval_results_path: str | None = None,
) -> None:
    """Print a tag-frequency table and flag tags with >2x skew between splits."""
    if not Path(train_path).exists():
        raise SystemExit(
            f"missing file: {train_path}  (run scripts/generate_traces.py first)"
        )
    if not Path(hard_path).exists():
        raise SystemExit(
            f"missing file: {hard_path}  (run scripts/download_data.py first)"
        )

    # Train JSONL (generate_traces.py output) stores tags under "tags"
    train_counts, train_n = _tag_counts_from_jsonl(train_path, "tags")

    # Eval split: Hard examples whose prompt_id is in the holdout
    eval_counts: Counter = Counter()
    eval_n = 0
    with open(hard_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("prompt_id") in eval_ids:
                for tag in obj.get("example_tags", []):
                    eval_counts[tag] += 1
                eval_n += 1

    if train_n == 0:
        raise SystemExit(f"{train_path} is empty")
    if eval_n == 0:
        raise SystemExit(
            "no eval examples found in HealthBench Hard — "
            "check that --eval-ids and --healthbench-hard point to the right files"
        )

    all_tags = sorted(set(train_counts) | set(eval_counts))

    col = 52
    header = (
        f"{'tag':<{col}} {'train_n':>7} {'eval_n':>7} "
        f"{'train%':>7} {'eval%':>7} {'ratio':>7}  {'flag':<5} safety"
    )
    print(header)
    print("-" * len(header))

    flagged: list[str] = []
    tag_ratios: dict[str, float] = {}

    for tag in all_tags:
        tc = train_counts.get(tag, 0)
        ec = eval_counts.get(tag, 0)
        t_pct = tc / train_n * 100
        e_pct = ec / eval_n * 100
        safe = "S" if tag in safety_themes else ""

        if t_pct == 0 and e_pct == 0:
            ratio_str = "      -"
            flag = ""
            tag_ratios[tag] = 1.0
        elif e_pct == 0:
            ratio_str = "    inf"
            flag = "WARN"
            flagged.append(tag)
            tag_ratios[tag] = float("inf")
        elif t_pct == 0:
            ratio_str = "   0.00"
            flag = "WARN"
            flagged.append(tag)
            tag_ratios[tag] = 0.0
        else:
            ratio = t_pct / e_pct
            ratio_str = f"{ratio:7.2f}"
            tag_ratios[tag] = ratio
            if ratio > 2.0 or ratio < 0.5:
                flag = "WARN"
                flagged.append(tag)
            else:
                flag = ""

        print(
            f"{tag:<{col}} {tc:>7} {ec:>7} "
            f"{t_pct:>6.1f}% {e_pct:>6.1f}% {ratio_str}  {flag:<5} {safe}"
        )

    print()
    print(f"train examples: {train_n}   eval examples: {eval_n}")
    print()

    if flagged:
        print(f"WARN: {len(flagged)} tag(s) with >2x frequency skew (train% / eval%):")
        for t in flagged:
            safe_marker = "  [SAFETY]" if t in safety_themes else ""
            print(f"  {t}{safe_marker}")
        print()

        if eval_results_path is None:
            print(
                "  Per-theme scores not shown: rerun with "
                "--eval-results <path/to/eval.json>"
            )
        elif not Path(eval_results_path).exists():
            print(
                f"  Per-theme scores not shown: eval file not found: {eval_results_path}"
            )
        else:
            tag_score_map = load_eval_tag_scores(eval_results_path)
            if not tag_score_map:
                print(
                    "  Per-theme scores not shown: no tag_scores field in eval results — "
                    "re-run eval_healthbench.py with a version that writes tag_scores "
                    "per result"
                )
            else:
                print(
                    f"PER-THEME SCORES (from {eval_results_path}) — "
                    "risk = train_ratio × mean_eval_score; "
                    "high risk on a flagged safety tag signals rubric memorisation:"
                )
                per_theme_score_report(
                    tag_score_map,
                    flagged,
                    safety_themes,
                    tag_ratios=tag_ratios,
                )
    else:
        print("tag overlap OK: no tag exceeds 2x frequency difference between splits")


def stratify_eval_holdout(
    hard_path: str,
    n: int = 200,
    seed: int = 42,
    exclude_ids: set | None = None,
) -> set[str]:
    """Sample n prompt_ids from HealthBench Hard, stratified by primary theme.

    Each example is assigned to its first 'theme:' tag; examples with no theme
    tag go into an '__untagged__' bucket.  The Largest Remainder Method ensures
    the returned set contains exactly n entries and that each theme's share is
    as close to proportional as possible.
    """
    rng = _random.Random(seed)
    exclude_ids = exclude_ids or set()

    by_theme: dict[str, list[str]] = defaultdict(list)
    with open(hard_path) as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("prompt_id")
            if not pid or pid in exclude_ids:
                continue
            themes = [t for t in obj.get("example_tags", []) if t.startswith("theme:")]
            key = themes[0] if themes else "__untagged__"
            by_theme[key].append(pid)

    total_available = sum(len(v) for v in by_theme.values())
    if total_available < n:
        raise SystemExit(
            f"only {total_available} examples available after exclusions, need {n}"
        )

    # Largest Remainder Method: proportional allocation summing to exactly n
    raw = {t: len(pids) / total_available * n for t, pids in by_theme.items()}
    floors = {t: math.floor(v) for t, v in raw.items()}
    remainder = n - sum(floors.values())
    by_remainder = sorted(raw, key=lambda t: -(raw[t] - floors[t]))
    for t in by_remainder[:remainder]:
        floors[t] += 1

    selected: list[str] = []
    for theme, k in floors.items():
        pids = by_theme[theme][:]
        rng.shuffle(pids)
        selected.extend(pids[:k])

    return set(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Verify HealthBench train/eval splits are not theme-contaminated."
    )
    parser.add_argument("--healthbench",
                        default="data/raw/healthbench.jsonl",
                        help="path to HealthBench full JSONL")
    parser.add_argument("--healthbench-hard",
                        default="data/raw/healthbench_hard.jsonl",
                        help="path to HealthBench Hard JSONL")
    parser.add_argument("--eval-ids",
                        default="data/raw/hard_200_sample_ids.json",
                        help="path to the eval holdout file")
    parser.add_argument("--train",
                        default="data/sft/train.jsonl",
                        help="path to the SFT train JSONL (for --tag-overlap)")
    parser.add_argument("--tag-overlap", action="store_true",
                        help="compare theme distribution between train and the eval "
                             "holdout; flag any tag with >2x frequency skew")
    parser.add_argument("--eval-results",
                        default=None,
                        metavar="PATH",
                        help="eval JSON from eval_healthbench.py; enables per-theme "
                             "scores and contamination-risk column in --tag-overlap output")
    parser.add_argument("--safety-themes",
                        nargs="*",
                        default=None,
                        metavar="TAG",
                        help="replace the built-in SAFETY_THEMES set; "
                             "e.g. --safety-themes theme:emergency_referrals theme:hedging")
    parser.add_argument("--regen-holdout", action="store_true",
                        help="generate a new theme-stratified eval holdout from "
                             "HealthBench Hard and write it to --holdout-out")
    parser.add_argument("--holdout-out",
                        default=None,
                        metavar="PATH",
                        help="destination for the regenerated holdout JSON "
                             "(default: print to stdout)")
    parser.add_argument("--holdout-n",
                        type=int,
                        default=200,
                        metavar="N",
                        help="number of samples in the regenerated holdout (default: 200)")
    parser.add_argument("--holdout-seed",
                        type=int,
                        default=42,
                        metavar="SEED",
                        help="random seed for stratified sampling (default: 42)")
    args = parser.parse_args()

    safety_themes = (
        frozenset(args.safety_themes) if args.safety_themes is not None else SAFETY_THEMES
    )

    if args.regen_holdout:
        if not Path(args.healthbench_hard).exists():
            raise SystemExit(
                f"missing file: {args.healthbench_hard}  "
                "(run scripts/download_data.py first)"
            )
        new_ids = stratify_eval_holdout(
            args.healthbench_hard,
            n=args.holdout_n,
            seed=args.holdout_seed,
        )
        holdout_obj = {
            "description": (
                f"Theme-stratified eval holdout "
                f"({args.holdout_n} samples, seed={args.holdout_seed})"
            ),
            "total_samples": len(new_ids),
            "random_seed": args.holdout_seed,
            "stratification": "theme (primary example_tag with theme: prefix)",
            "prompt_ids": sorted(new_ids),
        }
        if args.holdout_out:
            Path(args.holdout_out).write_text(
                json.dumps(holdout_obj, indent=2), encoding="utf-8"
            )
            print(f"wrote {len(new_ids)} stratified eval IDs → {args.holdout_out}")
        else:
            print(json.dumps(holdout_obj, indent=2))
        return

    if args.tag_overlap:
        eval_ids = load_eval_ids(args.eval_ids)
        tag_overlap_report(
            args.train,
            args.healthbench_hard,
            eval_ids,
            safety_themes=safety_themes,
            eval_results_path=args.eval_results,
        )
        print()

    for p in (args.healthbench, args.healthbench_hard, args.eval_ids):
        if not Path(p).exists():
            raise SystemExit(f"missing file: {p}  (run scripts/download_data.py first)")

    full = load_prompt_ids(args.healthbench)
    hard = load_prompt_ids(args.healthbench_hard)
    eval_ids = load_eval_ids(args.eval_ids)

    full_hard_overlap = full & hard
    eval_in_hard = eval_ids & hard
    eval_in_full = eval_ids & full
    hard_is_subset = full_hard_overlap == hard

    print(f"HealthBench full:     {len(full):>6} prompts")
    print(f"HealthBench Hard:     {len(hard):>6} prompts")
    print(f"Eval holdout:         {len(eval_ids):>6} prompts")
    print()
    if hard_is_subset:
        print("full >= hard:         Hard is a subset of Full "
              "(generalization run must exclude Hard via --exclude-ids)")
    elif full_hard_overlap:
        print(f"full & hard:          {len(full_hard_overlap):>6} overlap "
              "(not a clean subset; inspect before running anything)")
    else:
        print("full & hard:          disjoint")

    if eval_in_hard == eval_ids:
        print("eval in hard:         all 200 eval IDs live in HealthBench Hard "
              "(expected)")
    else:
        print(f"eval in hard:         {len(eval_in_hard)} of {len(eval_ids)} "
              "eval IDs are in Hard (unexpected)")

    if eval_in_full:
        print(f"eval in full:         {len(eval_in_full)} of {len(eval_ids)} "
              "eval IDs appear in Full (HealthBench-only runs must exclude these)")
    else:
        print("eval in full:         no eval IDs in Full "
              "(HealthBench-only run safe without --exclude-ids)")

    print()
    if hard_is_subset or eval_in_full:
        print("RECOMMENDATION for HealthBench-only generalization runs:")
        print("  --datasets healthbench \\")
        print("  --exclude-ids data/raw/healthbench_hard.jsonl "
              "data/raw/hard_200_sample_ids.json")
        print()
        raise SystemExit(
            "overlap present — Hard and/or eval IDs live inside Full. "
            "Not a bug; but a generalization run that trains on `--datasets "
            "healthbench` alone will leak Hard prompts unless excluded. "
            "See the recommendation above and re-run with --exclude-ids."
        )

    if eval_in_hard != eval_ids:
        missing = eval_ids - eval_in_hard
        print(f"WARN: {len(missing)} eval IDs are not in HealthBench Hard. "
              "The holdout should be drawn from Hard; inspect.")

    print("overlap check passed")


if __name__ == "__main__":
    main()
