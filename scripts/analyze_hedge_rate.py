"""analyze_hedge_rate.py — sanity check whether the BODHI wrapper produces
unconditionally-hedged outputs (issue #64).

If the wrapper hedges on (almost) every prompt regardless of difficulty, the
training signal is uninformative — the LoRA learns to hedge unconditionally
rather than calibrated to uncertainty. This script tests that.

Inputs:
  data/sft/raw_traces.jsonl                # output of generate_traces (no scores)
  -- or --
  data/sft/all_graded.jsonl                # output of filter_traces --graded-output
                                            (preserves full grade distribution)

Output:
  eval/hedge_rate.json with three views:
    * overall hedge rate
    * hedge rate stratified by rubric-score quartile (only with graded input)
    * hedge rate stratified by example_tags / theme

Hedge classifier:
  Lexical regex match on a curated list of hedge phrases (case-insensitive,
  word-boundary anchored to avoid false positives like 'consultation' matching
  'consult'). The list is intentionally conservative — we'd rather under-count
  hedging than inflate it. False negatives only weaken the unconditional-hedging
  hypothesis, false positives would inflate it.

Usage:
  python scripts/analyze_hedge_rate.py \
      --input data/sft/all_graded.jsonl \
      --output eval/hedge_rate.json
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


# Curated list of hedge phrases. See the comment above for the conservative
# under-counting bias. Each entry is treated as a regex; \b adds a word boundary
# so 'consult' doesn't match 'consultation' / 'consulted'.
HEDGE_PATTERNS = [
    r"\bI'?m\s+not\s+(certain|sure)\b",
    r"\bI\s+don'?t\s+know\b",
    r"\bI\s+can(?:not|'?t)\s+(?:diagnose|provide\s+a\s+diagnosis)\b",
    r"\bI\s+am\s+not\s+a\s+(?:doctor|medical\s+professional|physician)\b",
    r"\bnot\s+a\s+substitute\s+for\b",
    r"\bconsult\s+(?:a|your|with\s+a)\s+(?:doctor|physician|healthcare|medical)\b",
    r"\bseek\s+(?:medical\s+attention|professional\s+(?:medical\s+)?advice|emergency)\b",
    r"\bplease\s+(?:see|talk\s+to|reach\s+out\s+to)\s+a\s+(?:doctor|professional)\b",
    r"\bgo\s+to\s+(?:the\s+)?(?:hospital|emergency\s+room|ER|urgent\s+care)\b",
    r"\bcall\s+9-?1-?1\b",
    r"\bunder\s+the\s+supervision\s+of\b",
    r"\bunder\s+a\s+(?:doctor|physician)'?s\s+care\b",
    r"\bdepend(?:s|ing)?\s+on\s+(?:your|the)\s+(?:specific|individual|particular)\b",
    r"\bvar(?:ies|y)\s+(?:from\s+person|by\s+individual|case\s+by\s+case)\b",
    r"\bI\s+would\s+(?:strongly\s+)?recommend\s+(?:speaking|talking)\s+(?:to|with)\b",
]
HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def is_hedged(text):
    """Return True if the response contains any phrase from the hedge list."""
    if not text:
        return False
    return HEDGE_REGEX.search(text) is not None


def load_traces(path):
    """Yield trace dicts from a JSONL file. Accepts both raw_traces and
    all_graded.jsonl formats (graded variant has a .grade subdict)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def quartile_buckets(scores):
    """Return (q1, q2, q3) cutoffs for quartile binning, or None if too few values."""
    finite = [s for s in scores if s is not None]
    if len(finite) < 4:
        return None
    qs = statistics.quantiles(finite, n=4)
    return qs[0], qs[1], qs[2]


def quartile_label(score, q1, q2, q3):
    if score is None:
        return None
    if score <= q1:
        return "Q1 (lowest)"
    if score <= q2:
        return "Q2"
    if score <= q3:
        return "Q3"
    return "Q4 (highest)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="raw_traces.jsonl OR all_graded.jsonl")
    parser.add_argument("--output", default="eval/hedge_rate.json")
    parser.add_argument("--score-field", default="overall_score",
                        help="grade subfield to stratify on (only if input is graded)")
    args = parser.parse_args()

    traces = list(load_traces(args.input))
    if not traces:
        raise SystemExit(f"no traces found in {args.input}")
    print(f"Loaded {len(traces)} traces from {args.input}")

    # Split each trace into (response, optional score, tags).
    # Raw traces have .response and optional .messages/.tags; graded traces
    # additionally have .grade.{overall_score, ...}.
    responses = []
    for t in traces:
        text = t.get("response") or ""
        score = None
        if isinstance(t.get("grade"), dict):
            score = t["grade"].get(args.score_field)
        # tags can come in many shapes — be lenient.
        tags = t.get("example_tags") or t.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        # extract themes from "theme:..." tags (HealthBench convention)
        themes = [tg.split(":", 1)[1] for tg in tags if isinstance(tg, str) and tg.startswith("theme:")]
        responses.append({
            "response": text,
            "score": score,
            "themes": themes,
            "hedged": is_hedged(text),
        })

    # (a) overall
    n = len(responses)
    n_hedged = sum(1 for r in responses if r["hedged"])
    overall_rate = n_hedged / n if n else 0.0

    summary = {
        "input": args.input,
        "n_traces": n,
        "hedge_phrases_n": len(HEDGE_PATTERNS),
        "overall_hedge_rate": overall_rate,
    }

    # (b) by rubric score quartile (graded input only)
    have_scores = sum(1 for r in responses if r["score"] is not None)
    if have_scores >= 4:
        scored = [r for r in responses if r["score"] is not None]
        cutoffs = quartile_buckets([r["score"] for r in scored])
        if cutoffs is not None:
            q1, q2, q3 = cutoffs
            buckets = defaultdict(list)
            for r in scored:
                lab = quartile_label(r["score"], q1, q2, q3)
                buckets[lab].append(r["hedged"])
            summary["quartile_cutoffs"] = {
                "q1": q1, "q2": q2, "q3": q3,
                "score_field": args.score_field,
            }
            summary["hedge_by_quartile"] = {
                lab: {"n": len(v), "hedge_rate": sum(v) / len(v) if v else None}
                for lab, v in sorted(buckets.items())
            }
    else:
        summary["hedge_by_quartile"] = (
            "skipped — input has no .grade.overall_score; run "
            "filter_traces.py --graded-output first"
        )

    # (c) by theme
    theme_buckets = defaultdict(list)
    for r in responses:
        for theme in r["themes"]:
            theme_buckets[theme].append(r["hedged"])
    summary["hedge_by_theme"] = {
        theme: {"n": len(v), "hedge_rate": sum(v) / len(v) if v else None}
        for theme, v in sorted(theme_buckets.items())
    }

    # Write output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # Console summary
    print(f"\n=== overall hedge rate: {overall_rate*100:.1f}% ({n_hedged}/{n}) ===")
    if isinstance(summary.get("hedge_by_quartile"), dict):
        print("\n=== by rubric-score quartile ===")
        for lab in sorted(summary["hedge_by_quartile"]):
            s = summary["hedge_by_quartile"][lab]
            print(f"  {lab:<14}  n={s['n']:>4}  rate={s['hedge_rate']*100:>5.1f}%")
    if summary["hedge_by_theme"]:
        print("\n=== by theme ===")
        for theme, s in summary["hedge_by_theme"].items():
            print(f"  {theme:<24}  n={s['n']:>4}  rate={s['hedge_rate']*100:>5.1f}%")

    # Per the issue: if the rate is >80% unconditionally, the paper's calibration
    # framing is wrong. Print a loud warning so this isn't missed.
    if overall_rate > 0.80:
        print(f"\nWARNING: overall hedge rate {overall_rate*100:.1f}% > 80%.")
        print("If the rate is similarly high across all rubric-score quartiles,")
        print("the paper's 'calibrated abstention' framing should be revisited.")
        print("Add a paragraph to the paper's limitations section.")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
