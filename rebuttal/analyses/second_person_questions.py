"""Rebuttal analysis: how often does a response ask the reader a question? (FzpA, MR-W3)

A grader-free check on the active-inquiry shift. The posted replies
(posted/FzpA_2_followup.md:13, drafts/meta_review_part1.md:27) say the base model asks a
second-person question in 18.4% of responses and the adapter in 39.6%. No script or
output behind those two numbers was saved (PROV-7). This is the scripted count.

Definition, fixed before comparing with the posted values:

  1. Remove markdown emphasis and quote marks (* _ ` # >), so "**Are you ...?**"
     reads as a sentence that ends in "?".
  2. Split the response into sentences at whitespace after . ! ? and at line breaks.
  3. A sentence is a second-person question if it ends in "?" and contains one of
     you / your / yours / yourself / yourselves as a word.
  4. A response counts if it has at least one such sentence.

The denominator is every completion in the cell's seed_*/eval/<condition>.json. No
grader is involved, so epistemic parse failures do not drop rows.

It is a text match: it cannot tell an apt question from a rhetorical one, and a
question in a quoted example counts the same as one put to the reader.

Runs against the submitted cell's eval files (--results-dir). No inference.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.paper_stats import eval_path  # noqa: E402

# The submitted cell (results/PROVENANCE.md).
CELL = REPO / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05"
SEEDS = (7, 13, 42, 99, 101)
CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]
CONFIG_DISPLAY = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Wrapper (Base+CoT)",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+Wrapper",
}
# Transcribed from the posted replies, kept only for the comparison in the output.
POSTED = {"base_no_wrapper": 0.184, "lora_no_wrapper": 0.396,
          "source": "rebuttal/responses/posted/FzpA_2_followup.md:13; rebuttal/responses/drafts/meta_review_part1.md:27"}

MARKDOWN = re.compile(r"[*_`#>]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE)


def has_second_person_question(text):
    text = MARKDOWN.sub("", text or "")
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip().rstrip("\"')]")
        if sentence.endswith("?") and SECOND_PERSON.search(sentence):
            return True
    return False


def load_flags(results_dir, seeds=SEEDS):
    """config -> seed -> list of bools, one per completion."""
    out = {cfg: {} for cfg in CONFIG_ORDER}
    for seed in seeds:
        for cfg in CONFIG_ORDER:
            with open(eval_path(Path(results_dir), seed, cfg)) as f:
                results = json.load(f)["results"]
            out[cfg][seed] = [has_second_person_question(r["response"]) for r in results]
    return out


def summarize(flags, seeds=SEEDS):
    res = {"definition": "sentence ending in '?' with a second-person pronoun, "
                         "markdown stripped; denominator = all completions",
           "conditions": {}, "posted": POSTED}
    for cfg in CONFIG_ORDER:
        pooled = [f for s in seeds for f in flags[cfg][s]]
        res["conditions"][cfg] = {
            "count": int(sum(pooled)), "n": len(pooled), "rate": sum(pooled) / len(pooled),
            "per_seed_rate": {str(s): float(np.mean(flags[cfg][s])) for s in seeds},
        }
    # Seed-level paired comparison, the same unit as the paper's other tests.
    a = np.array([res["conditions"]["lora_no_wrapper"]["per_seed_rate"][str(s)] for s in seeds])
    b = np.array([res["conditions"]["base_no_wrapper"]["per_seed_rate"][str(s)] for s in seeds])
    d = a - b
    half = float(stats.t.ppf(0.975, len(d) - 1)) * d.std(ddof=1) / np.sqrt(len(d))
    t = stats.ttest_rel(a, b)
    res["lora_minus_base"] = {"diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                              "t": float(t.statistic), "df": len(d) - 1, "p": float(t.pvalue)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(CELL))
    ap.add_argument("--out", default=str(CELL / "analysis" / "second_person_questions.json"))
    args = ap.parse_args()

    res = summarize(load_flags(args.results_dir))
    print("=" * 72)
    print("Completions with a second-person question (pooled 5 seeds)")
    print("=" * 72)
    for cfg in CONFIG_ORDER:
        c = res["conditions"][cfg]
        print(f"  {CONFIG_DISPLAY[cfg]:22s} {100 * c['rate']:5.1f}%  ({c['count']}/{c['n']})")
    lb = res["lora_minus_base"]
    print(f"\n  LoRA - Base: {100 * lb['diff']:+.1f}pp, 95% CI [{100 * lb['ci95'][0]:+.1f}, "
          f"{100 * lb['ci95'][1]:+.1f}], paired t({lb['df']}) = {lb['t']:.2f}, p = {lb['p']:.2g}")
    print(f"  Posted: Base {100 * POSTED['base_no_wrapper']:.1f}%, adapter {100 * POSTED['lora_no_wrapper']:.1f}%"
          " (no script behind them; not reproduced exactly by this definition)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
