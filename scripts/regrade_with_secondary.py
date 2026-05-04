"""Cross-family grader robustness check.

Reads the existing 5-seed Stage 4 evaluation responses (Llama-3.1-8B graded)
and re-grades each (prompt, response) pair with a SECONDARY grader from a
DIFFERENT model family. Writes the secondary-grader scores alongside the
primary scores so we can report per-config Spearman ρ and Cohen's κ between
the two graders — the standard cross-family robustness check that reviewers
ask for to rule out grader-family bias.

Why this matters
----------------
Llama-3.1-8B is one model family. If the result holds with a Mistral or Qwen
grader (different families, different RLHF/SFT training data), the per-config
ordering is not an artifact of one grader's idiosyncrasies. The paper's
asymmetric-grading design already separates filter (Qwen-14B) from eval
(Llama-8B); this script adds a third grader (Mistral) as a tie-breaking
audit.

Inputs
------
- results_modal/seed_<N>/<config>.json   for N ∈ SEEDS, config ∈ CONFIGS
  Each has results[].response and results[].score (primary Llama grade).
- HealthBench Hard rubric per prompt (data/raw/healthbench_hard.jsonl) —
  same rubric the primary grader used.

Outputs
-------
- results_modal/seed_<N>/<config>_secondary_<grader>.json
  Same shape as the primary file but with new ``score`` and per-criterion
  results from the secondary grader.

This script is meant to run on the TPU with vllm-tpu. The docker container
is reused across all (seed, config) pairs so we pay the vllm cold-start
cost only once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse the existing grader plumbing.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _vllm_engine import VLLMEngine  # noqa: E402
from scripts.filter_traces import LocalGrader, grade_trace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_modal"
HEALTHBENCH_PATH = ROOT / "data" / "raw" / "healthbench_hard.jsonl"

SEEDS = (7, 13, 42, 99, 101)
CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")


def load_rubric_index() -> dict[str, list[dict]]:
    """prompt_id -> rubric items list (same source the primary grader used)."""
    out: dict[str, list[dict]] = {}
    with open(HEALTHBENCH_PATH) as f:
        for line in f:
            ex = json.loads(line)
            out[ex["prompt_id"]] = ex.get("rubrics", [])
    return out


def load_existing_results(seed: int, config: str) -> dict:
    """Load one seed/config eval JSON exactly as the primary run wrote it."""
    path = RESULTS_DIR / f"seed_{seed}" / f"{config}.json"
    with open(path) as f:
        return json.load(f)


def regrade_one(grader: LocalGrader, item: dict, rubric: list[dict]) -> dict:
    """Re-grade one response. Returns the same shape grade_trace does."""
    return grade_trace(grader, item["messages"], item["response"], rubric)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--secondary-grader",
                   default="mistralai/Mistral-7B-Instruct-v0.3",
                   help="HF model id for the secondary grader.")
    p.add_argument("--grader-tag", default="mistral",
                   help="Short tag used in output filenames "
                        "(seed_<N>/<cfg>_secondary_<tag>.json).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap per config (for smoke runs).")
    args = p.parse_args()

    if not HEALTHBENCH_PATH.is_file():
        raise SystemExit(f"missing {HEALTHBENCH_PATH}")
    rubric_idx = load_rubric_index()
    print(f"loaded {len(rubric_idx)} rubrics")

    # Single grader engine for the whole run — saves the per-config cold-start
    # cost (~3 min × 4 configs × 5 seeds = 1 hr saved).
    print(f"starting secondary grader: {args.secondary_grader}")
    with VLLMEngine(args.secondary_grader) as engine:
        grader = LocalGrader(engine)

        for seed in args.seeds:
            for cfg in args.configs:
                src = RESULTS_DIR / f"seed_{seed}" / f"{cfg}.json"
                if not src.is_file():
                    print(f"  SKIP {src} (missing)")
                    continue
                with open(src) as f:
                    data = json.load(f)
                results = data.get("results", [])
                if args.limit is not None:
                    results = results[: args.limit]

                new_results = []
                for i, item in enumerate(results, 1):
                    pid = item["prompt_id"]
                    rubric = rubric_idx.get(pid)
                    if not rubric:
                        new_results.append({
                            **item,
                            "secondary_grade_error": "rubric not found",
                        })
                        continue
                    try:
                        grade = regrade_one(grader, item, rubric)
                    except Exception as e:  # noqa: BLE001
                        new_results.append({
                            **item,
                            "secondary_grade_error": repr(e)[:200],
                        })
                        continue
                    new_results.append({
                        "prompt_id": pid,
                        "response": item.get("response", ""),
                        "primary_score": item.get("score"),
                        "secondary_score": grade["overall_score"],
                        "secondary_criteria_results": grade["criteria_results"],
                        "secondary_parse_failures": grade.get("parse_failures", 0),
                    })
                    if i % 25 == 0:
                        print(f"    seed={seed} cfg={cfg}  {i}/{len(results)}")

                out = {
                    "seed": seed,
                    "config": cfg,
                    "primary_grader": "meta-llama/Llama-3.1-8B-Instruct",
                    "secondary_grader": args.secondary_grader,
                    "n_examples": len(new_results),
                    "results": new_results,
                }
                dest = RESULTS_DIR / f"seed_{seed}" / f"{cfg}_secondary_{args.grader_tag}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(out, indent=2))
                print(f"wrote {dest}  ({len(new_results)} regraded)")


if __name__ == "__main__":
    main()
