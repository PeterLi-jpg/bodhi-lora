"""Theme-stratified Stage 4 / Stage 5 analysis (issue #60 follow-through).

Issue #60: even after dedup by prompt_id, rubric themes (``emergency_referrals``,
``hedging``, ``context_seeking``, etc.) can appear in both train and eval.
This script reports per-theme deltas across the 4 configs so a reviewer can
distinguish global gains from a single-theme spike that would suggest
memorization.

Inputs:
  - The 4 Stage 4 JSONs per seed: base_no_wrapper, base_bodhi, lora_no_wrapper,
    lora_bodhi. Each has ``results[*].tag_scores`` from the rubric grader.
  - Optional Stage 5 epistemic_scores.json — adds per-virtue breakdown
    (uncertainty_acknowledgment, active_inquiry_rate, ...).

Output: ``analysis/theme_stratified.json`` per seed with:
  {
    "tags": {
      "<tag>": {
        "n_prompts_with_tag": int,
        "scores": {
          "base_no_wrapper": {"mean": ..., "n": ...},
          "base_bodhi":      {...},
          "lora_no_wrapper": {...},
          "lora_bodhi":      {...},
        },
        "internalization_delta_mean":
            mean(lora_no_wrapper) - mean(base_no_wrapper),
      },
      ...
    },
    "epistemic_per_virtue": { ... }   // if Stage 5 present
  }

Usage:
  python -m scripts.analysis.theme_stratified \
      --eval-dir results_tunix/seed_42/eval/seed_42 \
      --output   results_tunix/seed_42/analysis/theme_stratified.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")


def _load_eval_results(path: Path) -> list:
    with open(path) as f:
        return json.load(f).get("results", [])


def _per_tag_scores(eval_results: list) -> dict:
    """Aggregate tag → list of scores across prompts.

    Each prompt has ``tag_scores: {tag: score}``. We accumulate one score
    per (prompt, tag) pair so a prompt with two tags contributes to two
    aggregates.
    """
    per_tag: dict = defaultdict(list)
    for r in eval_results:
        for tag, score in (r.get("tag_scores") or {}).items():
            if score is None:
                continue
            try:
                per_tag[tag].append(float(score))
            except (TypeError, ValueError):
                continue
    return dict(per_tag)


def _summary(values: list) -> dict:
    if not values:
        return {"n": 0, "mean": None, "stdev": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
    }


def _stage5_per_virtue(epistemic_path: Path) -> dict:
    """Pull the per-virtue mean per config from Stage 5 epistemic_scores.json."""
    if not epistemic_path.is_file():
        return {}
    with open(epistemic_path) as f:
        data = json.load(f)
    out: dict = {}
    for cfg in data.get("configs", []):
        name = cfg.get("name")
        agg = cfg.get("aggregates") or {}
        if name and agg:
            out[name] = agg
    return {
        "dimensions": data.get("epistemic_dimensions", []),
        "per_config_aggregates": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-dir", required=True, type=Path,
                   help="Directory containing the 4 Stage-4 config JSONs and "
                        "(optionally) epistemic_scores.json from Stage 5.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write theme_stratified.json.")
    p.add_argument("--imbalance-threshold", type=float, default=2.0,
                   help="Flag tags whose train/eval frequency ratio exceeds "
                        "this (matches scripts/check_dataset_overlap.py).")
    args = p.parse_args()

    eval_dir = args.eval_dir.expanduser()
    if not eval_dir.is_dir():
        raise SystemExit(f"--eval-dir {eval_dir} does not exist")

    # Pull tag → score lists per config.
    per_config_per_tag: dict = {}
    for cfg in CONFIGS:
        cfg_path = eval_dir / f"{cfg}.json"
        if not cfg_path.is_file():
            print(f"  WARN: missing {cfg_path.name}, skipping config")
            per_config_per_tag[cfg] = {}
            continue
        per_config_per_tag[cfg] = _per_tag_scores(_load_eval_results(cfg_path))

    # Union of tags across configs.
    all_tags = set()
    for tag_dict in per_config_per_tag.values():
        all_tags.update(tag_dict.keys())

    tags_out: dict = {}
    for tag in sorted(all_tags):
        per_cfg = {}
        for cfg in CONFIGS:
            per_cfg[cfg] = _summary(per_config_per_tag.get(cfg, {}).get(tag, []))
        # Internalization delta = lora_no_wrapper - base_no_wrapper at the
        # mean level for this specific tag. Positive = LoRA gained ground
        # on this theme without inference-time scaffolding.
        bnw = per_cfg["base_no_wrapper"]["mean"]
        lnw = per_cfg["lora_no_wrapper"]["mean"]
        delta = (lnw - bnw) if (bnw is not None and lnw is not None) else None
        tags_out[tag] = {
            "n_obs_min_across_configs":
                min(per_cfg[c]["n"] for c in CONFIGS) if all(
                    per_cfg[c]["n"] is not None for c in CONFIGS
                ) else 0,
            "scores": per_cfg,
            "internalization_delta_mean": delta,
        }

    # Optional Stage 5 per-virtue breakdown.
    epistemic_path = eval_dir / "epistemic_scores.json"
    epistemic_block = _stage5_per_virtue(epistemic_path)

    out = {
        "eval_dir": str(eval_dir),
        "n_tags": len(all_tags),
        "tags": tags_out,
        "epistemic_per_virtue": epistemic_block,
        "interpretation": (
            "For each rubric tag, scores[<config>].mean shows the average "
            "rubric-correctness across prompts that bear that tag in the 200 "
            "Hard subset for this seed. internalization_delta_mean = "
            "mean(lora_no_wrapper) - mean(base_no_wrapper) per tag; positive "
            "means LoRA training improved that theme's score even without the "
            "BODHI wrapper at inference. A theme with a much larger delta than "
            "the global mean delta may indicate memorization rather than "
            "generalization (cross-reference the per-seed tag-overlap report "
            "in pipeline.log; tags with >2x train/eval frequency skew are "
            "candidates for that suspicion). The Stage-5 per-virtue block "
            "summarises the BODHI epistemic-virtue grader's per-config means."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}  (n_tags={len(all_tags)})")
    # Top-3 tags by absolute delta — a quick eyeball of where LoRA moved most.
    deltas = sorted(
        ((t, info["internalization_delta_mean"]) for t, info in tags_out.items()
         if info["internalization_delta_mean"] is not None),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    if deltas:
        print("  largest |internalization_delta_mean| tags:")
        for tag, d in deltas[:5]:
            print(f"    {tag}: {d:+.4f}")


if __name__ == "__main__":
    main()
