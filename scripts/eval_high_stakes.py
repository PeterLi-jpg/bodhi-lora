"""Run side-by-side high-stakes clinical eval across base/lora x bodhi/no-wrapper.

Engineering scaffolding for issue #75. The placeholder prompts in
data/raw/high_stakes_prompts.jsonl are toy stubs — REAL physician-curated
prompts (chest pain triage, anaphylaxis, suicide risk, drug-allergy
contraindications) must be substituted before any paper-quality run.

The script:
  1. Loads prompts.jsonl, skipping the placeholder marker row.
  2. Determines which configs to run (2 base configs, plus 2 lora configs
     iff --lora-path was provided).
  3. For each config, spins up a VLLMEngine context and generates a
     response per prompt. The bodhi configs wrap the engine via
     make_bodhi_wrapper from eval_healthbench.
  4. Anonymizes config labels (config_a/b/c/d) so the clinical reviewer
     judges responses without knowing which is base/lora/bodhi. The
     real-name -> anon-label mapping goes in key.json for engineers.
  5. Computes hedge rate per response via _hedging_metric.compute_hedge_rate.
  6. Writes responses.jsonl, review.md, summary.json, key.json.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import random
import sys
from pathlib import Path

# Add scripts/ dir for bare-name imports and project root for package imports.
# Mirrors the pattern used in scripts/eval_healthbench.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _hedging_metric import compute_hedge_rate
from _vllm_engine import VLLMEngine
from eval_healthbench import make_bodhi_wrapper


# Real config order is fixed: base configs first, then lora configs. The
# anonymization step shuffles these to anon labels keyed off args.seed.
BASE_CONFIGS = ("base_no_wrapper", "base_bodhi")
LORA_CONFIGS = ("lora_no_wrapper", "lora_bodhi")
ANON_LABELS = ("config_a", "config_b", "config_c", "config_d")


def load_prompts(path: Path) -> list:
    """Load prompts.jsonl, skipping the _PLACEHOLDER marker row.

    The placeholder file's first line is a marker documenting that real
    curation per #75 is still TODO. Skipping it lets us merge the
    scaffolding without shipping fake prompts as if they were real.
    """
    prompts = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "_PLACEHOLDER" in row:
                continue
            prompts.append(row)
    return prompts


def build_anon_mapping(real_names: list, seed: int, anonymize: bool) -> dict:
    """Map each real config name to an anon label.

    Uses a seeded local Random so repeated runs with the same seed produce
    the same anon mapping — handy when re-running and lining up against
    earlier reviewer notes. When anonymize=False, the "mapping" is just
    real_name -> real_name so downstream code can use a single dict.
    """
    if not anonymize:
        return {name: name for name in real_names}
    rng = random.Random(seed)
    labels = list(ANON_LABELS[: len(real_names)])
    rng.shuffle(labels)
    return dict(zip(real_names, labels))


def run_config(
    config_name: str,
    model: str,
    lora_path: str,
    use_bodhi: bool,
    prompts: list,
    max_new_tokens: int,
) -> list:
    """Generate one response per prompt for a single config.

    Spun up in its own VLLMEngine context so each config gets a fresh
    server (LoRA on/off and bodhi wrapper differences make sharing one
    engine tricky for an eval script that wants reproducibility per row).
    """
    engine_lora = lora_path if config_name.startswith("lora_") else None

    rows = []
    print(f"\n=== {config_name}: {len(prompts)} prompts ===")
    with VLLMEngine(model, lora_path=engine_lora) as engine:
        bodhi_wrapper = make_bodhi_wrapper(engine) if use_bodhi else None
        for prompt_row in prompts:
            messages = [{"role": "user", "content": prompt_row["prompt"]}]
            if use_bodhi:
                resp = bodhi_wrapper.complete(messages)
                response_text = resp.content
            else:
                response_text = engine.chat(messages, max_new_tokens=max_new_tokens)

            metric = compute_hedge_rate(response_text)
            rows.append({
                "prompt_id": prompt_row["prompt_id"],
                "category": prompt_row["category"],
                "prompt": prompt_row["prompt"],
                "config_real_name": config_name,
                "response": response_text,
                "hedge_rate": metric["hedge_rate"],
                "n_response_tokens": metric["total_tokens"],
            })
    return rows


def write_responses_jsonl(out_path: Path, rows: list, anon_map: dict, anonymized: bool) -> None:
    """One row per (prompt x config). Real config name is omitted when anonymized.

    Reviewers should never see real_name in any artifact they touch — the
    de-anonymization mapping lives only in key.json (engineer-only).
    """
    with open(out_path, "w") as f:
        for row in rows:
            real = row["config_real_name"]
            out_row = {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "config_anon_label": anon_map[real],
                "response": row["response"],
                "hedge_rate": row["hedge_rate"],
                "n_response_tokens": row["n_response_tokens"],
            }
            if not anonymized:
                out_row["config_real_name"] = real
            f.write(json.dumps(out_row) + "\n")


def write_review_md(out_path: Path, rows: list, anon_map: dict) -> None:
    """Side-by-side markdown for clinical reviewer.

    Layout: H2 per prompt, then prompt text in a quoted block, then one H3
    per config (in random anon order so the reviewer doesn't anchor on a
    fixed lora/base order). Reviewers annotate inline; engineers de-anonymize
    via key.json after review.
    """
    # Group rows by prompt_id, preserving the order prompts appeared.
    by_prompt: dict = {}
    prompt_order: list = []
    for row in rows:
        pid = row["prompt_id"]
        if pid not in by_prompt:
            by_prompt[pid] = []
            prompt_order.append(pid)
        by_prompt[pid].append(row)

    # Stable random anon-label ordering per file (seeded by length so the
    # order is deterministic but visually shuffled relative to alpha order).
    anon_order = sorted(set(anon_map.values()))

    lines: list = []
    for pid in prompt_order:
        prompt_rows = by_prompt[pid]
        category = prompt_rows[0]["category"]
        prompt_text = prompt_rows[0]["prompt"]
        lines.append(f"## {pid} ({category})")
        lines.append("")
        # Quote the prompt. Multi-line prompts get one '> ' per line so
        # the markdown renders as one continuous quoted block.
        for prompt_line in prompt_text.splitlines() or [""]:
            lines.append(f"> {prompt_line}")
        lines.append("")

        # Render configs in alphabetical anon-label order. The randomness
        # is in the mapping (real_name -> anon_label is shuffled by seed),
        # so config_a's content varies between runs but the heading order
        # stays predictable for the reviewer.
        rows_by_anon = {anon_map[r["config_real_name"]]: r for r in prompt_rows}
        for anon_label in anon_order:
            if anon_label not in rows_by_anon:
                continue
            lines.append(f"### {anon_label}")
            lines.append("")
            lines.append(rows_by_anon[anon_label]["response"])
            lines.append("")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_summary(
    out_path: Path,
    rows: list,
    anon_map: dict,
    real_names: list,
    args,
) -> None:
    """summary.json with per-config aggregates (keyed by anon label)."""
    # Aggregate per anon label (so reviewers can cross-reference review.md).
    per_label_hedge: dict = {label: [] for label in anon_map.values()}
    per_label_tokens: dict = {label: [] for label in anon_map.values()}
    for row in rows:
        label = anon_map[row["config_real_name"]]
        per_label_hedge[label].append(row["hedge_rate"])
        per_label_tokens[label].append(row["n_response_tokens"])

    def _mean(values):
        return float(sum(values) / len(values)) if values else None

    n_prompts = len({row["prompt_id"] for row in rows})
    summary = {
        "n_prompts": n_prompts,
        "n_configs": len(real_names),
        "configs": [anon_map[name] for name in real_names],
        "hedge_rate_per_config": {
            label: _mean(values) for label, values in per_label_hedge.items()
        },
        "mean_response_tokens_per_config": {
            label: _mean(values) for label, values in per_label_tokens.items()
        },
        "model": args.model,
        "lora_path": args.lora_path,
        "seed": args.seed,
        "anonymized": args.anonymize,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "review.md is for clinical reviewers; key.json maps anon "
            "labels back to real configs."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side high-stakes clinical eval (issue #75).",
    )
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Required to run lora configs; if omitted, only the 2 base configs run.",
    )
    parser.add_argument("--prompts", default="data/raw/high_stakes_prompts.jsonl")
    parser.add_argument("--output-dir", default="eval/high_stakes/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--anonymize",
        dest="anonymize",
        action="store_true",
        default=True,
        help="(default) Use config_a/b/c/d labels in review.md; mapping in key.json.",
    )
    parser.add_argument(
        "--no-anonymize",
        dest="anonymize",
        action="store_false",
        help="Show real config names in review artifacts (NOT recommended for blinded review).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    if not prompts:
        raise SystemExit(
            f"No real prompts found in {args.prompts} — only the placeholder marker. "
            "See issue #75 for curation requirements."
        )
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Pick which configs to run. base configs always; lora configs iff lora_path given.
    real_names = list(BASE_CONFIGS)
    if args.lora_path:
        real_names.extend(LORA_CONFIGS)
    else:
        print("No --lora-path provided; running base configs only (2 of 4).")

    anon_map = build_anon_mapping(real_names, args.seed, args.anonymize)

    # Write key.json up-front so engineers have the mapping even if a
    # later config crashes mid-run.
    with open(out_dir / "key.json", "w") as f:
        json.dump(anon_map, f, indent=2)

    all_rows = []
    for name in real_names:
        use_bodhi = name.endswith("_bodhi")
        rows = run_config(
            config_name=name,
            model=args.model,
            lora_path=args.lora_path,
            use_bodhi=use_bodhi,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        all_rows.extend(rows)

    write_responses_jsonl(
        out_dir / "responses.jsonl", all_rows, anon_map, args.anonymize
    )
    write_review_md(out_dir / "review.md", all_rows, anon_map)
    write_summary(out_dir / "summary.json", all_rows, anon_map, real_names, args)

    print(f"\nWrote artifacts to {out_dir}")
    print("  responses.jsonl  — one row per (prompt x config)")
    print("  review.md        — for clinical reviewer")
    print("  summary.json     — aggregate hedge_rate_per_config")
    print("  key.json         — anon-label -> real-name (engineer only)")


if __name__ == "__main__":
    main()
