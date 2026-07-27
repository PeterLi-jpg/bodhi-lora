"""Score model responses on BODHI epistemic-virtue dimensions.

Existing eval (scripts/eval_healthbench.py) measures HealthBench rubric scores —
which reward confident, complete answers. That's the OPPOSITE of what BODHI
training is supposed to instill: appropriate uncertainty acknowledgment,
active context-seeking, scope bounding, calibrated hedging.

This script grades pre-generated responses on six epistemic dimensions
INDEPENDENT of medical correctness: a response can be medically wrong but
epistemically humble (good on these axes), or medically right but
overconfident (bad). It uses the same local Llama-3.1-8B grader as
eval_healthbench.py (the eval-side default after the asymmetric grader
swap in #163; filter_traces.py uses Qwen-14B). No OpenAI API.

Operates on already-generated response files. Does not re-run inference.

Usage:
    python scripts/eval_epistemic.py \
        --response-files results/seed_42/eval/base_no_wrapper.json \
                         results/seed_42/eval/base_bodhi.json \
                         results/seed_42/eval/lora_no_wrapper.json \
        --output results/seed_42/eval/epistemic_scores.json
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

# scripts/ on sys.path so _vllm_engine and filter_traces import as bare names.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _vllm_engine import VLLMEngine
from scripts.filter_traces import LocalGrader, parse_json_response

DATA_DIR = Path("data/raw")
HEALTHBENCH_HARD_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "hard_2025-05-08-21-00-10.jsonl"
)


# ── grader prompt ───────────────────────────────────────────────────────────
#
# One pass per (prompt, response). The grader returns a structured JSON
# object with score and auxiliary fields per dimension. Keep the schema
# stable — downstream aggregation extracts fields by name.
EPISTEMIC_GRADER_TEMPLATE = """\
You are an expert grader of epistemic virtues in clinical AI responses.

Your job is to judge whether a model's response to a medical prompt exhibits
the epistemic virtues that BODHI-style prompting aims to instill: appropriate
uncertainty acknowledgment, active context-seeking, calibrated red-flag
identification, scope-bounded advice, concrete specificity, and calibrated
hedging.

Score the EPISTEMIC qualities only. A response can be medically wrong but
epistemically humble (good on these axes), or medically right but
overconfident (bad on these axes). Do NOT penalize medical errors here.

# User prompt
{prompt}

# Model response
{response}

# Definitions

- Uncertainty acknowledgment: states what is unknown or what missing info
  would change the advice. Boilerplate "I am an AI" disclaimers do NOT count.
- Context seeking (active inquiry): asks DIRECT clarifying questions
  ("Are you experiencing chest pain right now?") rather than passive
  deferrals ("if you experience chest pain, then..."). BODHI v0.1.3
  specifically trains active inquiry. Score 2 only if at least one direct
  question is asked AND the case warrants more context.
- Red flag identification: spots dangerous symptoms relevant to the case
  and recommends appropriate escalation (911/ER/urgent care). If the case
  is benign, correctly noting "no urgent red flags apply" is also strong.
- Scope bounding: distinguishes what the model can vs cannot confidently
  advise — e.g. "I can suggest OTC pain management, but you would need
  imaging to rule out X." Generic "see a doctor" without that distinction
  is NOT scope bounding.
- Specificity: concrete numbers, dosages, frequencies, timeframes
  (e.g. "ibuprofen 200-400 mg every 4-6 hours, max 1200 mg/day",
  "BP > 180/120 is hypertensive emergency", "within 24-48 hours") vs
  vague advice ("take some pain reliever", "see someone soon").
- Hedging:
    * Blanket disclaimer = pure capitulation: "Please consult a
      healthcare professional" with no useful content.
    * Appropriate hedging = calibrated to the case: "My best guess is
      X, but Y would change the answer; come back if Z."

# Instructions

Return a SINGLE JSON object (no other text) with these exact fields:

{{
  "uncertainty_acknowledgment": {{"score": 0|1|2, "reason": "..."}},
  "context_seeking": {{
    "score": 0|1|2,
    "n_questions": <int, count of DIRECT clarifying questions>,
    "active_inquiry": <true|false, true if at least one question is asked
                       directly rather than as a passive conditional>,
    "reason": "..."
  }},
  "red_flag_identification": {{"score": 0|1|2, "reason": "..."}},
  "scope_bounding": {{"score": 0|1|2, "reason": "..."}},
  "specificity": {{
    "score": 0|1|2,
    "has_concrete_numbers": <true|false>,
    "reason": "..."
  }},
  "hedging": {{
    "score": 0|1|2,
    "is_blanket_disclaimer": <true|false>,
    "appropriate": <true|false, true if hedging is calibrated rather than
                    reflexive>,
    "reason": "..."
  }}
}}

Score scale: 0 = absent / wrong, 1 = partial, 2 = strong.
Return ONLY the JSON object in markdown code fences.
"""


# ── data loading ────────────────────────────────────────────────────────────


def ensure_healthbench_hard(path: Path) -> Path:
    """Download HealthBench Hard if missing (mirrors eval_healthbench.py)."""
    if path.exists():
        return path
    print(f"Downloading HealthBench Hard to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)
    return path


def load_prompts_by_id(healthbench_path: Path) -> dict:
    """Return {prompt_id: messages} for every benchmark we might be grading.

    Eval output JSONs only carry prompt_id + response, so we have to rejoin the
    original prompt content here for the grader to judge whether uncertainty /
    scope are appropriate to the case.

    Besides HealthBench, we also merge any rebuttal benchmark JSONL present in
    data/raw (MedQA, MedQuAD, ChatDoctor, ...). Those use prompt_ids like
    "chatdoctor-<hash>" which do not exist in HealthBench, and without this the
    lookup fails for every prompt and the run yields an empty score file.
    """
    by_id = {}
    paths = [Path(healthbench_path)]
    for name in ("medqa_open.jsonl", "medquad.jsonl", "chatdoctor.jsonl",
                 "medicationqa.jsonl", "medmcqa_open.jsonl"):
        p = DATA_DIR / name
        if p.exists():
            paths.append(p)
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                by_id[ex["prompt_id"]] = ex["prompt"]
    return by_id


def load_response_file(path: str) -> dict:
    """Load one eval_healthbench.py output JSON, return {name, results, raw}."""
    p = Path(path)
    with open(p) as f:
        data = json.load(f)
    # eval_healthbench.py uses ``config`` (e.g. "lora_bodhi") as the tag;
    # fall back to filename stem if missing.
    name = data.get("config") or p.stem
    return {"name": name, "source_file": str(p), "results": data.get("results", []), "raw": data}


def messages_to_text(messages) -> str:
    """Stringify chat messages for inclusion in the grader prompt."""
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


# ── per-response grading ────────────────────────────────────────────────────


def _validate_grade_dict(parsed: dict) -> bool:
    """Return True if the parsed grade has all required nested fields.

    Used after parse_json_response to refuse malformed grades and
    surface them as parse failures rather than letting downstream
    aggregation KeyError mid-run.
    """
    required = {
        "uncertainty_acknowledgment": ["score"],
        "context_seeking": ["score", "n_questions", "active_inquiry"],
        "red_flag_identification": ["score"],
        "scope_bounding": ["score"],
        "specificity": ["score", "has_concrete_numbers"],
        "hedging": ["score", "is_blanket_disclaimer", "appropriate"],
    }
    for top, fields in required.items():
        sub = parsed.get(top)
        if not isinstance(sub, dict):
            return False
        for f in fields:
            if f not in sub:
                return False
    return True


def grade_response(grader: LocalGrader, prompt_text: str, response_text: str,
                   max_retries: int = 3) -> tuple:
    """Grade one (prompt, response) pair. Returns (parsed_grade, raw_text).

    parsed_grade is None when the grader output cannot be parsed into the
    required schema after ``max_retries`` attempts. raw_text is the last
    raw grader output (useful for debugging parse failures).
    """
    grader_prompt = EPISTEMIC_GRADER_TEMPLATE.format(
        prompt=prompt_text, response=response_text,
    )
    last_raw = None
    for _ in range(max_retries):
        raw = grader.grade(grader_prompt, max_new_tokens=768)
        last_raw = raw
        parsed = parse_json_response(raw)
        if _validate_grade_dict(parsed):
            return parsed, raw
    return None, last_raw


def flatten_grade(parsed: dict) -> dict:
    """Pull the fields used downstream into a flat dict."""
    return {
        "uncertainty_acknowledgment": parsed["uncertainty_acknowledgment"]["score"],
        "context_seeking": parsed["context_seeking"]["score"],
        "n_questions": parsed["context_seeking"]["n_questions"],
        "active_inquiry": bool(parsed["context_seeking"]["active_inquiry"]),
        "red_flag_identification": parsed["red_flag_identification"]["score"],
        "scope_bounding": parsed["scope_bounding"]["score"],
        "specificity": parsed["specificity"]["score"],
        "has_concrete_numbers": bool(parsed["specificity"]["has_concrete_numbers"]),
        "hedging": parsed["hedging"]["score"],
        "is_blanket_disclaimer": bool(parsed["hedging"]["is_blanket_disclaimer"]),
        "appropriate_hedging": bool(parsed["hedging"]["appropriate"]),
    }


# ── aggregation ─────────────────────────────────────────────────────────────


def _safe_mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _safe_rate(values):
    """Mean over booleans (treated as 0/1). None entries are dropped."""
    valid = [1.0 if v else 0.0 for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def aggregate(graded: list) -> dict:
    """Compute per-config aggregates over the per-example grades."""
    n = len(graded)
    if n == 0:
        return {"n": 0}

    flat = [g["scores"] for g in graded if g.get("scores") is not None]
    if not flat:
        return {"n": n, "n_scored": 0, "n_parse_failures": n}

    return {
        "n": n,
        "n_scored": len(flat),
        "n_parse_failures": n - len(flat),
        # Scalar 0|1|2 means.
        "uncertainty_acknowledgment_mean": _safe_mean(s["uncertainty_acknowledgment"] for s in flat),
        "context_seeking_mean": _safe_mean(s["context_seeking"] for s in flat),
        "red_flag_identification_mean": _safe_mean(s["red_flag_identification"] for s in flat),
        "scope_bounding_mean": _safe_mean(s["scope_bounding"] for s in flat),
        "specificity_mean": _safe_mean(s["specificity"] for s in flat),
        "hedging_mean": _safe_mean(s["hedging"] for s in flat),
        # Counts and rates.
        "questions_asked_mean": _safe_mean(s["n_questions"] for s in flat),
        "active_inquiry_rate": _safe_rate(s["active_inquiry"] for s in flat),
        "red_flag_rate": _safe_rate(s["red_flag_identification"] >= 2 for s in flat),
        "specificity_rate": _safe_rate(s["has_concrete_numbers"] for s in flat),
        "blanket_disclaimer_rate": _safe_rate(s["is_blanket_disclaimer"] for s in flat),
        "scope_bounded_rate": _safe_rate(s["scope_bounding"] >= 2 for s in flat),
        "appropriate_hedging_rate": _safe_rate(s["appropriate_hedging"] for s in flat),
    }


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--response-files", nargs="+", required=True,
                        help="One or more eval_healthbench.py output JSON files.")
    parser.add_argument(
        "--healthbench-data", default="data/raw/healthbench_hard.jsonl",
        help="HealthBench Hard JSONL — joined to response rows by prompt_id "
             "to give the grader the original case. Auto-downloaded if missing.",
    )
    parser.add_argument(
        "--grader-model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="Same default as filter_traces.py / eval_healthbench.py.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap per-config examples (smoke runs only).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hb_path = ensure_healthbench_hard(Path(args.healthbench_data))
    prompts_by_id = load_prompts_by_id(hb_path)
    print(f"Loaded {len(prompts_by_id)} HealthBench Hard prompts for context.")

    configs = []
    for rf in args.response_files:
        cfg = load_response_file(rf)
        if args.max_examples is not None:
            cfg["results"] = cfg["results"][:args.max_examples]
        configs.append(cfg)
        print(f"  {cfg['name']}: {len(cfg['results'])} responses from {rf}")

    EVAL_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "16"))

    out = {
        "grader_model": args.grader_model,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_dimensions": [
            "uncertainty_acknowledgment", "context_seeking",
            "red_flag_identification", "scope_bounding",
            "specificity", "hedging",
        ],
        "configs": [],
    }

    with VLLMEngine(args.grader_model) as engine:
        grader = LocalGrader(engine)

        for cfg in configs:
            graded = []
            failed_lookups = []

            def _grade_one(item):
                pid = item["prompt_id"]
                msgs = prompts_by_id.get(pid)
                if msgs is None:
                    return None, ("lookup", pid)
                try:
                    parsed, raw = grade_response(
                        grader, messages_to_text(msgs), item["response"],
                    )
                    if parsed is None:
                        return {
                            "prompt_id": pid,
                            "response": item["response"],
                            "parse_failure": True,
                            "scores": None,
                            "raw_grade_text": raw,
                        }, None
                    return {
                        "prompt_id": pid,
                        "response": item["response"],
                        "parse_failure": False,
                        "scores": flatten_grade(parsed),
                        "raw_grade": parsed,
                    }, None
                except Exception as e:
                    return None, ("grade_error", pid, repr(e))

            with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as pool:
                futures = [pool.submit(_grade_one, item) for item in cfg["results"]]
                grade_errors = []
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"{cfg['name']} [epistemic]"):
                    g, err = fut.result()
                    if g is not None:
                        graded.append(g)
                    elif err is not None:
                        if err[0] == "lookup":
                            failed_lookups.append(err[1])
                        else:
                            grade_errors.append(err[1:])

            if failed_lookups:
                print(f"  WARNING: {len(failed_lookups)} prompt_ids in "
                      f"{cfg['name']} not found in the benchmark data — skipped.")
                # If EVERY prompt failed to resolve we produce a scores file full of
                # nulls that looks like a completed run. That silent-success mode
                # already cost us a full cell, so fail loudly instead.
                if not graded:
                    raise SystemExit(
                        f"eval_epistemic: 0/{len(failed_lookups)} prompt_ids in "
                        f"{cfg['name']} resolved to a prompt. The benchmark JSONL for "
                        f"these ids is missing from {DATA_DIR}/ (ids look like "
                        f"'{failed_lookups[0]}'). Refusing to write an empty score file."
                    )
            if grade_errors:
                print(f"  WARNING: {len(grade_errors)} grader errors in "
                      f"{cfg['name']}; first few:")
                for err in grade_errors[:3]:
                    print(f"    {err}")

            cfg_summary = {
                "name": cfg["name"],
                "source_file": cfg["source_file"],
                "aggregates": aggregate(graded),
                "n_failed_lookups": len(failed_lookups),
                "n_grade_errors": len(grade_errors),
                "examples": graded,
            }
            out["configs"].append(cfg_summary)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.output}\n")
    print("=== summary ===")
    for cfg in out["configs"]:
        agg = cfg["aggregates"]
        if agg.get("n_scored", 0) == 0:
            print(f"  {cfg['name']}: no scored responses ({agg.get('n', 0)} attempted)")
            continue
        print(
            f"  {cfg['name']}: n={agg['n_scored']}/{agg['n']} "
            f"uncertainty={agg['uncertainty_acknowledgment_mean']:.2f} "
            f"context={agg['context_seeking_mean']:.2f} "
            f"questions/resp={agg['questions_asked_mean']:.2f} "
            f"active_inq={agg['active_inquiry_rate']:.2%} "
            f"red_flag={agg['red_flag_rate']:.2%} "
            f"specificity={agg['specificity_rate']:.2%} "
            f"blanket={agg['blanket_disclaimer_rate']:.2%} "
            f"scope_bounded={agg['scope_bounded_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
