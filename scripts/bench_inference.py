#!/usr/bin/env python3
"""
scripts/bench_inference.py

Latency / cost benchmark for issue #29.
Measures four inference configurations on the 200-prompt holdout:
  base_no_wrapper   – base model, single call
  base_wrapper      – base model + BOHDI wrapper (3-5 calls/prompt)
  lora_no_wrapper   – LoRA-merged weights, single call
  lora_wrapper      – LoRA-merged weights + BOHDI wrapper

Outputs eval/latency.json with per-prompt timings and aggregates.

Usage:
  python scripts/bench_inference.py \
      --config base_no_wrapper \
      --model google/medgemma-27b-text-it \
      --sample-ids data/raw/hard_200_sample_ids.json \
      --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
      --output eval/latency.json

  python scripts/bench_inference.py \
      --config lora_no_wrapper \
      --model google/medgemma-27b-text-it \
      --lora-path checkpoints/best \
      --sample-ids data/raw/hard_200_sample_ids.json \
      --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
      --output eval/latency.json

Run once per config, each result is merged into the same output JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONFIGS = [
    "base_no_wrapper",
    "base_wrapper",
    "lora_no_wrapper",
    "lora_wrapper",
]

# Target response length used by the issue spec for wall-clock normalisation.
TARGET_RESPONSE_TOKENS = 512

# GPU-hours formula:  (total_wall_seconds / 3600) * n_gpus  / n_prompts * 1000
N_GPUS = 1  # single H100 as specified in the issue


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference latency benchmark (issue #29)")
    p.add_argument(
        "--config",
        required=True,
        choices=VALID_CONFIGS,
        help="Which of the four configurations to benchmark.",
    )
    p.add_argument(
        "--model",
        default="google/medgemma-27b-text-it",
        help="HuggingFace model ID for the base model.",
    )
    p.add_argument(
        "--lora-path",
        default=None,
        help="Path to PEFT LoRA checkpoint (required for lora_* configs).",
    )
    p.add_argument(
        "--sample-ids",
        required=True,
        help="Path to JSON file with the 200-prompt holdout sample IDs.",
    )
    p.add_argument(
        "--healthbench-data",
        nargs="+",
        required=True,
        help="One or more HealthBench .jsonl files to load prompts from.",
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=200,
        help="Number of prompts to benchmark (default 200).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=TARGET_RESPONSE_TOKENS,
        help="Max tokens to generate per prompt.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--output",
        default="eval/latency.json",
        help="Path for the output JSON (existing results are preserved and merged).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prompts(sample_ids_path: str, hb_data_paths: list[str], n: int) -> list[str]:
    """Return up to n prompt strings from the holdout set."""
    with open(sample_ids_path) as f:
        holdout_ids: set[str] = set(json.load(f))

    prompts: list[str] = []
    for path in hb_data_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # HealthBench rows use 'sample_id' or 'id'
                sid = str(row.get("sample_id") or row.get("id", ""))
                if sid in holdout_ids:
                    # Extract the user turn from 'conversation' list or fall back to 'prompt'
                    conversation = row.get("conversation") or []
                    user_turns = [m["content"] for m in conversation if m.get("role") == "user"]
                    prompt = user_turns[0] if user_turns else row.get("prompt", "")
                    if prompt:
                        prompts.append(prompt)
                if len(prompts) >= n:
                    break
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        print(
            f"[warn] Only found {len(prompts)} prompts matching holdout IDs "
            f"(requested {n}). Proceeding with what's available.",
            file=sys.stderr,
        )
    return prompts[:n]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_id: str,
    lora_path: str | None,
    config: str,
) -> tuple[Any, Any]:
    """Load model + tokenizer, merging LoRA weights if needed."""
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.startswith("lora_"):
        if lora_path is None:
            raise ValueError("--lora-path is required for lora_* configs.")
        print(f"Loading and merging LoRA weights from {lora_path} ...")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
        # Merge adapter into base weights so inference cost equals a plain model.
        model = model.merge_and_unload()
        print("LoRA weights merged.")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[int, int, float]:
    """
    Run one forward pass.
    Returns (prompt_tokens, response_tokens, wall_clock_seconds).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_token_count = inputs["input_ids"].shape[1]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic, fair comparison
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - t0

    response_tokens = output_ids.shape[1] - prompt_token_count
    return prompt_token_count, response_tokens, wall


def run_no_wrapper(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[dict]:
    device = next(model.parameters()).device.type
    results = []
    for prompt in tqdm(prompts, desc="no_wrapper"):
        p_tok, r_tok, wall = _generate_once(model, tokenizer, prompt, max_new_tokens, device)
        results.append(
            {
                "prompt_tokens": p_tok,
                "response_tokens": r_tok,
                "wall_clock_s": wall,
                "model_calls": 1,
            }
        )
    return results


def run_with_wrapper(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[dict]:
    """
    Simulate the BOHDI wrapper multi-call pattern.

    The wrapper calls the model multiple times per prompt:
      1. Pre-call: inject BOHDI system prompt, get initial response
      2. Calibration call: ask model to rate its own certainty
      3. Final call: produce the BOHDI-formatted final answer

    We replicate this 3-call structure here. If bodhi-llm exposes a
    public `BodhiWrapper` class you can replace the body of this function
    with a real wrapper call — the timing harness stays the same.
    """
    BODHI_SYSTEM_PROMPT = (
        "You are a careful, epistemically humble medical assistant. "
        "Before answering, consider: (1) how confident you are, "
        "(2) whether you should abstain or ask a clarifying question, "
        "(3) whether your answer is appropriately hedged. "
        "Be honest about uncertainty. Do not overclaim."
    )

    CALIBRATION_TEMPLATE = (
        "Given your previous response, rate your confidence from 0.0 to 1.0. "
        "Reply with only a JSON object: {{\"confidence\": <float>}}"
    )

    FINAL_TEMPLATE = (
        "Now produce your final BOHDI-formatted response. "
        "Include appropriate hedges and, if uncertain, suggest the user "
        "consult a healthcare professional."
    )

    device = next(model.parameters()).device.type
    results = []

    for prompt in tqdm(prompts, desc="with_wrapper"):
        total_wall = 0.0
        total_prompt_tokens = 0
        total_response_tokens = 0
        n_calls = 0

        # Call 1 – BOHDI system prompt + user prompt
        full_prompt_1 = f"{BODHI_SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        p1, r1, w1 = _generate_once(model, tokenizer, full_prompt_1, max_new_tokens, device)
        total_wall += w1
        total_prompt_tokens += p1
        total_response_tokens += r1
        n_calls += 1

        # Call 2 – calibration / self-rating
        p2, r2, w2 = _generate_once(model, tokenizer, CALIBRATION_TEMPLATE, 32, device)
        total_wall += w2
        total_prompt_tokens += p2
        total_response_tokens += r2
        n_calls += 1

        # Call 3 – final formatted response
        p3, r3, w3 = _generate_once(model, tokenizer, FINAL_TEMPLATE, max_new_tokens, device)
        total_wall += w3
        total_prompt_tokens += p3
        total_response_tokens += r3
        n_calls += 1

        results.append(
            {
                "prompt_tokens": total_prompt_tokens,
                "response_tokens": total_response_tokens,
                "wall_clock_s": total_wall,
                "model_calls": n_calls,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(per_prompt: list[dict], n_gpus: int = N_GPUS) -> dict:
    n = len(per_prompt)
    total_wall = sum(r["wall_clock_s"] for r in per_prompt)
    total_response_tokens = sum(r["response_tokens"] for r in per_prompt)

    avg_wall = total_wall / n
    tokens_per_sec = total_response_tokens / total_wall if total_wall > 0 else 0.0

    # GPU-hours per 1000 prompts: scale observed total to 1000-prompt baseline
    gpu_hours_per_1000 = (total_wall / 3600) * n_gpus / n * 1000

    return {
        "n_prompts": n,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "avg_wall_clock_per_response_s": round(avg_wall, 4),
        "total_wall_clock_s": round(total_wall, 2),
        "gpu_hours_per_1000_prompts": round(gpu_hours_per_1000, 4),
        "avg_model_calls_per_prompt": round(
            sum(r["model_calls"] for r in per_prompt) / n, 2
        ),
        "avg_response_tokens": round(total_response_tokens / n, 1),
    }


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------


def load_existing_output(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_output(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.config.startswith("lora_") and args.lora_path is None:
        print(f"[error] --lora-path is required for config '{args.config}'", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)

    print(f"=== bench_inference.py | config: {args.config} ===")
    print(f"Model:      {args.model}")
    print(f"LoRA path:  {args.lora_path or '(none)'}")
    print(f"N prompts:  {args.n_prompts}")
    print(f"Seed:       {args.seed}")

    # Load holdout prompts
    prompts = load_prompts(args.sample_ids, args.healthbench_data, args.n_prompts)
    print(f"Loaded {len(prompts)} prompts from holdout set.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.lora_path, args.config)

    # Run benchmark
    use_wrapper = args.config.endswith("_wrapper")
    if use_wrapper:
        per_prompt = run_with_wrapper(model, tokenizer, prompts, args.max_new_tokens)
    else:
        per_prompt = run_no_wrapper(model, tokenizer, prompts, args.max_new_tokens)

    agg = aggregate(per_prompt)
    print(f"\n--- Results for {args.config} ---")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    # Merge into output file (so you can run each config separately)
    existing = load_existing_output(args.output)

    # Top-level metadata (written once, overwritten on each run to stay fresh)
    existing["_meta"] = {
        "hardware": os.environ.get("BENCH_HARDWARE", "H100"),
        "git_sha": get_git_sha(),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if "configs" not in existing:
        existing["configs"] = {}

    existing["configs"][args.config] = {
        **agg,
        "per_prompt": per_prompt,  # full trace for post-hoc analysis
    }

    save_output(args.output, existing)


if __name__ == "__main__":
    main()
