#!/usr/bin/env python3
"""
scripts/bench_inference.py  —  issue #29

The paper claims: distilling the BOHDI wrapper into LoRA weights removes the
wrapper at serve time, making inference faster and cheaper. This script
produces the actual numbers that back that claim (Paper Table 3).

Four configs are timed on the 200-prompt HealthBench Hard holdout:
  base_no_wrapper  — baseline single-call inference
  base_wrapper     — baseline + BOHDI wrapper (~3 model calls/prompt)
  lora_no_wrapper  — LoRA-merged weights, single call (should match base speed)
  lora_wrapper     — LoRA-merged weights + wrapper

Reported per config: tokens/sec, wall-clock per response, GPU-hours per 1000 prompts.
Run once per config; results merge into the same eval/latency.json.

Usage:
  python scripts/bench_inference.py \
      --config base_no_wrapper \
      --model google/medgemma-27b-text-it \
      --sample-ids data/raw/hard_200_sample_ids.json \
      --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
      --output eval/latency.json

  python scripts/bench_inference.py \
      --config lora_no_wrapper \
      --lora-path checkpoints/best \
      [... same flags ...]
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

VALID_CONFIGS = ["base_no_wrapper", "base_wrapper", "lora_no_wrapper", "lora_wrapper"]

# spec: 512-token responses, single H100, 200-prompt holdout.
TARGET_RESPONSE_TOKENS = 512
N_GPUS = 1


# args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency benchmark for Paper Table 3 (issue #29)")
    p.add_argument("--config", required=True, choices=VALID_CONFIGS)
    p.add_argument("--model", default="google/medgemma-27b-text-it")
    p.add_argument("--lora-path", default=None, help="Required for lora_* configs.")
    p.add_argument("--sample-ids", required=True, help="200-prompt holdout ID file.")
    p.add_argument("--healthbench-data", nargs="+", required=True)
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=TARGET_RESPONSE_TOKENS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="eval/latency.json")
    return p.parse_args()


# reproductivity

def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# data

def load_prompts(sample_ids_path: str, hb_data_paths: list[str], n: int) -> list[str]:
    """Load up to n prompts that match the 200-prompt holdout IDs."""
    with open(sample_ids_path) as f:
        holdout_ids: set[str] = set(json.load(f))

    prompts: list[str] = []
    for path in hb_data_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                sid = str(row.get("sample_id") or row.get("id", ""))
                if sid in holdout_ids:
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
        print(f"[warn] Only {len(prompts)} holdout prompts found (wanted {n}).", file=sys.stderr)
    return prompts[:n]


# model loading

def load_model_and_tokenizer(model_id: str, lora_path: str | None, config: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    if config.startswith("lora_"):
        if lora_path is None:
            raise ValueError("--lora-path is required for lora_* configs.")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        # Merge adapter into base weights so lora_no_wrapper has identical
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# inference

def _generate_once(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> tuple[int, int, float]:
    """Single forward pass. Returns (prompt_tokens, response_tokens, wall_seconds)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic, comparable timings
            pad_token_id=tokenizer.pad_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return prompt_len, output_ids.shape[1] - prompt_len, wall


def run_no_wrapper(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """Single model call per prompt — baseline and lora_no_wrapper both use this."""
    device = next(model.parameters()).device.type
    results = []
    for prompt in tqdm(prompts, desc="no_wrapper"):
        p, r, w = _generate_once(model, tokenizer, prompt, max_new_tokens, device)
        results.append({"prompt_tokens": p, "response_tokens": r, "wall_clock_s": w, "model_calls": 1})
    return results


def run_with_wrapper(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """
    3-call BOHDI wrapper pattern per prompt:
      1. System prompt + user query -> initial response
      2. Self-calibration -> confidence score
      3. Final BOHDI-formatted answer

    This is what makes base_wrapper and lora_wrapper ~3x slower than their
    no-wrapper counterparts. If bodhi-llm exposes a public BodhiWrapper class,
    replace the three _generate_once calls with the real wrapper — the timing
    harness around them stays the same.
    """
    BODHI_SYSTEM = (
        "You are a careful, epistemically humble medical assistant. "
        "Before answering, consider: (1) how confident you are, "
        "(2) whether you should abstain or ask a clarifying question, "
        "(3) whether your answer is appropriately hedged. "
        "Be honest about uncertainty. Do not overclaim."
    )
    CALIBRATION = (
        "Given your previous response, rate your confidence from 0.0 to 1.0. "
        'Reply with only a JSON object: {"confidence": <float>}'
    )
    FINAL = (
        "Now produce your final BOHDI-formatted response. "
        "Include appropriate hedges and, if uncertain, suggest the user consult a healthcare professional."
    )

    device = next(model.parameters()).device.type
    results = []

    for prompt in tqdm(prompts, desc="with_wrapper"):
        p1, r1, w1 = _generate_once(model, tokenizer, f"{BODHI_SYSTEM}\n\nUser: {prompt}\nAssistant:", max_new_tokens, device)
        p2, r2, w2 = _generate_once(model, tokenizer, CALIBRATION, 32, device)
        p3, r3, w3 = _generate_once(model, tokenizer, FINAL, max_new_tokens, device)

        results.append({
            "prompt_tokens": p1 + p2 + p3,
            "response_tokens": r1 + r2 + r3,
            "wall_clock_s": w1 + w2 + w3,
            "model_calls": 3,
        })

    return results


# aggregation

def aggregate(per_prompt: list[dict]) -> dict:
    n = len(per_prompt)
    total_wall = sum(r["wall_clock_s"] for r in per_prompt)
    total_resp_tokens = sum(r["response_tokens"] for r in per_prompt)

    # tokens/sec
    tokens_per_sec = total_resp_tokens / total_wall if total_wall > 0 else 0.0

    # scale from observed n prompts to 1000-prompt baseline
    gpu_hours_per_1000 = (total_wall / 3600) * N_GPUS / n * 1000

    return {
        "n_prompts": n,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "avg_wall_clock_per_response_s": round(total_wall / n, 4),
        "total_wall_clock_s": round(total_wall, 2),
        "gpu_hours_per_1000_prompts": round(gpu_hours_per_1000, 4),
        "avg_model_calls_per_prompt": round(sum(r["model_calls"] for r in per_prompt) / n, 2),
        "avg_response_tokens": round(total_resp_tokens / n, 1),
    }

# output: merges each config run into one eval/latency.json

def load_existing(path: str) -> dict:
    p = Path(path)
    return json.load(open(p)) if p.exists() else {}


def save_output(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w"), indent=2)
    print(f"Results written to {path}")


# main

def main() -> None:
    args = parse_args()

    if args.config.startswith("lora_") and args.lora_path is None:
        print(f"[error] --lora-path required for '{args.config}'", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)
    print(f"=== bench_inference.py | config: {args.config} | n={args.n_prompts} | seed={args.seed} ===")

    prompts = load_prompts(args.sample_ids, args.healthbench_data, args.n_prompts)
    model, tokenizer = load_model_and_tokenizer(args.model, args.lora_path, args.config)

    if args.config.endswith("_wrapper"):
        per_prompt = run_with_wrapper(model, tokenizer, prompts, args.max_new_tokens)
    else:
        per_prompt = run_no_wrapper(model, tokenizer, prompts, args.max_new_tokens)

    agg = aggregate(per_prompt)
    print(f"\n--- {args.config} ---")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    existing = load_existing(args.output)
    existing["_meta"] = {
        "hardware": os.environ.get("BENCH_HARDWARE", "H100"),
        "git_sha": get_git_sha(),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    existing.setdefault("configs", {})[args.config] = {**agg, "per_prompt": per_prompt}
    save_output(args.output, existing)


if __name__ == "__main__":
    main()