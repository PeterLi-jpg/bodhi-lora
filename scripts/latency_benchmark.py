"""Latency benchmark across base/lora x with/without BOHDI wrapper.

Measures end-to-end latency for a fixed batch of HealthBench Hard prompts,
running each of 4 configs through its own vLLM engine and recording per-
prompt wall-clock time plus output-token count.  The four configs are:

    base_no_wrapper   model only, raw chat
    base_bodhi        model wrapped in BODHI
    lora_no_wrapper   model + LoRA adapter, raw chat
    lora_bodhi        model + LoRA adapter, BODHI wrapper

Default is `--enforce-eager` for ALL four configs to keep the comparison
apples-to-apples.  Pass `--no-enforce-eager` to compare graph-mode
throughput, but be aware that LoRA + CUDA-graph capture currently takes
~130 min before the first prompt, so disabling eager is only useful once
that capture-time regression is fixed upstream.

Usage on a GPU pod:
    python scripts/latency_benchmark.py \
        --model google/medgemma-27b-text-it \
        --lora-path checkpoints/seed_42/best \
        --sample-ids data/raw/hard_200_sample_ids.json \
        --n-prompts 50 \
        --output eval/latency.json
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Mirror the import shim used by other scripts so `_vllm_engine` resolves
# regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _vllm_engine import VLLMEngine

HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
DATA_DIR = Path("data/raw")


def load_eval_data(sample_ids_path):
    """Mirror of eval_healthbench.load_eval_data — kept inline so we don't
    pull in eval_healthbench's grader/torch deps just to read prompts."""
    path = DATA_DIR / "healthbench_hard.jsonl"
    if not path.exists():
        print("Downloading HealthBench Hard...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    with open(sample_ids_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    eval_ids = set(data)

    filtered = [ex for ex in examples if ex["prompt_id"] in eval_ids]
    print(f"{len(filtered)} eval examples loaded")
    return filtered


def make_bodhi_wrapper(engine: VLLMEngine):
    """Build a reusable BODHI wrapper backed by a vLLM engine.

    Imported lazily so `--help` works without the bodhi package installed.
    """
    from bodhi import BODHI, BODHIConfig
    chat_fn = lambda msgs: engine.chat(msgs)
    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def _count_output_tokens(text: str) -> int:
    """Approximate output-token count via whitespace splitting.

    Avoids pulling in transformers just to tokenize; the response-length
    estimate is good enough for throughput-style summaries and the raw
    list of latencies / output-text lengths is preserved in the JSON for
    re-analysis with a real tokenizer if needed.
    """
    return len(text.split())


def _run_one_config(
    config_name: str,
    model: str,
    lora_path,
    use_bodhi: bool,
    prompts,
    enforce_eager: bool,
    max_new_tokens: int,
):
    """Spin up one vLLM engine, time each prompt, return per-prompt records.

    We deliberately use ThreadPoolExecutor(max_workers=1) instead of a plain
    `for` loop so the timing context is identical across configs and easy
    to swap in a higher concurrency for throughput benchmarks later.
    Concurrent timing would distort per-prompt latency because vLLM batches
    requests server-side.
    """
    print(f"\n=== {config_name} ===", flush=True)
    latencies_s = []
    output_tokens = []
    with VLLMEngine(model, lora_path=lora_path, enforce_eager=enforce_eager) as engine:
        bodhi_wrapper = make_bodhi_wrapper(engine) if use_bodhi else None

        def _gen_one(messages):
            t0 = time.perf_counter()
            if use_bodhi:
                resp = bodhi_wrapper.complete(messages)
                text = resp.content
            else:
                text = engine.chat(messages, max_new_tokens=max_new_tokens)
            elapsed_s = time.perf_counter() - t0
            return elapsed_s, text

        # Sequential timing — max_workers=1 keeps requests strictly serial so
        # each latency reflects single-prompt service time, not batched
        # throughput.  Concurrent runs would pollute the latency distribution
        # because vLLM coalesces in-flight requests.
        with ThreadPoolExecutor(max_workers=1) as pool:
            for elapsed_s, text in pool.map(_gen_one, prompts):
                latencies_s.append(elapsed_s)
                output_tokens.append(_count_output_tokens(text))

    n = len(latencies_s)
    arr = np.array(latencies_s)
    total_tokens = sum(output_tokens)
    total_time = float(arr.sum()) if n else 0.0
    summary = {
        "n": n,
        "latencies_s": latencies_s,
        "output_tokens": output_tokens,
        "median_s": float(np.median(arr)) if n else None,
        "p50_s": float(np.percentile(arr, 50)) if n else None,
        "p90_s": float(np.percentile(arr, 90)) if n else None,
        "p99_s": float(np.percentile(arr, 99)) if n else None,
        "throughput_tok_per_s": (total_tokens / total_time) if total_time > 0 else None,
    }
    median_str = f"{summary['median_s']:.2f}s" if summary['median_s'] is not None else "n/a"
    p90_str = f"{summary['p90_s']:.2f}s" if summary['p90_s'] is not None else "n/a"
    tput = summary["throughput_tok_per_s"]
    tput_str = f"{tput:.2f} tok/s" if tput is not None else "n/a"
    print(
        f"  {config_name}: n={n} median={median_str} p90={p90_str} throughput={tput_str}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument(
        "--sample-ids",
        default="data/raw/hard_200_sample_ids.json",
    )
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --enforce-eager to vLLM (default).  Use --no-enforce-eager "
            "to compare graph-mode throughput.  Note: with --enable-lora, "
            "graph-mode capture currently takes ~130 min before the first "
            "prompt is served on A100."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="eval/latency.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    examples = load_eval_data(args.sample_ids)
    # Stable sample of n prompts via seeded shuffle so re-runs hit the
    # same prompt set (otherwise sort order in the JSONL would dominate).
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    examples = examples[: args.n_prompts]
    prompts = [ex["prompt"] for ex in examples]
    print(f"Benchmarking {len(prompts)} prompts (enforce_eager={args.enforce_eager})")

    configs = [
        ("base_no_wrapper", None,           False),
        ("base_bodhi",      None,           True),
        ("lora_no_wrapper", args.lora_path, False),
        ("lora_bodhi",      args.lora_path, True),
    ]

    out = {
        "enforce_eager": args.enforce_eager,
        "model": args.model,
        "lora_path": args.lora_path,
        "n_prompts": len(prompts),
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for name, lora_path, use_bodhi in configs:
        if name.startswith("lora_") and lora_path is None:
            print(f"Skipping {name}: --lora-path not provided", flush=True)
            continue
        out[name] = _run_one_config(
            name, args.model, lora_path, use_bodhi,
            prompts, args.enforce_eager, args.max_new_tokens,
        )
        # Drain accelerator state between engines.  Same 30 s pause used in
        # eval_healthbench.py before swapping inference->grader engine —
        # avoids "Engine core initialization failed" when the previous
        # container's TPU/GPU resources haven't fully released yet.
        print("  draining accelerator 30s before next config...", flush=True)
        time.sleep(30)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
