"""Pretraining contamination probe (issue #72).

Feeds the base model the first N tokens of HealthBench Hard prompts and
measures exact-completion / prefix-overlap rates against a PubMedQA
medical-text baseline. If HealthBench Hard was in MedGemma-27B's
pretraining corpus, we expect the HB rate to be significantly higher
than the PubMedQA rate (Fisher exact test).

Usage (TPU/GPU pod, after merge):

    python scripts/contamination_probe.py \
        --model google/medgemma-27b-text-it \
        --n-prompts 100 \
        --output eval/contamination_probe.json

CPU box: --help and py_compile only. Actual probe needs 27B inference.
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
from transformers import AutoTokenizer, set_seed

# Same import dance as eval_healthbench.py: scripts/ on path for the
# bare _vllm_engine import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_engine import VLLMEngine  # noqa: E402

HEALTHBENCH_HARD_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "hard_2025-05-08-21-00-10.jsonl"
)


# ── pure helpers (unit-testable) ────────────────────────────────────────────

def normalize_for_exact_match(text: str) -> str:
    """Lowercase + collapse whitespace. Used for the exact_match metric."""
    return " ".join(text.lower().split())


def compute_prefix_overlap_tokens(toks_a, toks_b) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = 0
    for a, b in zip(toks_a, toks_b):
        if a != b:
            break
        n += 1
    return n


def extract_user_prompt_text(prompt_messages) -> str:
    """Concatenate user-role content from a HealthBench prompt list.

    HealthBench rows store `prompt` as [{role, content}, ...]. We pull
    just the user turns and join with newlines so multi-turn prompts
    flatten to a single string for tokenization.
    """
    return "\n".join(
        m["content"] for m in prompt_messages if m.get("role") == "user"
    )


# ── data loaders ────────────────────────────────────────────────────────────

def load_healthbench(path: Path) -> list:
    """Load HealthBench Hard jsonl, downloading if missing."""
    if not path.exists():
        print(f"Downloading HealthBench Hard to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_pubmedqa(cache_path: Path) -> list:
    """Load PubMedQA `pqa_artificial` questions, caching to a local jsonl.

    On a network-enabled box we hydrate from HF `bigbio/pubmed_qa`, then
    write a thin jsonl of {"question", "_source"} so subsequent runs are
    offline. If both the HF load and the cache are unavailable, raise a
    clear error pointing at the cache path.
    """
    if cache_path.exists():
        rows = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            f"datasets package missing AND no cache at {cache_path}. "
            "Install datasets>=2.18.0 or pre-populate the cache."
        ) from e

    try:
        ds = load_dataset("bigbio/pubmed_qa", "pqa_artificial", split="train")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load bigbio/pubmed_qa AND no cache at {cache_path}. "
            "On a network-enabled box, run this script once to populate the "
            "cache, then copy data/raw/pubmedqa_baseline.jsonl onto the "
            "offline pod. Original error: " + repr(e)
        ) from e

    rows = [{"question": ex["question"], "_source": "pubmed_qa"} for ex in ds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Cached {len(rows)} PubMedQA questions -> {cache_path}")
    return rows


# ── stats ───────────────────────────────────────────────────────────────────

def two_proportion_z_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided z-test for proportions. Fallback when scipy is missing.

    Uses the pooled-variance form:
        z = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    """
    from math import erf, sqrt
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    denom = p * (1 - p) * (1 / n1 + 1 / n2)
    if denom <= 0:
        # No variance: both rates are 0 or both are 1, no signal.
        return 1.0
    z = (p1 - p2) / sqrt(denom)
    # Two-sided p-value from the standard-normal CDF.
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


def fisher_or_z(hb_hits: int, hb_n: int, pq_hits: int, pq_n: int):
    """Run Fisher exact if scipy is available; else fall back to z-test."""
    try:
        from scipy.stats import fisher_exact
        # 2x2: rows = dataset, cols = (match, no_match)
        table = [
            [hb_hits, hb_n - hb_hits],
            [pq_hits, pq_n - pq_hits],
        ]
        _, p = fisher_exact(table, alternative="two-sided")
        return float(p), "fisher_exact"
    except ImportError:
        p = two_proportion_z_pvalue(hb_hits, hb_n, pq_hits, pq_n)
        return p, "two_proportion_z"


# ── core probe ──────────────────────────────────────────────────────────────

def probe_one(prompt_text: str, prompt_id, tokenizer, engine, prefix_tokens: int,
              continuation_tokens: int):
    """Run a single contamination probe.

    Returns (sample_dict, skip_reason). If skip_reason is set, the example
    was too short and sample_dict is None.
    """
    full_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(full_ids) < prefix_tokens + continuation_tokens:
        return None, "too_short"

    prefix_ids = full_ids[:prefix_tokens]
    gt_ids = full_ids[prefix_tokens:prefix_tokens + continuation_tokens]
    leak_prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    gt_continuation = tokenizer.decode(gt_ids, skip_special_tokens=True)

    model_output = engine.chat(
        [{"role": "user", "content": leak_prefix}],
        max_new_tokens=continuation_tokens,
        temperature=0.0,
    )
    model_ids = tokenizer.encode(model_output, add_special_tokens=False)

    exact_match = (
        normalize_for_exact_match(model_output)
        == normalize_for_exact_match(gt_continuation)
    )
    overlap = compute_prefix_overlap_tokens(model_ids, gt_ids)

    return {
        "prompt_id_or_index": prompt_id,
        "leak_prefix": leak_prefix,
        "ground_truth_continuation": gt_continuation,
        "model_output": model_output,
        "exact_match": exact_match,
        "prefix_overlap_tokens": overlap,
    }, None


def aggregate(samples: list, n_skipped: int) -> dict:
    if not samples:
        return {
            "n": 0,
            "n_skipped": n_skipped,
            "exact_match_rate": None,
            "mean_prefix_overlap_tokens": None,
            "samples": [],
        }
    n = len(samples)
    exact_hits = sum(1 for s in samples if s["exact_match"])
    return {
        "n": n,
        "n_skipped": n_skipped,
        "exact_match_rate": exact_hits / n,
        "mean_prefix_overlap_tokens": float(
            np.mean([s["prefix_overlap_tokens"] for s in samples])
        ),
        "samples": samples,
    }


def run_probe_set(name: str, prompts: list, tokenizer, engine, prefix_tokens: int,
                  continuation_tokens: int, concurrency: int) -> dict:
    """Probe a list of (prompt_id, prompt_text) tuples concurrently.

    Per-task try/except so a single HTTP failure doesn't kill the whole set,
    same pattern as eval_healthbench.py.
    """
    samples = []
    n_skipped = 0
    failed = []

    def _one(item):
        prompt_id, text = item
        try:
            return probe_one(
                text, prompt_id, tokenizer, engine,
                prefix_tokens, continuation_tokens,
            )
        except Exception as e:
            return None, repr(e)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, item) for item in prompts]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=name):
            sample, reason = fut.result()
            if sample is not None:
                samples.append(sample)
            elif reason == "too_short":
                n_skipped += 1
            else:
                failed.append(reason)

    if failed:
        print(f"WARNING [{name}]: {len(failed)} probe failures (skipped):")
        for err in failed[:5]:
            print(f"  {err}")

    print(f"  {name}: probed={len(samples)}  skipped_too_short={n_skipped}  "
          f"failed={len(failed)}")
    return aggregate(samples, n_skipped)


# ── main ───────────────────────────────────────────────────────────────────

INTERPRETATION = (
    "p<0.05 with HB rate >> baseline rate is evidence of HealthBench Hard "
    "contamination in MedGemma-27B pretraining."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--n-prompts", type=int, default=100,
                        help="Per-set sample size (HB and PubMedQA each).")
    parser.add_argument("--prefix-tokens", type=int, default=30)
    parser.add_argument("--continuation-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/contamination_probe.json")
    parser.add_argument(
        "--healthbench-data",
        default="data/raw/healthbench_hard.jsonl",
        help="Auto-downloads if missing.",
    )
    parser.add_argument(
        "--pubmedqa-data",
        default="data/raw/pubmedqa_baseline.jsonl",
        help="Cache file; auto-populated from bigbio/pubmed_qa on first run.",
    )
    args = parser.parse_args()

    # Same seed setup as generate_traces.py: covers random + numpy +
    # transformers (vLLM is greedy here, but tokenizer init still benefits).
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    rng = random.Random(args.seed)

    hb_path = Path(args.healthbench_data)
    pq_path = Path(args.pubmedqa_data)

    hb_all = load_healthbench(hb_path)
    pq_all = load_pubmedqa(pq_path)
    print(f"Loaded {len(hb_all)} HealthBench Hard rows and "
          f"{len(pq_all)} PubMedQA rows.")

    if len(hb_all) < args.n_prompts:
        raise RuntimeError(
            f"HealthBench has only {len(hb_all)} rows; need {args.n_prompts}."
        )
    if len(pq_all) < args.n_prompts:
        raise RuntimeError(
            f"PubMedQA has only {len(pq_all)} rows; need {args.n_prompts}."
        )

    hb_sample = rng.sample(hb_all, args.n_prompts)
    pq_sample = rng.sample(pq_all, args.n_prompts)

    hb_prompts = [
        (ex.get("prompt_id", i), extract_user_prompt_text(ex["prompt"]))
        for i, ex in enumerate(hb_sample)
    ]
    pq_prompts = [(i, ex["question"]) for i, ex in enumerate(pq_sample)]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eval_concurrency = int(os.environ.get("EVAL_CONCURRENCY", "16"))

    # Single base-model engine, no LoRA, no wrapper.
    with VLLMEngine(args.model) as engine:
        hb_result = run_probe_set(
            "healthbench", hb_prompts, tokenizer, engine,
            args.prefix_tokens, args.continuation_tokens, eval_concurrency,
        )
        pq_result = run_probe_set(
            "pubmedqa", pq_prompts, tokenizer, engine,
            args.prefix_tokens, args.continuation_tokens, eval_concurrency,
        )

    hb_hits = sum(1 for s in hb_result["samples"] if s["exact_match"])
    pq_hits = sum(1 for s in pq_result["samples"] if s["exact_match"])
    p_value, test_used = fisher_or_z(
        hb_hits, hb_result["n"], pq_hits, pq_result["n"]
    )

    summary = {
        "healthbench": hb_result,
        "pubmedqa": pq_result,
        "fisher_p_value": p_value,
        "test_used": test_used,
        "model": args.model,
        "seed": args.seed,
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "n_prompts_per_set": args.n_prompts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": INTERPRETATION,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    hb_rate = hb_result["exact_match_rate"]
    pq_rate = pq_result["exact_match_rate"]
    hb_rate_str = f"{hb_rate:.4f}" if hb_rate is not None else "n/a"
    pq_rate_str = f"{pq_rate:.4f}" if pq_rate is not None else "n/a"
    print(
        f"\nHB exact_match_rate={hb_rate_str}  "
        f"PubMedQA exact_match_rate={pq_rate_str}  "
        f"p={p_value:.4g} ({test_used})  -> {out}"
    )


if __name__ == "__main__":
    main()
