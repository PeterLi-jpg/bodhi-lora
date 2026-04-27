"""Evaluate model on HealthBench Hard (base vs lora, with/without BOHDI wrapper)."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
import random
import subprocess
import urllib.request

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import set_seed

# Add scripts/ dir for bare-name imports and project root for package imports.
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _vllm_engine import VLLMEngine
from scripts.filter_traces import GRADER_TEMPLATE, LocalGrader, parse_json_response, grade_trace

HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
DATA_DIR = Path("data/raw")


def load_eval_data(sample_ids_path):
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
    """Build a reusable BODHI wrapper backed by a vLLM engine."""
    from bodhi import BODHI, BODHIConfig
    chat_fn = lambda msgs: engine.chat(msgs)
    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def gen_response(engine: VLLMEngine, messages, use_bodhi, bodhi_wrapper=None, max_new_tokens=1024):
    """Return (response_text, token_logprobs).

    token_logprobs is a list of per-output-token log-probs from the generation
    call itself (logprobs=True on vLLM).  For the BODHI path we can't intercept
    the internal calls, so logprobs are None there.
    """
    if not use_bodhi:
        text, token_logprobs = engine.chat_with_logprobs(messages, max_new_tokens)
        return text, token_logprobs

    resp = bodhi_wrapper.complete(messages)
    return resp.content, None


def score_response_confidence(token_logprobs):
    """Derive confidence metrics from per-output-token log-probs.

    token_logprobs is the list returned by VLLMEngine.chat_with_logprobs —
    piggybacked from the generation call itself, so no extra forward pass needed.
    Returns None fields when logprobs are unavailable (e.g. BODHI path).
    """
    if not token_logprobs:
        return {
            "response_token_count": 0,
            "mean_token_logprob": None,
            "geomean_token_prob": None,
        }
    mean_logprob = float(np.mean(token_logprobs))
    return {
        "response_token_count": len(token_logprobs),
        "mean_token_logprob": mean_logprob,
        "geomean_token_prob": float(np.exp(mean_logprob)),
    }


# -- confidence / calibration-style metrics --

def _collect_binary_labels(results, confidence_key):
    """Expand per-example confidence into per-criterion binary labels."""
    y_true = []
    y_pred = []
    for result in results:
        confidence = result.get(confidence_key)
        if confidence is None:
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        for criterion in result["criteria_results"]:
            # Positive-point criteria are the only ones that behave like
            # correctness labels; negative-point criteria are penalties.
            if criterion["points"] > 0:
                y_true.append(1.0 if criterion["criteria_met"] else 0.0)
                y_pred.append(confidence)
    return y_true, y_pred


def compute_brier_score(results, confidence_key):
    """Brier score across positive-point rubric criteria."""
    y_true, y_pred = _collect_binary_labels(results, confidence_key)
    if not y_true:
        return None
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(results, confidence_key, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    y_true, y_pred = _collect_binary_labels(results, confidence_key)
    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        if lo == 0.0:
            mask = mask | (y_pred == 0.0)
        n = mask.sum()
        if n == 0:
            continue
        avg_conf = y_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n / len(y_true)) * abs(avg_acc - avg_conf)
    return float(ece)


def _safe_package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_run_metadata(seed, grader_model):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _safe_git_sha(),
        "seed": seed,
        "grader_model": grader_model,
        "bodhi_version": _safe_package_version("bodhi-llm"),
        "transformers_version": _safe_package_version("transformers"),
        "peft_version": _safe_package_version("peft"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--grader-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Greedy decoding is deterministic; seed covers BODHI internals + grader sampling.
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    examples = load_eval_data(args.sample_ids)
    if args.max_examples:
        examples = examples[:args.max_examples]

    tag = f"{'lora' if args.lora_path else 'base'}_{'bodhi' if args.use_bodhi else 'no_wrapper'}"
    print(f"\nEval: {tag}  ({len(examples)} examples)\n")

    # ── Pass 1: generate responses with the inference engine ─────────────────
    # Concurrent generation — vLLM batches concurrent requests server-side.
    # 16 is conservative; we saw vLLM crash at 32 on Qwen-14B grading.
    # Per-task try/except so one HTTP error doesn't kill all 200 examples.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    EVAL_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "16"))

    raw_generations = []  # list of {prompt_id, messages, rubrics, response, token_logprobs}
    failed_inference = []
    with VLLMEngine(args.model, lora_path=args.lora_path) as engine:
        bodhi_wrapper = make_bodhi_wrapper(engine) if args.use_bodhi else None

        def _gen_one(ex):
            try:
                resp, token_logprobs = gen_response(
                    engine, ex["prompt"], args.use_bodhi, bodhi_wrapper
                )
                return {
                    "prompt_id": ex["prompt_id"],
                    "messages": ex["prompt"],
                    "rubrics": ex["rubrics"],
                    "response": resp,
                    "token_logprobs": token_logprobs,
                }, None
            except Exception as e:
                return None, (ex.get("prompt_id", "?"), repr(e))

        with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex_pool:
            futures = [ex_pool.submit(_gen_one, ex) for ex in examples]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{tag} [inference]"):
                gen, err = fut.result()
                if gen is not None:
                    raw_generations.append(gen)
                else:
                    failed_inference.append(err)

    if failed_inference:
        print(f"WARNING: {len(failed_inference)} inference failures (skipped):")
        for pid, err in failed_inference[:5]:
            print(f"  {pid}: {err}")
    print(f"Generated {len(raw_generations)} responses; starting grader engine...")

    # Drain TPU before spinning up the grader vLLM. Without a small sleep,
    # the grader engine hits "Engine core initialization failed" because the
    # inference container's TPU resources haven't fully released yet.
    import time as _t
    print("  draining TPU 30s before grader startup...", flush=True)
    _t.sleep(30)

    # ── Pass 2: grade with the grader engine ──────────────────────────────────
    all_results = []
    scores = []
    model_confidences = []
    total_parse_failures = 0
    total_rubric_items = 0
    failed_grading = []
    with VLLMEngine(args.grader_model) as grader_engine:
        grader = LocalGrader(grader_engine)

        def _grade_one(item):
            try:
                confidence = score_response_confidence(item["token_logprobs"])
                grade = grade_trace(grader, item["messages"], item["response"], item["rubrics"])
                return {
                    "prompt_id": item["prompt_id"], "response": item["response"],
                    "score": grade["overall_score"], "tag_scores": grade["tag_scores"],
                    "criteria_results": grade["criteria_results"],
                    "parse_failures": grade["parse_failures"],
                    "model_confidence_geomean_prob": confidence["geomean_token_prob"],
                    "model_confidence_mean_token_logprob": confidence["mean_token_logprob"],
                    "response_token_count": confidence["response_token_count"],
                }, None
            except Exception as e:
                return None, (item.get("prompt_id", "?"), repr(e))

        with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex_pool:
            futures = [ex_pool.submit(_grade_one, item) for item in raw_generations]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{tag} [grading]"):
                result, err = fut.result()
                if result is not None:
                    all_results.append(result)
                    scores.append(result["score"])
                    if result["model_confidence_geomean_prob"] is not None:
                        model_confidences.append(result["model_confidence_geomean_prob"])
                    total_parse_failures += result["parse_failures"]
                    total_rubric_items += len(result["criteria_results"])
                else:
                    failed_grading.append(err)

    if failed_grading:
        print(f"WARNING: {len(failed_grading)} grading failures (skipped):")
        for pid, err in failed_grading[:5]:
            print(f"  {pid}: {err}")

    model_brier = compute_brier_score(all_results, "model_confidence_geomean_prob")
    model_ece = compute_ece(all_results, "model_confidence_geomean_prob")
    grader_brier = compute_brier_score(all_results, "score")
    grader_ece = compute_ece(all_results, "score")
    parse_fail_rate = (total_parse_failures / total_rubric_items) if total_rubric_items else None

    summary = {
        "config": tag, "model": args.model,
        "lora_path": args.lora_path, "use_bodhi": args.use_bodhi,
        "n_examples": len(examples),
        "run_metadata": collect_run_metadata(args.seed, args.grader_model),
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "model_confidence_method": (
            "geometric_mean_token_probability over the emitted response, "
            "computed from next-token logprobs conditioned on the prompt"
        ),
        "mean_model_confidence": (
            float(np.mean(model_confidences)) if model_confidences else None
        ),
        "brier_model_calibration": model_brier,
        "ece_model_calibration": model_ece,
        # Keep the legacy grader-derived fields so older comparisons do not
        # silently break, but label them explicitly as grader consistency.
        "brier_grader_consistency": grader_brier,
        "ece_grader_consistency": grader_ece,
        "grader_parse_failure_rate": parse_fail_rate,
        "grader_parse_failures_total": total_parse_failures,
        "grader_rubric_items_total": total_rubric_items,
        "results": all_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    model_brier_str = f"{model_brier:.4f}" if model_brier is not None else "n/a"
    model_ece_str = f"{model_ece:.4f}" if model_ece is not None else "n/a"
    grader_brier_str = f"{grader_brier:.4f}" if grader_brier is not None else "n/a"
    grader_ece_str = f"{grader_ece:.4f}" if grader_ece is not None else "n/a"
    mean_str = f"{summary['mean']:.4f}" if summary['mean'] is not None else "n/a"
    std_str = f"{summary['std']:.4f}" if summary['std'] is not None else "n/a"
    fail_str = f"{parse_fail_rate*100:.2f}%" if parse_fail_rate is not None else "n/a"
    print(f"\n{tag}: score={mean_str} +/- {std_str}  "
          f"model_brier={model_brier_str}  model_ece={model_ece_str}  "
          f"grader_brier*={grader_brier_str}  grader_ece*={grader_ece_str}  "
          f"grader_parse_fail={fail_str}  -> {out}")
    print("  (* grader_brier / grader_ece are legacy grader-consistency proxies)")


if __name__ == "__main__":
    main()
