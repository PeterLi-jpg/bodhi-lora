"""Generate BOHDI wrapper traces over HealthBench for SFT training data."""

import argparse
import json
import os
import random
import traceback
from pathlib import Path

import sys
import os
# Insert scripts/ dir so _vllm_engine can be imported as a bare module name,
# regardless of CWD or whether the project root is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from _vllm_engine import VLLMEngine

DATA_DIR = Path("data/raw")

DATASET_URLS = {
    "healthbench": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def ensure_downloaded(name):
    """Download dataset to data/raw/ if not already there."""
    import urllib.request
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"Downloading {name}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URLS[name], path)
    return path


def load_healthbench(name):
    path = ensure_downloaded(name)
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ex["_source"] = name
            examples.append(ex)
    print(f"  {name}: {len(examples)} examples")
    return examples


def load_multiple_datasets(names):
    all_ex = []
    seen = set()
    for name in names:
        for ex in load_healthbench(name):
            if ex["prompt_id"] not in seen:
                seen.add(ex["prompt_id"])
                all_ex.append(ex)
    print(f"Total unique: {len(all_ex)}")
    return all_ex


def load_exclude_ids(paths):
    """Collect prompt_ids to exclude, from one or more files.

    Accepts:
      - .json  containing a list of ids, or {"prompt_ids": [...]}
      - .jsonl containing one HealthBench example per line (reads
        ``prompt_id`` from each). Useful for excluding an entire dataset,
        e.g. ``data/raw/healthbench_hard.jsonl`` to drop all 1000 Hard
        prompts for the HealthBench-only generalization experiment.

    Accepts a string (one path) or a list of paths.
    """
    if isinstance(paths, str):
        paths = [paths]
    ids = set()
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    ids.add(ex["prompt_id"])
        else:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data["prompt_ids"]
            ids.update(data)
    return ids


def make_bodhi_wrapper(engine):
    """Set up BODHI wrapper once, reuse across examples."""
    from bodhi import BODHI, BODHIConfig
    chat_fn = lambda msgs: engine.chat(msgs)
    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def generate_response(engine, messages, use_bodhi, bodhi_wrapper=None):
    """Return {content, analysis, metadata}. analysis/metadata are None for
    non-BODHI runs so callers get a stable schema.

    Per Sebastian: saving analysis + metadata lets us audit *why* the model
    decided what it did, not just what it said — critical for finding where
    the humility wrapper went wrong on specific examples.
    """
    if not use_bodhi:
        return {"content": engine.chat(messages), "analysis": None, "metadata": None}
    resp = bodhi_wrapper.complete(messages)
    return {
        "content": resp.content,
        "analysis": resp.analysis,
        "metadata": resp.metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--datasets", nargs="+", default=["healthbench_hard", "healthbench"],
                        choices=list(DATASET_URLS.keys()))
    parser.add_argument("--exclude-ids", nargs="+", default=None,
                        help="one or more files listing prompt_ids to skip. "
                             "Accepts .json (list or {prompt_ids: [...]}) and "
                             ".jsonl (reads prompt_id from each row). For the "
                             "HealthBench-only generalization experiment, pass "
                             "data/raw/healthbench_hard.jsonl here to drop all "
                             "1000 Hard prompts from training.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-resume", action="store_true",
                        help="skip the model/bodhi metadata consistency check on resume "
                             "(issue #6/7). Only use when you intentionally want to "
                             "append rows generated with different settings.")
    args = parser.parse_args()

    # Greedy decoding is deterministic without a seed, but the BODHI wrapper
    # may use sampling internally (prompt shuffling, tie-breaking) — seed so
    # reruns on identical hardware produce identical raw_traces.jsonl.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    examples = load_multiple_datasets(args.datasets)

    if args.exclude_ids:
        exclude = load_exclude_ids(args.exclude_ids)
        before = len(examples)
        examples = [ex for ex in examples if ex["prompt_id"] not in exclude]
        print(f"Excluded {before - len(examples)} eval examples, {len(examples)} left")

    if args.max_examples:
        examples = examples[:args.max_examples]

    done_ids = set()
    if args.resume_from and Path(args.resume_from).exists():
        # Issue #6/7: validate that existing rows were generated with the
        # same model + bodhi setting. Mixing settings silently corrupts the
        # training corpus. Refuse to resume on mismatch unless --force-resume.
        prev_models = set()
        prev_bodhi = set()
        with open(args.resume_from) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_ids.add(row["prompt_id"])
                    if "model" in row:
                        prev_models.add(row["model"])
                    if "bodhi" in row:
                        prev_bodhi.add(row["bodhi"])
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupt lines from interrupted runs

        model_mismatch = prev_models and prev_models != {args.model}
        bodhi_mismatch = prev_bodhi and prev_bodhi != {args.use_bodhi}
        if model_mismatch or bodhi_mismatch:
            msg = (
                f"resume config mismatch — refusing to append to {args.resume_from}\n"
                f"  existing rows: model={prev_models or '{unknown}'} "
                f"bodhi={prev_bodhi or '{unknown}'}\n"
                f"  this run:      model={{{args.model!r}}} bodhi={{{args.use_bodhi}}}\n"
                f"  re-running with different settings would silently mix "
                f"outputs (issue #6/7)\n"
                f"  if you truly want to append across settings, pass --force-resume"
            )
            if not args.force_resume:
                raise SystemExit(msg)
            print(f"WARNING: {msg}\n(continuing because --force-resume was passed)")

        examples = [ex for ex in examples if ex["prompt_id"] not in done_ids]
        print(f"Resuming, skipping {len(done_ids)} already done "
              f"(prev model={prev_models or '?'} bodhi={prev_bodhi or '?'})")

    print(f"\nGenerating {len(examples)} traces, bodhi={args.use_bodhi}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done_ids else "w"
    ok, fail = 0, 0

    with VLLMEngine(args.model) as engine:
        bodhi_wrapper = make_bodhi_wrapper(engine) if args.use_bodhi else None
        with open(out_path, mode) as f:
            for ex in tqdm(examples):
                try:
                    out = generate_response(engine, ex["prompt"], args.use_bodhi, bodhi_wrapper)
                    trace = {
                        "prompt_id": ex["prompt_id"],
                        "messages": ex["prompt"],
                        "response": out["content"],
                        "bodhi_analysis": out["analysis"],
                        "bodhi_metadata": out["metadata"],
                        "tags": ex.get("example_tags", []),
                        "source_dataset": ex.get("_source", "unknown"),
                        "model": args.model,
                        "bodhi": args.use_bodhi,
                    }
                    f.write(json.dumps(trace) + "\n")
                    f.flush()
                    ok += 1
                except Exception as e:
                    # Full traceback helps distinguish OOM from tokenizer/BODHI bugs
                    # when a 48h run has a few failures we want to diagnose later.
                    traceback.print_exc()
                    print(f"  Error on {ex['prompt_id']}: {e}")
                    fail += 1

    print(f"\nDone: {ok} ok, {fail} failed -> {out_path}")


if __name__ == "__main__":
    main()
