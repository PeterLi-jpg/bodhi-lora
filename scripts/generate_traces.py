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
from _bodhi_ablation import make_ablated_bodhi_config

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


def make_bodhi_wrapper(engine, ablate_component: str = "none"):
    """Set up BODHI wrapper once, reuse across examples.

    ``ablate_component`` is one of the values accepted by
    ``make_ablated_bodhi_config``. ``"none"`` is the production wrapper
    (medical domain, all components active).
    """
    from bodhi import BODHI
    chat_fn = lambda msgs: engine.chat(msgs)
    config = make_ablated_bodhi_config(ablate_component)
    return BODHI(chat_function=chat_fn, config=config)


def generate_response(engine, messages, use_bodhi, bodhi_wrapper=None,
                      ablate_component: str = "none"):
    """Return {content, analysis, metadata}. analysis/metadata are None for
    non-BODHI runs so callers get a stable schema.

    Per Sebastian: saving analysis + metadata lets us audit *why* the model
    decided what it did, not just what it said — critical for finding where
    the humility wrapper went wrong on specific examples.

    ``ablate_component`` is stamped into the returned metadata dict so each
    row carries its own provenance (paper figure-3 ablation, issue #69).
    """
    if not use_bodhi:
        return {
            "content": engine.chat(messages),
            "analysis": None,
            "metadata": {"ablate_component": ablate_component},
        }
    resp = bodhi_wrapper.complete(messages)
    metadata = dict(resp.metadata) if resp.metadata else {}
    metadata["ablate_component"] = ablate_component
    return {
        "content": resp.content,
        "analysis": resp.analysis,
        "metadata": metadata,
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
    parser.add_argument(
        "--ablate-component",
        choices=["none", "no_calibration", "no_questions",
                 "no_abstention", "no_domain_framing"],
        default="none",
        help="Disable one BODHI wrapper component for the paper figure-3 "
             "ablation (issue #69). Only meaningful when --use-bodhi is set. "
             "'none' (default) = production wrapper.",
    )
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
        prev_ablate = set()
        with open(args.resume_from) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_ids.add(row["prompt_id"])
                    if "model" in row:
                        prev_models.add(row["model"])
                    if "bodhi" in row:
                        prev_bodhi.add(row["bodhi"])
                    if "ablate_component" in row:
                        prev_ablate.add(row["ablate_component"])
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupt lines from interrupted runs

        model_mismatch = prev_models and prev_models != {args.model}
        bodhi_mismatch = prev_bodhi and prev_bodhi != {args.use_bodhi}
        # Mixing ablation tags would invalidate the figure-3 ablation (issue
        # #69): different rows would have been generated with different
        # wrapper components disabled. Treat it like the model/bodhi mismatch.
        ablate_mismatch = prev_ablate and prev_ablate != {args.ablate_component}
        if model_mismatch or bodhi_mismatch or ablate_mismatch:
            msg = (
                f"resume config mismatch — refusing to append to {args.resume_from}\n"
                f"  existing rows: model={prev_models or '{unknown}'} "
                f"bodhi={prev_bodhi or '{unknown}'} "
                f"ablate={prev_ablate or '{unknown}'}\n"
                f"  this run:      model={{{args.model!r}}} "
                f"bodhi={{{args.use_bodhi}}} "
                f"ablate={{{args.ablate_component!r}}}\n"
                f"  re-running with different settings would silently mix "
                f"outputs (issue #6/7, #69)\n"
                f"  if you truly want to append across settings, pass --force-resume"
            )
            if not args.force_resume:
                raise SystemExit(msg)
            print(f"WARNING: {msg}\n(continuing because --force-resume was passed)")

        examples = [ex for ex in examples if ex["prompt_id"] not in done_ids]
        print(f"Resuming, skipping {len(done_ids)} already done "
              f"(prev model={prev_models or '?'} bodhi={prev_bodhi or '?'} "
              f"ablate={prev_ablate or '?'})")

    print(f"\nGenerating {len(examples)} traces, bodhi={args.use_bodhi}, "
          f"ablate={args.ablate_component}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done_ids else "w"
    ok, fail = 0, 0

    # Short-circuit: if the resume file already has at least max_examples
    # traces, we have enough — don't bother spinning up vLLM.  Booting and
    # tearing down a 27B vLLM container holds the TPU for ~5 min and used
    # to break the next stage's vLLM init too.
    #
    # Two cases that count as "have enough":
    #   - len(examples) == 0 after resume filter (no prompts left to do)
    #   - len(done_ids) >= max_examples (already at target, even if a
    #     reshuffled prompt order leaves some extras to do)
    if len(examples) == 0 or (args.max_examples and len(done_ids) >= args.max_examples):
        print(f"Nothing to generate (have {len(done_ids)} traces, need "
              f"{args.max_examples or 'all'}). Skipping vLLM startup.")
        return

    # Concurrent inference: vLLM batches concurrent requests server-side via
    # PagedAttention + continuous batching, so feeding the scheduler more
    # in-flight requests lets it pack each batch fuller and amortizes
    # prefill cost across more decode steps.
    #
    # The previous default (16) was severely under-subscribed for a 27B
    # model on v6e-8 (256 GB HBM, ~200 GB free for KV after weights): the
    # vLLM scheduler's max-num-seqs is 128+ on this footprint, so clamping
    # the client to 16 left most scheduler slots empty. Bumping the default
    # to 32 doubles in-flight requests while staying well under the server
    # ceiling so first-run hardware behavior is predictable. See
    # https://blog.vllm.ai/2024/09/05/perf-update.html for the underlying
    # PagedAttention + continuous-batching behavior.
    #
    # Override per hardware:
    #   GEN_CONCURRENCY=32   v6e-8   (default)
    #   GEN_CONCURRENCY=64   v6e-8   stretch (verify HBM headroom in vllm startup log)
    #   GEN_CONCURRENCY=128  v6e-16  (more HBM, more KV-cache slots)
    #   GEN_CONCURRENCY=8    fallback if you hit HBM OOM at startup
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    GEN_CONCURRENCY = int(os.environ.get("GEN_CONCURRENCY", "32"))

    with VLLMEngine(args.model) as engine:
        bodhi_wrapper = (
            make_bodhi_wrapper(engine, ablate_component=args.ablate_component)
            if args.use_bodhi else None
        )
        # Lock around the JSONL append: workers can finish in any order, but
        # f.write() / f.flush() must not interleave or we get corrupt lines.
        write_lock = threading.Lock()

        def _gen_one(ex):
            try:
                out = generate_response(
                    engine, ex["prompt"], args.use_bodhi, bodhi_wrapper,
                    ablate_component=args.ablate_component,
                )
                return {
                    "prompt_id": ex["prompt_id"],
                    "messages": ex["prompt"],
                    "response": out["content"],
                    "bodhi_analysis": out["analysis"],
                    "bodhi_metadata": out["metadata"],
                    "tags": ex.get("example_tags", []),
                    "source_dataset": ex.get("_source", "unknown"),
                    "model": args.model,
                    "bodhi": args.use_bodhi,
                    # Top-level for the resume-from consistency check.
                    # Also lives inside bodhi_metadata for analysis tools.
                    "ablate_component": args.ablate_component,
                }, None
            except Exception as e:
                # Per-task try/except: a single bad prompt or transient HTTP
                # error from vLLM should not kill the whole run.  Collect the
                # error and keep going.
                return None, (ex.get("prompt_id", "?"), repr(e))

        with open(out_path, mode) as f:
            with ThreadPoolExecutor(max_workers=GEN_CONCURRENCY) as ex_pool:
                futures = [ex_pool.submit(_gen_one, ex) for ex in examples]
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    trace, err = fut.result()
                    if trace is not None:
                        with write_lock:
                            f.write(json.dumps(trace) + "\n")
                            f.flush()
                        ok += 1
                    else:
                        pid, err_repr = err
                        print(f"  Error on {pid}: {err_repr}")
                        fail += 1

    print(f"\nDone: {ok} ok, {fail} failed -> {out_path}")


if __name__ == "__main__":
    main()
