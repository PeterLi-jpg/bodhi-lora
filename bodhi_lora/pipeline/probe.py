"""Stage probe: contamination of the base model by the benchmark (issue #72).

    python -m bodhi_lora.pipeline.probe --cell <cell id> [--stub-engine]

The base model gets the first prefix_tokens tokens of a prompt and continues it
greedily. A prompt counts as memorised when the continuation equals the prompt's next
continuation_tokens tokens. The probe compares that rate on the cell's eval pool with
the rate on control text from another benchmark (the cell's probe.control), by a
two-sided Fisher exact test. Output: shared/probe/probe.json.
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from bodhi_lora import manifest as mf
from bodhi_lora.data import load_rows
from bodhi_lora.engine import make_engine
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser

INTERPRETATION = ("p < 0.05 with a benchmark rate well above the control rate is evidence that "
                  "the benchmark prompts were in the base model's pretraining data.")


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
    """The user turns of a prompt, joined by newlines, so a multi-turn prompt
    flattens to one string for tokenization."""
    return "\n".join(
        m["content"] for m in prompt_messages if m.get("role") == "user"
    )


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

    Per-task try/except, so a single HTTP failure does not end the whole set.
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


def run(ctx: StageContext) -> dict:
    from transformers import AutoTokenizer, set_seed

    probe = ctx.cell.probe
    if probe is None:
        raise SystemExit(f"{ctx.cell.id} has no probe block")
    out_dir = ctx.paths.shared("probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    random.seed(probe.seed)
    np.random.seed(probe.seed)
    set_seed(probe.seed)
    rng = random.Random(probe.seed)
    rows = ctx.rows_by_id()
    pool = sorted(ctx.pools().eval)
    control_bench = ctx.reg.benchmarks[probe.control]
    control = load_rows(control_bench, ctx.project.data_root, fetch=False)
    for name, items in (("eval pool", pool), ("control", control)):
        if len(items) < probe.n_prompts:
            raise SystemExit(f"the {name} has {len(items)} rows; the probe needs {probe.n_prompts}")
    bench_prompts = [(pid, extract_user_prompt_text(rows[pid].messages))
                     for pid in rng.sample(pool, probe.n_prompts)]
    control_prompts = [(r.prompt_id, extract_user_prompt_text(r.messages))
                       for r in rng.sample(control, probe.n_prompts)]
    tokenizer = AutoTokenizer.from_pretrained(ctx.model.hf_id, revision=ctx.model.revision,
                                              **ctx.model.chat_template.tokenizer_kwargs)
    engine = make_engine(ctx.model, serving=probe.serving, stub=ctx.stub,
                         log_path=ctx.log_path("vllm_probe.log"))
    with engine as eng:
        bench_result = run_probe_set(ctx.bench.id, bench_prompts, tokenizer, eng, probe.prefix_tokens,
                                     probe.continuation_tokens, probe.serving.concurrency)
        control_result = run_probe_set(probe.control, control_prompts, tokenizer, eng, probe.prefix_tokens,
                                       probe.continuation_tokens, probe.serving.concurrency)
    b_hits = sum(1 for s in bench_result["samples"] if s["exact_match"])
    c_hits = sum(1 for s in control_result["samples"] if s["exact_match"])
    p_value, test_used = fisher_or_z(b_hits, bench_result["n"], c_hits, control_result["n"])
    out = {"benchmark_id": ctx.bench.id, "control_id": probe.control, "model": ctx.model.hf_id,
           "benchmark": bench_result, "control": control_result, "fisher_p_value": p_value,
           "test_used": test_used, "settings": probe.model_dump(), "interpretation": INTERPRETATION}
    path = out_dir / "probe.json"
    path.write_text(json.dumps(out, indent=2))
    counts = {"benchmark_hits": b_hits, "control_hits": c_hits, "n_per_set": probe.n_prompts,
              "fisher_p_value": p_value}
    ctx.write_manifest(out_dir, "probe", timer=timer, models={"base": ctx.model},
                       engines={"base": engine.settings()}, outputs=[path], counts=counts,
                       extra={"status": "complete"})
    print(f"probe: {b_hits}/{bench_result['n']} benchmark, {c_hits}/{control_result['n']} control, "
          f"p={p_value:.4g} ({test_used}) -> {path}", flush=True)
    return counts


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0]).parse_args(argv)
    run(context_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
