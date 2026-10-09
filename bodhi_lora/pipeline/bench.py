"""Stage bench: measured time and tokens per reply, for every condition of a seed.

    python -m bodhi_lora.pipeline.bench --cell <cell id> --seed N [--stub-engine]

The first bench.n_prompts of the seed's eval ids (in a fixed shuffle) are answered one
at a time, each condition on its own engine, so a latency is the service time of one
reply and not of a batch. Run it on an otherwise idle GPU: a shared node's queueing
would enter the times. Output: seed_<n>/bench/bench.json with per-reply seconds, calls
and generated tokens (from the server's token counts), and their summaries.
"""
from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser
from bodhi_lora.protocols import build, decoding_for


def summarize(seconds: list[float], tokens: list[int]) -> dict:
    arr = np.array(seconds)
    total = float(arr.sum()) if len(arr) else 0.0
    return {
        "n": len(seconds), "seconds": seconds, "completion_tokens": tokens,
        "median_s": float(np.median(arr)) if len(arr) else None,
        "p90_s": float(np.percentile(arr, 90)) if len(arr) else None,
        "p99_s": float(np.percentile(arr, 99)) if len(arr) else None,
        "throughput_tok_per_s": sum(tokens) / total if total > 0 else None,
    }


def run(ctx: StageContext, seed: int) -> dict:
    b = ctx.cell.bench
    if b is None:
        raise SystemExit(f"{ctx.cell.id} has no bench block")
    out_dir = ctx.paths.stage(seed, "bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    ids = list(ctx.eval_ids(seed))
    random.Random(seed).shuffle(ids)
    rows = ctx.rows_by_id()
    prompts = [rows[pid].messages for pid in ids[:b.n_prompts]]
    out = {"cell": ctx.cell.id, "seed": seed, "n_prompts": len(prompts), "settings": b.model_dump(),
           "conditions": {}}
    engines = {}
    for condition in ctx.cell.eval.conditions:
        weights, proto_rec = ctx.condition_protocol(condition)
        proto = build(proto_rec, ctx.model)
        decoding = decoding_for(proto, max_new_tokens=b.max_new_tokens,
                                max_new_tokens_per_pass=b.max_new_tokens)
        adapter = ctx.adapter_dir(seed) if weights == "lora" else None
        engine = make_engine(ctx.model, serving=b.serving, adapter=adapter, quantize=ctx.quantized_base(),
                             stub=ctx.stub, enforce_eager=b.enforce_eager,
                             log_path=ctx.log_path(f"vllm_bench_s{seed}.log"))
        seconds, tokens, calls = [], [], []
        with engine as eng:
            for messages in prompts:
                t0 = time.perf_counter()
                o = proto.respond(eng, messages, decoding)
                seconds.append(time.perf_counter() - t0)
                tokens.append(o.completion_tokens)
                calls.append(o.calls)
        out["conditions"][condition] = {**summarize(seconds, tokens), "calls": calls}
        engines[condition] = engine.settings()
        print(f"bench: {condition} median {out['conditions'][condition]['median_s']}", flush=True)
    path = out_dir / "bench.json"
    path.write_text(json.dumps(out, indent=2))
    ctx.write_manifest(out_dir, "bench", seed=seed, timer=timer, models={"base": ctx.model},
                       engines=engines, outputs=[path], extra={"status": "complete"})
    return {c: v["median_s"] for c, v in out["conditions"].items()}


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0], seed="required").parse_args(argv)
    run(context_from_args(args), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
