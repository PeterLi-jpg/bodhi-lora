"""Stage traces: the teacher's replies to the train pool, under the cell's protocol.

    python -m bodhi_lora.pipeline.traces --cell <cell id> [--stub-engine]

The teacher is the cell's base model with its protocol. It answers the train pool in
pool order, up to traces.max_prompts prompts, and every reply goes to
shared/traces/traces.jsonl, because no training seed changes the traces. A cell with
traces_from trains on another cell's traces and generates none.

The stage resumes: prompts already in the file are skipped, and a file written with
another model or protocol is refused, so a corpus never mixes settings (issues #6, #7).
Each line keeps the reply, the protocol's internal text for audit, and the reply's
calls, tokens and whether a call hit its token limit.
"""
from __future__ import annotations

import json
import random
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser
from bodhi_lora.protocols import build, decoding_for

# The stub teacher's reply carries the ask-first markers, so that protocol's parser runs too.
STUB_TEACHER_REPLY = "<plan>stub plan</plan>\nREPLY:\nSTUB RESPONSE"


def read_existing(path: Path) -> tuple[dict, int]:
    """Traces already written, by prompt id, and the count of unreadable lines.

    An interrupted run can leave a cut-off last line; it is skipped and counted.
    """
    done, bad = {}, 0
    if not path.is_file():
        return done, bad
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done[row["prompt_id"]] = row
            except (json.JSONDecodeError, KeyError):
                bad += 1
    return done, bad


def check_resume(done: dict, model_repo: str, protocol_id: str, ablate: str) -> None:
    """Refuse to append to traces made with another model, protocol or ablation."""
    seen = {(r.get("model"), r.get("protocol"), r.get("ablate_component", "none")) for r in done.values()}
    want = (model_repo, protocol_id, ablate)
    if seen and seen != {want}:
        raise SystemExit(f"traces in the file were made with {sorted(seen)}, this run is {want}; "
                         "mixing them would corrupt the corpus. Move the file away to start over.")


def run(ctx: StageContext) -> dict:
    out_dir = ctx.paths.shared("traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    if ctx.cell.traces_from:
        src = ctx.traces_file()
        if not src.is_file():
            raise SystemExit(f"{src} is missing: run the traces stage of {ctx.cell.traces_from} first")
        ctx.write_manifest(out_dir, "traces", timer=timer, inputs=[src],
                           extra={"status": "complete", "traces_from": ctx.cell.traces_from,
                                  "traces_file": str(src)})
        return {"traces_from": ctx.cell.traces_from}

    t = ctx.cell.traces
    rows = ctx.rows_by_id()
    pool = ctx.pools().train[:t.max_prompts]
    out_path = out_dir / "traces.jsonl"
    proto = build(ctx.protocol, ctx.model)
    ablate = getattr(proto, "ablate", "none")
    done, n_bad = read_existing(out_path)
    check_resume(done, ctx.model.hf_id, ctx.protocol.id, ablate)
    todo = [pid for pid in pool if pid not in done]
    print(f"traces: {len(pool)} prompts in the pool cap, {len(done)} done, {len(todo)} to go"
          f"{f', {n_bad} unreadable lines skipped' if n_bad else ''}", flush=True)
    decoding = decoding_for(proto, max_new_tokens=t.max_new_tokens_per_pass,
                            max_new_tokens_per_pass=t.max_new_tokens_per_pass,
                            temperature=t.temperature, top_p=t.top_p,
                            seed=t.seed if t.decoding == "sample" else None)
    # The protocol code may sample (prompt order, tie breaks); seed it for reruns.
    random.seed(t.seed)

    ok, failed = 0, []
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "finish_length": 0}
    engine = make_engine(ctx.model, serving=t.serving, stub=ctx.stub, stub_reply=STUB_TEACHER_REPLY,
                         log_path=ctx.log_path("vllm_traces.log"))
    if todo:
        lock = threading.Lock()
        with engine as eng, open(out_path, "a") as f:
            def one(pid):
                try:
                    row = rows[pid]
                    o = proto.respond(eng, row.messages, decoding)
                    return {
                        "prompt_id": pid, "messages": row.messages, "response": o.response,
                        "bodhi_analysis": o.analysis, "bodhi_metadata": o.metadata,
                        "tags": row.tags, "source_dataset": ctx.bench.id,
                        "model": ctx.model.hf_id, "model_id": ctx.model.id,
                        "bodhi": ctx.protocol.impl == "bodhi_llm", "protocol": ctx.protocol.id,
                        "ablate_component": ablate, "calls": o.calls,
                        "prompt_tokens_total": o.prompt_tokens,
                        "completion_tokens_total": o.completion_tokens,
                        "finish_length": o.finish_length,
                    }, None
                except Exception:
                    # One failed prompt (a timeout, a bad reply) must not stop the run.
                    return None, (pid, "\n".join(traceback.format_exc().strip().splitlines()[-30:]))

            with ThreadPoolExecutor(max_workers=t.serving.concurrency) as pool_ex:
                futures = [pool_ex.submit(one, pid) for pid in todo]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="traces"):
                    trace, err = fut.result()
                    if trace is None:
                        failed.append(err)
                        continue
                    with lock:
                        f.write(json.dumps(trace) + "\n")
                        f.flush()
                    ok += 1
                    totals["calls"] += trace["calls"]
                    totals["prompt_tokens"] += trace["prompt_tokens_total"]
                    totals["completion_tokens"] += trace["completion_tokens_total"]
                    totals["finish_length"] += trace["finish_length"]
    for pid, tb in failed[:5]:
        print(f"  failed {pid}:\n{tb}", file=sys.stderr)
    counts = {"n_pool_cap": len(pool), "n_resumed": len(done), "n_ok": ok, "n_failed": len(failed),
              "n_unreadable_lines": n_bad, "tokens": totals}
    ctx.write_manifest(out_dir, "traces", timer=timer, models={"teacher": ctx.model},
                       config={"traces": t.model_dump(), "protocol": ctx.protocol.model_dump()},
                       engines={"teacher": engine.settings()}, outputs=[out_path], counts=counts,
                       extra={"status": "incomplete" if failed else "complete"})
    print(f"traces: {ok} written, {len(failed)} failed -> {out_path}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} prompts failed; run the stage again to retry them")
    return counts


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0]).parse_args(argv)
    run(context_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
