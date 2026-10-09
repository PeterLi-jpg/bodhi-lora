"""Stage filter: grade the teacher traces against the rubric, keep those above tau, and
split the kept traces into train and val.

    python -m bodhi_lora.pipeline.filter --cell <cell id> [--stub-engine]   # grade (GPU)
    python -m bodhi_lora.pipeline.filter --cell <cell id> --seed N           # split (CPU)

Grading runs once per cell, with the filter grader, into shared/filter/graded.jsonl.
Only traces of prompts in the cell's train pool are graded, so a held-out prompt can
never reach training through a stale trace file (issue #60). A rerun carries existing
grades forward and grades only what is missing.

tau follows the cell's filter.tau rule, applied to every graded trace:
  fixed                  tau = value
  percentile             the percentile rank rule below
  fixed_then_percentile  value, then the percentile when the train split would hold less
                         than min_kept_share of the graded traces
  none                   keep every graded trace (tau is not applied)
  random_same_count      a seeded random sample as large as the fixed rule keeps
The value used and how it was chosen go into shared/filter/manifest.json.

The split shuffles the kept traces, sorted by prompt id first so the result does not
depend on the order the grades finished in, and puts val_ratio of them in val. With
split_policy once it runs once with split_seed into shared/filter/; with per_seed each
seed shuffles with its own seed into seed_<n>/filter/.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.grading.rubric import Grader, grade_rubric
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser

# A percentile of fewer graded traces than this is too noisy to set tau, and so few
# grades usually mean the grader failed.
MIN_GRADED_FOR_PERCENTILE = 100
SCORE_FIELD = "overall_score"
# The stub grader says every criterion is met, so a stub score is pos / (pos - neg) of
# the row's rubric: above the usual tau, so stub runs keep their traces.
STUB_GRADE_REPLY = '```json\n{"explanation": "stub", "criteria_met": true}\n```'


def percentile_tau(scores, percentile):
    """tau at a percentile of the graded scores, by the rank rule the 2026-07 launcher used.

    The rule is: sorted score at index int(p / 100 * n) - 1, minus 1e-9, rounded to 12
    decimals. It is kept exactly, and not replaced by numpy.percentile, because numpy's
    "lower" method takes index floor(p / 100 * (n - 1)). For some n that is a different
    trace, so a rerun would keep a different set.

    The 1e-9 nudge keeps every trace tied at the threshold score. Synthesized-rubric
    scores have large mass points (for example 8/21), and a tau that lands just above
    one would drop the whole spike.
    """
    sc = sorted(s for s in scores if isinstance(s, (int, float)))
    if not sc:
        raise ValueError("no graded scores to take a percentile of")
    return float(f"{sc[max(0, int(percentile / 100 * len(sc)) - 1)] - 1e-9:.12f}")


def n_val_of(n_kept: int, val_ratio: float) -> int:
    """Val rows of a split: at least one when two or more traces are kept."""
    return max(1, int(n_kept * val_ratio)) if n_kept > 1 else 0


def apply_tau(graded: list[dict], tau_rule, val_ratio: float, n_traces: int) -> tuple[list[dict], dict]:
    """The kept traces and the record of how tau was chosen.

    n_traces is the number of traces the filter was given (in the train pool); the
    fixed_then_percentile rule compares the train split with it, as the 2026-07
    launcher compared train.jsonl with the raw trace file.
    """
    scores = [g["grade"][SCORE_FIELD] for g in graded]

    def at(tau):
        return [g for g in graded if g["grade"][SCORE_FIELD] >= tau]

    def percentile():
        if len(scores) < MIN_GRADED_FOR_PERCENTILE:
            raise SystemExit(f"only {len(scores)} graded traces; a percentile tau needs "
                             f"{MIN_GRADED_FOR_PERCENTILE}. Grading probably failed.")
        return percentile_tau(scores, tau_rule.percentile)

    record = {"rule": tau_rule.rule, "value": tau_rule.value, "percentile": tau_rule.percentile,
              "min_kept_share": tau_rule.min_kept_share, "score_field": SCORE_FIELD}
    if tau_rule.rule == "fixed":
        tau, kept = tau_rule.value, at(tau_rule.value)
    elif tau_rule.rule == "percentile":
        tau = percentile()
        kept = at(tau)
    elif tau_rule.rule == "fixed_then_percentile":
        kept = at(tau_rule.value)
        n_train = len(kept) - n_val_of(len(kept), val_ratio)
        record["train_share_at_value"] = n_train / n_traces if n_traces else None
        record["triggered"] = bool(n_traces) and n_train / n_traces < tau_rule.min_kept_share
        tau = percentile() if record["triggered"] else tau_rule.value
        kept = at(tau)
    elif tau_rule.rule == "none":
        tau, kept = None, list(graded)
    else:  # random_same_count
        k = len(at(tau_rule.value))
        pool = sorted(graded, key=lambda g: g["prompt_id"])
        tau, kept = None, random.Random(tau_rule.seed).sample(pool, k)
    record["tau"] = tau
    if scores:
        record["score_summary"] = {"min": min(scores), "median": statistics.median(scores),
                                   "mean": sum(scores) / len(scores), "max": max(scores)}
    return kept, record


def split_kept(kept: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(kept, key=lambda g: g["prompt_id"])
    random.Random(seed).shuffle(ordered)
    n_val = n_val_of(len(ordered), val_ratio)
    return ordered[n_val:], ordered[:n_val]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def grade(ctx: StageContext) -> dict:
    """Grade every trace of a train-pool prompt; write graded.jsonl, tau and the kept set."""
    flt = ctx.cell.filter
    out_dir = ctx.paths.shared("filter")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    traces_path = ctx.traces_file()
    traces = read_jsonl(traces_path)
    if not traces:
        raise SystemExit(f"{traces_path} has no traces; run the traces stage first")
    pool = set(ctx.pools().train)
    in_pool = [t for t in traces if t["prompt_id"] in pool]
    rows = ctx.rows_by_id()
    graded_path = out_dir / "graded.jsonl"
    graded = {g["prompt_id"]: g for g in (read_jsonl(graded_path) if graded_path.is_file() else [])
              if g["prompt_id"] in pool and "grade" in g}
    todo = [t for t in in_pool if t["prompt_id"] not in graded]
    print(f"filter: {len(traces)} traces, {len(in_pool)} in the train pool, "
          f"{len(graded)} already graded, {len(todo)} to grade", flush=True)
    grader_model = ctx.reg.models[ctx.cell.grading.filter]
    engine = make_engine(grader_model, serving=flt.serving, stub=ctx.stub,
                         stub_reply=STUB_GRADE_REPLY, log_path=ctx.log_path("vllm_filter.log"))
    failed = []
    if todo:
        with engine as eng, open(graded_path, "a") as f:
            grader = Grader(eng)

            def one(trace):
                try:
                    rubric = rows[trace["prompt_id"]].rubrics
                    return {**trace, "grade": grade_rubric(grader, trace["messages"], trace["response"],
                                                           rubric)}, None
                except Exception:
                    return None, (trace["prompt_id"],
                                  "\n".join(traceback.format_exc().strip().splitlines()[-30:]))

            with ThreadPoolExecutor(max_workers=flt.serving.concurrency) as ex:
                futures = [ex.submit(one, t) for t in todo]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="grading"):
                    row, err = fut.result()
                    if row is None:
                        failed.append(err)
                        continue
                    # Appended as they finish, so a crash keeps every finished grade.
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    graded[row["prompt_id"]] = row
    for pid, tb in failed[:5]:
        print(f"  failed {pid}:\n{tb}", file=sys.stderr)
    graded_rows = sorted(graded.values(), key=lambda g: g["prompt_id"])
    write_jsonl(graded_path, graded_rows)
    parse_failures = sum(g["grade"]["parse_failures"] for g in graded_rows)
    items = sum(len(g["grade"]["criteria_results"]) for g in graded_rows)
    if items and parse_failures > 0.05 * items:
        print(f"WARNING: grader parse failures {parse_failures}/{items} exceed 5%", file=sys.stderr)
    kept, tau_record = apply_tau(graded_rows, flt.tau, flt.val_ratio, len(in_pool))
    write_jsonl(out_dir / "kept.jsonl", sorted(kept, key=lambda g: g["prompt_id"]))
    outputs = [graded_path, out_dir / "kept.jsonl"]
    counts = {"n_traces": len(traces), "n_in_train_pool": len(in_pool),
              "n_outside_train_pool": len(traces) - len(in_pool), "n_graded": len(graded_rows),
              "n_failed_grading": len(failed), "n_kept": len(kept),
              "kept_of_graded": len(kept) / len(graded_rows) if graded_rows else None,
              "grader_parse_failures_total": parse_failures, "grader_rubric_items_total": items}
    if flt.split_policy == "once":
        train, val = split_kept(kept, flt.val_ratio, flt.split_seed)
        write_jsonl(out_dir / "train.jsonl", train)
        write_jsonl(out_dir / "val.jsonl", val)
        outputs += [out_dir / "train.jsonl", out_dir / "val.jsonl"]
        counts.update(n_train=len(train), n_val=len(val))
    status = "incomplete" if failed else "complete"
    ctx.write_manifest(out_dir, "filter", timer=timer, models={"grader": grader_model},
                       config={"filter": flt.model_dump()}, engines={"grader": engine.settings()},
                       inputs=[traces_path], outputs=outputs, counts=counts,
                       extra={"status": status, "tau": tau_record})
    print(f"filter: tau {tau_record['tau']} ({flt.tau.rule}), kept {len(kept)}/{len(graded_rows)}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} traces failed grading; run the stage again to retry them")
    return counts


def split(ctx: StageContext, seed: int) -> dict:
    """The seed's train and val files: its own shuffle, or a pointer to the shared split."""
    flt = ctx.cell.filter
    shared = ctx.paths.shared("filter")
    out_dir = ctx.paths.stage(seed, "filter")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    if flt.split_policy == "once":
        files = [shared / "train.jsonl", shared / "val.jsonl"]
        ctx.write_manifest(out_dir, "filter", seed=seed, timer=timer, inputs=files,
                           extra={"status": "complete", "shared": str(shared),
                                  "shared_manifest_sha256": mf.sha256_file(shared / "manifest.json")})
        return {"shared": str(shared)}
    kept = read_jsonl(shared / "kept.jsonl")
    train, val = split_kept(kept, flt.val_ratio, seed)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    counts = {"n_kept": len(kept), "n_train": len(train), "n_val": len(val)}
    ctx.write_manifest(out_dir, "filter", seed=seed, timer=timer, inputs=[shared / "kept.jsonl"],
                       outputs=[out_dir / "train.jsonl", out_dir / "val.jsonl"], counts=counts,
                       extra={"status": "complete", "split_seed": seed})
    return counts


def split_files(ctx: StageContext, seed: int) -> tuple[Path, Path]:
    """The train and val files the seed trains on."""
    folder = ctx.paths.shared("filter") if ctx.cell.filter.split_policy == "once" else ctx.paths.stage(seed, "filter")
    return folder / "train.jsonl", folder / "val.jsonl"


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0], seed="optional").parse_args(argv)
    ctx = context_from_args(args)
    if args.seed is None:
        grade(ctx)
    else:
        split(ctx, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
