"""Stage eval: the replies of each condition to a seed's eval prompts, graded on the rubric.

    python -m bodhi_lora.pipeline.eval --cell <cell id> --seeds 42,7 [--stub-engine]
    python -m bodhi_lora.pipeline.eval --cell <cell id> --seed N --condition <condition>
                                       [--stub-engine]

A condition is the weights and an optional protocol: base, lora, base+<protocol>,
lora+<protocol>. "lora" serves the seed's adapter (or that of the adapters_from cell),
and with eval.base_precision train the base is served as it was trained (4-bit for a
QLoRA cell), in every condition. The evaluator then grades each reply against the row's
rubric. The output, seed_<n>/eval/<condition>.json, keeps the fields of the earlier
result files and adds, per reply, the condition, the protocol, the calls, the tokens
over all calls and whether a call hit its token limit.

Two paths write the same files (run_shared and run):
  --seeds   one vLLM server serves the base and each seed's adapter as a named LoRA
            module; every condition of every seed goes through it, each request routed
            by its model name, from one pool of client threads. Then one grader server
            grades every reply. Each condition's manifest records the shared server, and
            the run's time is in one manifest in shared/eval/. Each reply goes to
            <condition>.replies.jsonl as it comes, so a crash loses only the replies in
            flight and a rerun makes only the missing ones. The runner's default
            (eval.shared_server).
  --seed N --condition C
            one server for that condition, then one grader server: the path of the runs
            up to R2, kept as the fallback (run --no-shared-server, or
            eval.shared_server: false).
With fixed eval ids (eval.ids.policy fixed_pool or fixed_file) every seed answers the
same prompts, and a base condition gives the same replies for every seed under greedy
decoding (docs/ANALYSIS_PLAN.md section 2.1). The shared path then evaluates each base
condition once and writes every seed's file from it; the manifests say so. A
per_seed_draw cell is not affected.
A row with no rubric (a benchmark with rubric: none) gets no rubric grade: its score is
null and its criteria list is empty, so the dimensions and the text counts still apply.

A non-empty output is never rewritten (a rerun after a crash skips finished conditions);
an empty file counts as a crashed write and is replaced. A condition with a failed reply
or a failed grade gets no output: its manifest says incomplete, with the failures
counted, the stage exits with an error, and a rerun makes the condition again (on the
shared path, only its missing replies). A request too long for a server's context fails
the same way on every rerun, so it is listed in the manifest (too_long_ids) and skipped,
as in the other stages.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm
from transformers import set_seed

from bodhi_lora import manifest as mf
from bodhi_lora.engine import ContextLengthError, grader_eager, make_engine
from bodhi_lora.grading.rubric import Grader, grade_rubric
from bodhi_lora.pipeline.context import StageContext, context_from_args, eval_ids_file, stage_parser
from bodhi_lora.protocols import Decoding, Output, Protocol, build, decoding_for
from bodhi_lora.registry.schema import ProtocolConfig


# Stub replies. The policy reply carries the ask-first markers so that parser runs; the
# grader reply is a valid rubric verdict.
STUB_POLICY_REPLY = "<plan>stub plan</plan>\nREPLY:\nSTUB RESPONSE"
STUB_GRADER_REPLY = '```json\n{"explanation": "stub", "criteria_met": false}\n```'


def score_response_confidence(token_logprobs):
    """Derive a token-fluency proxy from per-output-token log-probs.

    token_logprobs come from the generation call itself (no extra forward pass).
    Returns None fields when logprobs are unavailable (a wrapper's reply comes from
    its last call of several, so it has none).

    WARNING: this is a FLUENCY metric, not a clinical-calibration metric.
    The downstream Brier / ECE numbers derived from `geomean_token_prob`
    (written out as `model_fluency_geomean_prob` per result) should NOT
    be cited as model-calibration claims without explicitly noting the
    following limitations:

    1. Measures fluency, not confidence. Common medical phrasing has high
       per-token probabilities regardless of clinical accuracy, so a
       confidently wrong answer can score as high as a correct one.
    2. Response-level metric applied per-criterion. The same scalar is
       broadcast across every rubric item for a response, so a 2000-token
       answer with 15 rubric items inflates the effective sample size 15x
       while contributing zero additional calibration signal.
    3. Brier expansion bias. `_collect_binary_labels()` emits one
       (y_true, y_pred) pair per positive-point criterion per example, so
       the Brier score is dominated by prompts with many rubric items
       rather than by calibration quality.
    4. ECE bin pathology. Token-prob geomeans typically sit in 0.3-0.7,
       so the 0-0.1 and 0.9-1.0 bins are near-empty and the 10-bin ECE
       estimate is unstable.

    Use the judges' re-grades for calibration claims. This proxy is kept for
    per-response fluency reporting comparable with earlier result files only.
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


def _aggregate_scores(scores):
    """Compute mean/std/median, dropping NaN/Inf values.

    When all inputs are non-finite, returns score_status="all_nan_or_inf".
    When some are non-finite, adds n_excluded_nan so downstream tools can
    flag unstable runs. Filtering here also keeps json.dump output free of
    NaN literals, which non-Python parsers (jq, JS) reject.
    """
    if not scores:
        return {"mean": None, "std": None, "median": None}
    finite = [s for s in scores if s is not None and math.isfinite(s)]
    n_excluded = len(scores) - len(finite)
    if not finite:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "n_excluded_nan": n_excluded,
            "score_status": "all_nan_or_inf",
        }
    out = {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
    }
    if n_excluded:
        out["n_excluded_nan"] = n_excluded
    return out


def _safe_package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_run_metadata(seed, grader_model, **settings):
    """Provenance stored inside the result JSON.

    The keys up to peft_version are the ones the submitted results carry.
    settings adds the eval settings (context length, batch slots, revisions,
    eval-id hash and so on), so a result file states how it was produced.
    The full record is the manifest next to the output.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": mf.git_state().get("sha"),
        "seed": seed,
        "grader_model": grader_model,
        "bodhi_version": _safe_package_version("bodhi-llm"),
        "transformers_version": _safe_package_version("transformers"),
        "peft_version": _safe_package_version("peft"),
        "vllm_version": _safe_package_version("vllm"),
        "torch_version": _safe_package_version("torch"),
        **settings,
    }




def adapter_name(seed: int) -> str:
    """The served name of a seed's adapter on the shared server."""
    return f"adapter-s{seed}"


@dataclass
class Job:
    """One condition of one seed: its protocol, its weights, its prompts and its file."""

    seed: int
    condition: str
    weights: str
    proto_rec: Optional[ProtocolConfig]
    proto: Protocol
    decoding: Decoding
    adapter: Optional[Path]
    examples: list
    out_path: Path

    @property
    def key(self) -> tuple:
        return (self.seed, self.condition)


def make_job(ctx: StageContext, seed: int, condition: str) -> Job:
    ev = ctx.cell.eval
    if condition not in ev.conditions:
        raise SystemExit(f"{condition!r} is not a condition of {ctx.cell.id}: {ev.conditions}")
    weights, proto_rec = ctx.condition_protocol(condition)
    proto = build(proto_rec, ctx.model)
    decoding = decoding_for(proto, max_new_tokens=ev.max_new_tokens,
                            max_new_tokens_per_pass=ev.max_new_tokens_per_pass)
    rows = ctx.rows_by_id()
    return Job(seed=seed, condition=condition, weights=weights, proto_rec=proto_rec, proto=proto,
               decoding=decoding, adapter=ctx.adapter_dir(seed) if weights == "lora" else None,
               examples=[rows[pid] for pid in ctx.eval_ids(seed)],
               out_path=ctx.paths.eval_file(seed, condition))


def finished(path: Path) -> bool:
    """A non-empty output is kept; an empty file is a crashed write."""
    return path.is_file() and path.stat().st_size > 0


def check_adapters(ctx: StageContext, adapters) -> None:
    """Stop before any server starts when an adapter folder it will load is missing."""
    missing = sorted({str(a) for a in adapters if a is not None and not (a / "adapter_config.json").is_file()})
    if missing and not ctx.stub:
        raise SystemExit(f"no adapter in {', '.join(missing)}; run the train stage first")


def _traceback() -> str:
    return "\n".join(traceback.format_exc().strip().splitlines()[-30:])


def generate(jobs: list, work: list, engine_for: Callable, workers: int, desc: str,
             on_reply: Optional[Callable] = None) -> tuple[dict, dict, dict]:
    """Pass 1: the replies to the (job, row) pairs of work, from one pool of client threads.

    engine_for(job) is what the job's protocol calls: the engine, or a route of a shared
    server to the job's weights. on_reply(job, reply) runs for each reply as it comes.
    Returns the replies, the failures and the prompt ids too long for the context window,
    by job key.
    """
    replies = {j.key: [] for j in jobs}
    failures = {j.key: [] for j in jobs}
    too_long = {j.key: [] for j in jobs}

    def one(job, engine, row):
        try:
            return job, {"row": row, "out": job.proto.respond(engine, row.messages, job.decoding)}, None
        except ContextLengthError:
            return job, None, (row.prompt_id, None)
        except Exception:
            return job, None, (row.prompt_id, _traceback())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job, engine_for(job), row) for job, row in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            job, g, err = fut.result()
            if g is None:
                (too_long[job.key].append(err[0]) if err[1] is None else failures[job.key].append(err))
                continue
            replies[job.key].append(g)
            if on_reply is not None:
                on_reply(job, g)
    return replies, failures, too_long


def replies_file(job: Job) -> Path:
    """The replies of a shared run's job, one JSON line each as it comes, so a crash loses
    only the replies in flight and a rerun makes only the missing ones. Removed once the
    job's result file is written with no failed reply. Git ignores it, as it ignores
    every JSONL file under the results root."""
    return job.out_path.with_name(f"{job.condition}.replies.jsonl")


def saved_replies(job: Job) -> list:
    """The replies an earlier, crashed shared run kept for this job, for its rows."""
    path = replies_file(job)
    if not path.is_file():
        return []
    rows = {r.prompt_id: r for r in job.examples}
    kept = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue    # a line cut by the crash
        if rec.get("prompt_id") in rows:
            kept[rec["prompt_id"]] = {"row": rows[rec["prompt_id"]], "out": Output(**rec["out"])}
    return list(kept.values())


def grade_replies(jobs: list, replies: dict, engine, workers: int, desc: str) -> tuple[dict, dict, dict]:
    """Pass 2: the evaluator's rubric grade of every reply, from one pool of client threads.
    Returns the result rows (sorted by prompt id), the failures and the prompt ids too long
    for the grader's context window, by job key."""
    grader = Grader(engine)
    results = {j.key: [] for j in jobs}
    failures = {j.key: [] for j in jobs}
    too_long = {j.key: [] for j in jobs}

    def one(job, g):
        row, out = g["row"], g["out"]
        try:
            confidence = score_response_confidence(out.token_logprobs)
            if row.rubrics:
                grade_ = grade_rubric(grader, row.messages, out.response, row.rubrics)
            else:
                grade_ = {"overall_score": None, "tag_scores": {}, "criteria_results": [],
                          "parse_failures": 0}
            return job, {
                "prompt_id": row.prompt_id, "pair_id": row.pair_id, "response": out.response,
                "score": grade_["overall_score"], "tag_scores": grade_["tag_scores"],
                "criteria_results": grade_["criteria_results"],
                "parse_failures": grade_["parse_failures"],
                "model_fluency_geomean_prob": confidence["geomean_token_prob"],
                "model_confidence_mean_token_logprob": confidence["mean_token_logprob"],
                "response_token_count": confidence["response_token_count"],
                "condition": job.condition, "protocol": job.proto_rec.id if job.proto_rec else None,
                "calls": out.calls, "prompt_tokens_total": out.prompt_tokens,
                "completion_tokens_total": out.completion_tokens,
                "finish_length": out.finish_length,
                # What the protocol recorded about this reply, e.g. an abstain decision;
                # format_parsed false marks a reply whose internal format did not parse.
                "protocol_metadata": out.metadata,
            }, None
        except ContextLengthError:
            return job, None, (row.prompt_id, None)
        except Exception:
            return job, None, (row.prompt_id, _traceback())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, j, g) for j in jobs for g in replies[j.key]]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            job, r, err = fut.result()
            if r:
                results[job.key].append(r)
            else:
                (too_long[job.key].append(err[0]) if err[1] is None else failures[job.key].append(err))
    for rows in results.values():
        rows.sort(key=lambda r: r["prompt_id"])
    return results, failures, too_long


def _print_failures(what: str, failures: list) -> None:
    for pid, tb in failures[:5]:
        print(f"  {what} failed {pid}:\n{tb}", file=sys.stderr)


def write_outputs(ctx: StageContext, job: Job, results: list, n_generated: int, failed_inference: list,
                  failed_grading: list, too_long: list, engines: dict, timer: Optional[mf.Timer],
                  extra: dict) -> dict:
    """The job's result file, in the form of the earlier result files, and its manifest.

    With a failed reply or grade the job is incomplete: no result file, so that nothing
    reads a condition with rows missing as finished and a rerun makes it again, and a
    manifest with status incomplete and the failures counted. Rows too long for a
    server's context are listed and left out; they do not make the job incomplete.
    """
    complete = not failed_inference and not failed_grading
    ev = ctx.cell.eval
    proto_rec, adapter = job.proto_rec, job.adapter
    evaluator = ctx.reg.models[ctx.cell.grading.evaluator]
    quantize = ctx.quantized_base()
    scores = [r["score"] for r in results]
    fluency = [r["model_fluency_geomean_prob"] for r in results if r["model_fluency_geomean_prob"] is not None]
    parse_failures = sum(r["parse_failures"] for r in results)
    items = sum(len(r["criteria_results"]) for r in results)
    ids_file = ctx.paths.inputs() / eval_ids_file(job.seed)
    ids_digest = mf.id_list_digest(ids_file)
    summary = {
        "config": job.condition, "condition": job.condition, "model": ctx.model.hf_id,
        "lora_path": str(adapter) if adapter else None,
        "use_bodhi": bool(proto_rec and proto_rec.impl == "bodhi_llm"),
        "wrapper": proto_rec.id if proto_rec and proto_rec.kind == "wrapper" else "none",
        "baseline": proto_rec.id if proto_rec and proto_rec.kind == "prompt" else None,
        "instruction_placement": getattr(job.proto, "placement", None),
        "protocol": proto_rec.id if proto_rec else None,
        "n_examples": len(job.examples),
        "run_metadata": collect_run_metadata(
            job.seed, evaluator.hf_id, model_revision=ctx.model.revision,
            grader_revision=evaluator.revision, max_model_len=ev.serving.max_model_len,
            max_num_seqs=ev.serving.max_num_seqs, concurrency=ev.serving.concurrency,
            request_timeout_s=ev.serving.request_timeout_s,
            base_quantization="bitsandbytes" if quantize else None,
            eval_ids_n=ids_digest.get("n"), eval_ids_sha256=ids_digest.get("ids_sha256"),
            stub=ctx.stub),
        **_aggregate_scores(scores),
        "score_metric": "normalized_score",
        "model_confidence_method": (
            "geometric_mean_token_probability over the emitted response, "
            "computed from next-token logprobs conditioned on the prompt"),
        "mean_model_confidence": float(np.mean(fluency)) if fluency else None,
        "brier_model_calibration": compute_brier_score(results, "model_fluency_geomean_prob"),
        "ece_model_calibration": compute_ece(results, "model_fluency_geomean_prob"),
        # Grader-consistency proxies, kept so older comparisons do not break silently.
        "brier_grader_consistency": compute_brier_score(results, "score"),
        "ece_grader_consistency": compute_ece(results, "score"),
        "grader_parse_failure_rate": parse_failures / items if items else None,
        "grader_parse_failures_total": parse_failures,
        "grader_rubric_items_total": items,
        "results": results,
    }
    out_path = job.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if complete:
        tmp = out_path.with_suffix(".part")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(out_path)
    counts = {"n_examples": len(job.examples), "n_generated": n_generated,
              "n_failed_inference": len(failed_inference), "n_graded": len(results),
              "n_failed_grading": len(failed_grading), "n_too_long": len(too_long),
              "grader_parse_failures_total": parse_failures,
              "grader_rubric_items_total": items,
              "calls": sum(r["calls"] for r in results),
              "completion_tokens_total": sum(r["completion_tokens_total"] for r in results),
              "n_finish_length": sum(r["finish_length"] for r in results),
              "n_format_unparsed": sum(r["protocol_metadata"].get("format_parsed") is False for r in results)}
    ctx.write_manifest(out_path.parent, "eval", seed=job.seed, name=f"{job.condition}.manifest.json",
                       timer=timer, models={"policy": ctx.model, "grader": evaluator},
                       config={"condition": job.condition, "eval": ev.model_dump()},
                       engines=engines, adapter_dir=adapter, eval_ids=ids_file,
                       outputs=[out_path] if complete else [], counts=counts,
                       extra={"status": "complete" if complete else "incomplete", **extra,
                              "too_long_ids": sorted(too_long)})
    if not complete:
        print(f"eval: s{job.seed} {job.condition} incomplete: {len(failed_inference)} replies and "
              f"{len(failed_grading)} grades failed; no result file", flush=True)
        return counts
    mean = summary["mean"]
    print(f"eval: s{job.seed} {job.condition} score {mean if mean is None else round(mean, 4)} -> {out_path}",
          flush=True)
    return counts


def incomplete(counts: dict) -> bool:
    """A job that write_outputs left without a result file."""
    return bool(counts["n_failed_inference"] or counts["n_failed_grading"])


def run(ctx: StageContext, seed: int, condition: str) -> dict:
    """One condition of one seed on its own server, then its own grader server."""
    ev = ctx.cell.eval
    if condition not in ev.conditions:
        raise SystemExit(f"{condition!r} is not a condition of {ctx.cell.id}: {ev.conditions}")
    out_path = ctx.paths.eval_file(seed, condition)
    if finished(out_path):
        print(f"eval: {out_path} exists, skipping", file=sys.stderr)
        return {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    job = make_job(ctx, seed, condition)
    check_adapters(ctx, [job.adapter])
    quantize = ctx.quantized_base()
    print(f"eval: {condition} on {len(job.examples)} prompts"
          f"{', 4-bit base' if quantize else ''}{' [stub]' if ctx.stub else ''}", flush=True)

    policy = make_engine(ctx.model, serving=ev.serving, adapter=job.adapter, lora_rank=ctx.lora_rank(),
                         quantize=quantize, stub=ctx.stub, stub_reply=STUB_POLICY_REPLY,
                         log_path=ctx.log_path(f"vllm_eval_s{seed}.log"))
    with policy as eng:
        replies, failed_inference, long_replies = generate([job], [(job, row) for row in job.examples],
                                                           lambda j: eng, ev.serving.concurrency,
                                                           f"{condition} [replies]")
    _print_failures("reply", failed_inference[job.key])
    # Some engines need a pause before the next one starts on the same accelerator.
    if ev.serving.drain_s and not ctx.stub:
        time.sleep(ev.serving.drain_s)

    evaluator = ctx.reg.models[ctx.cell.grading.evaluator]
    grader_engine = make_engine(evaluator, serving=ev.serving, enforce_eager=grader_eager(ev.serving), stub=ctx.stub,
                                stub_reply=STUB_GRADER_REPLY,
                                log_path=ctx.log_path(f"vllm_eval_grader_s{seed}.log"))
    with grader_engine as geng:
        results, failed_grading, long_grades = grade_replies([job], replies, geng, ev.serving.concurrency,
                                                             f"{condition} [grades]")
    _print_failures("grade", failed_grading[job.key])
    counts = write_outputs(ctx, job, results[job.key], len(replies[job.key]), failed_inference[job.key],
                           failed_grading[job.key], long_replies[job.key] + long_grades[job.key],
                           {"policy": policy.settings(), "grader": grader_engine.settings()},
                           timer, {"condition": condition})
    if incomplete(counts):
        raise SystemExit(f"eval s{seed} {condition} is incomplete ({counts['n_failed_inference']} replies, "
                         f"{counts['n_failed_grading']} grades failed); run the stage again")
    return counts


def seed_invariant_ids(ctx: StageContext, seeds: list) -> bool:
    """Every one of these seeds evaluates the same id list: a fixed_pool or fixed_file
    cell. A per_seed_draw cell never counts, even if two draws happened to agree."""
    if ctx.cell.eval.ids.policy == "per_seed_draw" or len(seeds) < 2:
        return False
    first = ctx.eval_ids(seeds[0])
    return all(ctx.eval_ids(s) == first for s in seeds[1:])


def run_shared(ctx: StageContext, seeds: list) -> dict:
    """Every unfinished condition of these seeds through one shared server, then one
    grader server; the files and manifests per seed and condition as run() writes them."""
    ev = ctx.cell.eval
    unknown = sorted(set(seeds) - set(ctx.cell.seeds))
    if unknown:
        raise SystemExit(f"seeds {unknown} are not seeds of {ctx.cell.id}: {ctx.cell.seeds}")
    jobs = []
    for s in seeds:
        for c in ev.conditions:
            if finished(ctx.paths.eval_file(s, c)):
                print(f"eval: {ctx.paths.eval_file(s, c)} exists, skipping", file=sys.stderr)
            else:
                jobs.append(make_job(ctx, s, c))
    if not jobs:
        return {}
    timer = mf.Timer()
    # Nothing in the eval draws random numbers (greedy decoding, no request seed); the
    # seeds are set as run() sets them, to the first seed of the run.
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    set_seed(seeds[0])

    # Base once: with the same ids for every seed, the first pending seed of each base
    # condition is evaluated, and the other seeds' files are written from it.
    copied_from = {}
    if seed_invariant_ids(ctx, sorted({j.seed for j in jobs})):
        first = {}
        for j in jobs:
            if j.weights != "base":
                continue
            if j.condition in first:
                copied_from[j.key] = first[j.condition].key
            else:
                first[j.condition] = j
    to_run = [j for j in jobs if j.key not in copied_from]
    # The server holds the adapter of every seed of the run when the cell has a lora
    # condition, whatever is left to run. Its settings (eager mode, --max-loras) then do
    # not depend on which files a crash or a deletion left missing, so the base files of a
    # rerun come from the same serving as the first run's.
    has_lora = any(ctx.condition_protocol(c)[0] == "lora" for c in ev.conditions)
    adapters = {adapter_name(s): ctx.adapter_dir(s) for s in seeds} if has_lora else {}
    check_adapters(ctx, adapters.values())
    quantize = ctx.quantized_base()
    # Enough requests in flight to fill every scheduler slot (max_num_seqs) the record
    # gives the server, or the record's client threads when that is more.
    workers = max(ev.serving.concurrency, ev.serving.max_num_seqs)
    label = "-".join(str(s) for s in seeds)
    print(f"eval: {len(to_run)} conditions of seeds {label} through one server ({len(adapters)} adapters, "
          f"{workers} client threads){', 4-bit base' if quantize else ''}"
          f"{' [stub]' if ctx.stub else ''}; {len(copied_from)} base files written from another seed",
          flush=True)

    policy = make_engine(ctx.model, serving=ev.serving, adapters=adapters, lora_rank=ctx.lora_rank(),
                         quantize=quantize, stub=ctx.stub, stub_reply=STUB_POLICY_REPLY,
                         log_path=ctx.log_path(f"vllm_eval_s{label}.log"))
    routes = {j.key: adapter_name(j.seed) if j.adapter is not None else None for j in to_run}
    saved = {j.key: saved_replies(j) for j in to_run}
    have = {j.key: {g["row"].prompt_id for g in saved[j.key]} for j in to_run}
    # Wrapper conditions make two calls per reply and take longest; sent first, they leave
    # the short one-call replies for the end, so the batch stays full until the last one.
    order = sorted(to_run, key=lambda j: j.proto.kind != "wrapper")
    work = [(j, row) for j in order for row in j.examples if row.prompt_id not in have[j.key]]
    for j in jobs:
        j.out_path.parent.mkdir(parents=True, exist_ok=True)

    def keep(job, g):
        # generate() calls this in its result loop, one reply at a time.
        with open(replies_file(job), "a") as fh:
            fh.write(json.dumps({"prompt_id": g["row"].prompt_id, "out": asdict(g["out"])}) + "\n")

    replies, failed_inference = {j.key: [] for j in to_run}, {j.key: [] for j in to_run}
    long_replies = {j.key: [] for j in to_run}
    if work:
        print(f"eval: {len(work)} replies to make, {sum(map(len, saved.values()))} kept from an earlier run",
              flush=True)
        with policy as eng:
            replies, failed_inference, long_replies = generate(to_run, work, lambda j: eng.route(routes[j.key]),
                                                               workers, "replies", on_reply=keep)
    for j in to_run:
        replies[j.key] = saved[j.key] + replies[j.key]
        _print_failures(f"s{j.seed} {j.condition} reply", failed_inference[j.key])
    if ev.serving.drain_s and not ctx.stub:
        time.sleep(ev.serving.drain_s)

    evaluator = ctx.reg.models[ctx.cell.grading.evaluator]
    grader_engine = make_engine(evaluator, serving=ev.serving, enforce_eager=grader_eager(ev.serving), stub=ctx.stub,
                                stub_reply=STUB_GRADER_REPLY,
                                log_path=ctx.log_path(f"vllm_eval_grader_s{label}.log"))
    with grader_engine as geng:
        results, failed_grading, long_grades = grade_replies(order, replies, geng, workers, "grades")
    for j in to_run:
        _print_failures(f"s{j.seed} {j.condition} grade", failed_grading[j.key])

    engines = {"policy": policy.settings(), "grader": grader_engine.settings()}
    stamp = timer.start.strftime("%Y%m%dT%H%M%SZ")
    run_manifest = ctx.paths.shared("eval") / f"eval_s{label}_{stamp}.manifest.json"
    shared = {"run_manifest": ctx.project.rel(run_manifest), "seeds": list(seeds),
              "conditions_run": [list(j.key) for j in to_run], "client_threads": workers,
              "replies_kept_from_an_earlier_run": sum(map(len, saved.values()))}
    once = {}
    for key, src in copied_from.items():
        once.setdefault(src, [src[0]]).append(key[0])
    counts, outputs = {}, []
    for j in jobs:
        src = copied_from.get(j.key, j.key)
        extra = {"condition": j.condition, "shared_server": {**shared, "route": routes[src] or ctx.model.hf_id}}
        if src in once:
            # The replies and grades in this file were made once, with the seed named
            # here, and every seed listed has the same rows (docs/ANALYSIS_PLAN.md 2.1).
            extra["base_evaluated_once"] = {"evaluated_with_seed": src[0], "written_for_seeds": once[src]}
        # No time here: the run's time is in the run manifest, once for all its files.
        counts[j.key] = write_outputs(ctx, j, results[src], len(replies[src]), failed_inference[src],
                                      failed_grading[src], long_replies[src] + long_grades[src], engines, None,
                                      extra)
        if not incomplete(counts[j.key]):
            outputs.append(j.out_path)
    # Kept while the job is incomplete, so a rerun makes only the replies that are missing.
    for j in to_run:
        if not incomplete(counts[j.key]):
            replies_file(j).unlink(missing_ok=True)
    unfinished = [f"s{s} {c}" for (s, c) in counts if incomplete(counts[(s, c)])]
    # What the servers did: the conditions that ran, not the files written from them.
    totals = {k: sum(counts[j.key][k] for j in to_run)
              for k in ("n_examples", "n_generated", "n_graded", "calls", "completion_tokens_total")}
    ctx.write_manifest(run_manifest.parent, "eval", name=run_manifest.name, timer=timer,
                       models={"policy": ctx.model, "grader": evaluator},
                       config={"seeds": list(seeds), "eval": ev.model_dump()}, engines=engines,
                       outputs=outputs, counts={**totals, "n_files": len(outputs), "n_conditions_run": len(to_run),
                                                "n_incomplete": len(unfinished)},
                       extra={"status": "incomplete" if unfinished else "complete", "shared_server": shared,
                              "incomplete": unfinished,
                              "base_evaluated_once": [{"condition": src[1], "evaluated_with_seed": src[0],
                                                       "written_for_seeds": s} for src, s in once.items()]})
    if unfinished:
        raise SystemExit(f"eval: {len(unfinished)} of {len(jobs)} conditions are incomplete ({', '.join(unfinished)}): "
                         f"replies or grades failed. Run the stage again; it makes only the missing replies.")
    return {f"s{s}/{c}": v for (s, c), v in counts.items()}


def main(argv=None) -> int:
    ap = stage_parser(__doc__.splitlines()[0], seed="optional")
    ap.add_argument("--condition", default=None, help="with --seed: one condition on its own server")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds: every condition of each through one shared server")
    args = ap.parse_args(argv)
    if args.seeds is not None:
        if args.seed is not None or args.condition is not None:
            ap.error("give --seeds, or --seed with --condition, not both")
        run_shared(context_from_args(args), [int(s) for s in args.seeds.split(",") if s.strip()])
    elif args.seed is not None and args.condition is not None:
        run(context_from_args(args), args.seed, args.condition)
    else:
        ap.error("give --seeds, or --seed with --condition")
    return 0


if __name__ == "__main__":
    sys.exit(main())
