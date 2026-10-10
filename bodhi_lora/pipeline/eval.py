"""Stage eval: one condition's replies to the seed's eval prompts, graded on the rubric.

    python -m bodhi_lora.pipeline.eval --cell <cell id> --seed N --condition <condition>
                                       [--stub-engine]

A condition is the weights and an optional protocol: base, lora, base+<protocol>,
lora+<protocol>. "lora" serves the seed's adapter (or that of the adapters_from cell),
and with eval.base_precision train the base is served as it was trained (4-bit for a
QLoRA cell), in every condition. The evaluator then grades each reply against the row's
rubric. The output, seed_<n>/eval/<condition>.json, keeps the fields of the earlier
result files and adds, per reply, the condition, the protocol, the calls, the tokens
over all calls and whether a call hit its token limit.

A row with no rubric (a benchmark with rubric: none) gets no rubric grade: its score is
null and its criteria list is empty, so the dimensions and the text counts still apply.

A non-empty output is never rewritten (a rerun after a crash skips finished conditions);
an empty file counts as a crashed write and is replaced.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
from tqdm import tqdm
from transformers import set_seed

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.grading.rubric import Grader, grade_rubric
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser
from bodhi_lora.protocols import build, decoding_for

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


def run(ctx: StageContext, seed: int, condition: str) -> dict:
    ev = ctx.cell.eval
    if condition not in ev.conditions:
        raise SystemExit(f"{condition!r} is not a condition of {ctx.cell.id}: {ev.conditions}")
    out_path = ctx.paths.eval_file(seed, condition)
    if out_path.is_file() and out_path.stat().st_size > 0:
        print(f"eval: {out_path} exists, skipping", file=sys.stderr)
        return {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    weights, proto_rec = ctx.condition_protocol(condition)
    proto = build(proto_rec, ctx.model)
    decoding = decoding_for(proto, max_new_tokens=ev.max_new_tokens,
                            max_new_tokens_per_pass=ev.max_new_tokens_per_pass)
    adapter = ctx.adapter_dir(seed) if weights == "lora" else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file() and not ctx.stub:
        raise SystemExit(f"{adapter} holds no adapter; run the train stage first")
    rows = ctx.rows_by_id()
    ids = ctx.eval_ids(seed)
    examples = [rows[pid] for pid in ids]
    quantize = ctx.quantized_base()
    print(f"eval: {condition} on {len(examples)} prompts"
          f"{', 4-bit base' if quantize else ''}{' [stub]' if ctx.stub else ''}", flush=True)

    # Pass 1: replies.
    policy = make_engine(ctx.model, serving=ev.serving, adapter=adapter, quantize=quantize,
                         stub=ctx.stub, stub_reply=STUB_POLICY_REPLY,
                         log_path=ctx.log_path(f"vllm_eval_s{seed}.log"))
    generations, failed_inference = [], []
    with policy as eng:
        def gen(row):
            try:
                return {"row": row, "out": proto.respond(eng, row.messages, decoding)}, None
            except Exception:
                return None, (row.prompt_id, "\n".join(traceback.format_exc().strip().splitlines()[-30:]))

        with ThreadPoolExecutor(max_workers=ev.serving.concurrency) as pool:
            futures = [pool.submit(gen, r) for r in examples]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{condition} [replies]"):
                g, err = fut.result()
                (generations.append(g) if g else failed_inference.append(err))
    for pid, tb in failed_inference[:5]:
        print(f"  reply failed {pid}:\n{tb}", file=sys.stderr)
    # Some engines need a pause before the next one starts on the same accelerator.
    if ev.serving.drain_s and not ctx.stub:
        time.sleep(ev.serving.drain_s)

    # Pass 2: rubric grades.
    evaluator = ctx.reg.models[ctx.cell.grading.evaluator]
    grader_engine = make_engine(evaluator, serving=ev.serving, stub=ctx.stub,
                                stub_reply=STUB_GRADER_REPLY,
                                log_path=ctx.log_path(f"vllm_eval_grader_s{seed}.log"))
    results, failed_grading = [], []
    with grader_engine as geng:
        grader = Grader(geng)

        def grade(g):
            row, out = g["row"], g["out"]
            try:
                confidence = score_response_confidence(out.token_logprobs)
                if row.rubrics:
                    grade_ = grade_rubric(grader, row.messages, out.response, row.rubrics)
                else:
                    grade_ = {"overall_score": None, "tag_scores": {}, "criteria_results": [],
                              "parse_failures": 0}
                return {
                    "prompt_id": row.prompt_id, "response": out.response,
                    "score": grade_["overall_score"], "tag_scores": grade_["tag_scores"],
                    "criteria_results": grade_["criteria_results"],
                    "parse_failures": grade_["parse_failures"],
                    "model_fluency_geomean_prob": confidence["geomean_token_prob"],
                    "model_confidence_mean_token_logprob": confidence["mean_token_logprob"],
                    "response_token_count": confidence["response_token_count"],
                    "condition": condition, "protocol": proto_rec.id if proto_rec else None,
                    "calls": out.calls, "prompt_tokens_total": out.prompt_tokens,
                    "completion_tokens_total": out.completion_tokens,
                    "finish_length": out.finish_length,
                    # What the protocol recorded about this reply, e.g. an abstain decision;
                    # format_parsed false marks a reply whose internal format did not parse.
                    "protocol_metadata": out.metadata,
                }, None
            except Exception:
                return None, (row.prompt_id, "\n".join(traceback.format_exc().strip().splitlines()[-30:]))

        with ThreadPoolExecutor(max_workers=ev.serving.concurrency) as pool:
            futures = [pool.submit(grade, g) for g in generations]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{condition} [grades]"):
                r, err = fut.result()
                (results.append(r) if r else failed_grading.append(err))
    for pid, tb in failed_grading[:5]:
        print(f"  grade failed {pid}:\n{tb}", file=sys.stderr)
    results.sort(key=lambda r: r["prompt_id"])

    scores = [r["score"] for r in results]
    fluency = [r["model_fluency_geomean_prob"] for r in results if r["model_fluency_geomean_prob"] is not None]
    parse_failures = sum(r["parse_failures"] for r in results)
    items = sum(len(r["criteria_results"]) for r in results)
    ids_digest = mf.id_list_digest(ctx.paths.inputs() / f"eval_ids_seed_{seed}.json")
    summary = {
        "config": condition, "condition": condition, "model": ctx.model.hf_id,
        "lora_path": str(adapter) if adapter else None,
        "use_bodhi": bool(proto_rec and proto_rec.impl == "bodhi_llm"),
        "wrapper": proto_rec.id if proto_rec and proto_rec.kind == "wrapper" else "none",
        "baseline": proto_rec.id if proto_rec and proto_rec.kind == "prompt" else None,
        "instruction_placement": getattr(proto, "placement", None),
        "protocol": proto_rec.id if proto_rec else None,
        "n_examples": len(examples),
        "run_metadata": collect_run_metadata(
            seed, evaluator.hf_id, model_revision=ctx.model.revision,
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
    tmp = out_path.with_suffix(".part")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out_path)
    counts = {"n_examples": len(examples), "n_generated": len(generations),
              "n_failed_inference": len(failed_inference), "n_graded": len(results),
              "n_failed_grading": len(failed_grading), "grader_parse_failures_total": parse_failures,
              "grader_rubric_items_total": items,
              "calls": sum(r["calls"] for r in results),
              "completion_tokens_total": sum(r["completion_tokens_total"] for r in results),
              "n_finish_length": sum(r["finish_length"] for r in results),
              "n_format_unparsed": sum(r["protocol_metadata"].get("format_parsed") is False for r in results)}
    ctx.write_manifest(out_path.parent, "eval", seed=seed, name=f"{condition}.manifest.json",
                       timer=timer, models={"policy": ctx.model, "grader": evaluator},
                       config={"condition": condition, "eval": ev.model_dump()},
                       engines={"policy": policy.settings(), "grader": grader_engine.settings()},
                       adapter_dir=adapter, eval_ids=ctx.paths.inputs() / f"eval_ids_seed_{seed}.json",
                       outputs=[out_path], counts=counts,
                       extra={"status": "complete", "condition": condition})
    mean = summary["mean"]
    print(f"eval: {condition} score {mean if mean is None else round(mean, 4)} -> {out_path}", flush=True)
    return counts


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0], seed="required", condition=True).parse_args(argv)
    run(context_from_args(args), args.seed, args.condition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
