"""Stage grade: grade the stored replies of a seed, never changing them.

    python -m bodhi_lora.pipeline.grade --cell <cell id> --seed N [--judge <model id>]
                                        [--stub-engine]

Without --judge, the evaluator scores every condition's replies on the seven epistemic
dimensions into seed_<n>/grade/epistemic__<evaluator>.json (the format of the earlier
epistemic_scores.json: configs[] with one entry per condition and its examples).

With --judge, a judge from grading.judges grades the same replies again into
seed_<n>/grade/judge__<judge>.json, in three parts:
  rubric      the criteria the evaluator graded, read from the eval file, so the two
              verdicts pair criterion by criterion; each example keeps both lists of
              verdicts and both scores, so agreement is computed from this file alone
  dimensions  the seven epistemic dimensions, with the evaluator's template
  harm        extent and likelihood of possible harm and delayed care (grading/harm.py)
A reply whose row has no rubric (a benchmark with rubric: none) gets no rubric part.
The registry's family rule (R3) has already checked that the judge shares no family
with the base, the teacher or the filter.

A judge re-grade adds files only, so it also runs on recorded cells (the runner allows
it there).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.grading.epistemic import (STUB_EPISTEMIC_REPLY, aggregate, flatten_grade,
                                          grade_dimensions, messages_to_text)
from bodhi_lora.grading.harm import HARM_PROMPT_MARKER, STUB_HARM_REPLY, aggregate_harm, grade_harm
from bodhi_lora.grading.rubric import Grader, grade_rubric
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser

STUB_RUBRIC_REPLY = '```json\n{"explanation": "stub", "criteria_met": false}\n```'


def _stub_reply(messages):
    """A dimension grade, a harm grade or a rubric verdict, by the prompt it answers."""
    prompt = messages[-1]["content"]
    if "epistemic virtues" in prompt:
        return STUB_EPISTEMIC_REPLY
    return STUB_HARM_REPLY if HARM_PROMPT_MARKER in prompt else STUB_RUBRIC_REPLY


def graded_criteria(item: dict, row) -> tuple[list, list | None, list | None]:
    """The criteria the evaluator graded for one reply, its verdicts, and which verdicts
    are parse failures (scored not met, so agreement can leave them out).

    Read from the eval file, so a judge grades exactly what the evaluator graded even if
    a benchmark's rows were rebuilt since. A file without criteria_results falls back
    to the row's rubric, with no evaluator verdicts.
    """
    stored = item.get("criteria_results")
    if stored:
        return ([{"criterion": c["criterion"], "points": c["points"], "tags": c.get("tags", [])}
                 for c in stored], [bool(c["criteria_met"]) for c in stored],
                [bool(c.get("parse_failed", False)) for c in stored])
    return row.rubrics, None, None


def load_condition_files(ctx: StageContext, seed: int) -> dict[str, dict]:
    """The eval output of every condition of the seed, by condition."""
    out = {}
    for condition in ctx.cell.eval.conditions:
        path = ctx.paths.eval_file(seed, condition)
        if not path.is_file():
            raise SystemExit(f"{path} is missing: run the eval stage for {condition} first")
        out[condition] = json.loads(path.read_text())
    return out


def _parallel(fn, items, concurrency, desc):
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in tqdm(as_completed([pool.submit(fn, it) for it in items]), total=len(items), desc=desc):
            ok, err = fut.result()
            (done.append(ok) if ok is not None else errors.append(err))
    return done, errors


def grade_dimensions_of(ctx: StageContext, seed: int, files: dict) -> tuple:
    """The evaluator's dimension grades of every condition."""
    evaluator = ctx.reg.models[ctx.cell.grading.evaluator]
    rows = ctx.rows_by_id()
    serving = ctx.cell.eval.serving
    engine = make_engine(evaluator, serving=serving, stub=ctx.stub, stub_reply=STUB_EPISTEMIC_REPLY,
                         log_path=ctx.log_path(f"vllm_grade_s{seed}.log"))
    out = {"grader_model": evaluator.hf_id, "grader_id": evaluator.id, "seed": seed,
           "run_metadata": {"git_sha": mf.git_state().get("sha"), "grader_revision": evaluator.revision,
                            "max_model_len": serving.max_model_len, "max_num_seqs": serving.max_num_seqs,
                            "concurrency": serving.concurrency,
                            "request_timeout_s": serving.request_timeout_s,
                            "packages": mf.package_versions(), "stub": ctx.stub},
           "epistemic_dimensions": ["uncertainty_acknowledgment", "context_seeking",
                                    "red_flag_identification", "scope_bounding", "specificity", "hedging"],
           "configs": []}
    with engine as eng:
        grader = Grader(eng)
        for condition, data in files.items():
            def one(item):
                pid = item["prompt_id"]
                try:
                    parsed, raw = grade_dimensions(grader, messages_to_text(rows[pid].messages),
                                                   item["response"])
                except Exception as e:      # one bad request must not stop the condition
                    return None, (pid, repr(e))
                if parsed is None:
                    return {"prompt_id": pid, "response": item["response"], "parse_failure": True,
                            "scores": None, "raw_grade_text": raw}, None
                return {"prompt_id": pid, "response": item["response"], "parse_failure": False,
                        "scores": flatten_grade(parsed), "raw_grade": parsed}, None

            graded, errors = _parallel(one, data.get("results", []), serving.concurrency,
                                       f"{condition} [dimensions]")
            graded.sort(key=lambda g: g["prompt_id"])
            out["configs"].append({"name": condition, "source_file": str(ctx.paths.eval_file(seed, condition)),
                                   "aggregates": aggregate(graded), "n_failed_lookups": 0,
                                   "n_grade_errors": len(errors), "examples": graded})
    return out, engine.settings(), evaluator


def judge_of(ctx: StageContext, seed: int, files: dict, judge_id: str) -> tuple:
    """A judge's rubric, dimension and harm grades of every condition."""
    if judge_id not in ctx.cell.grading.judges:
        raise SystemExit(f"{judge_id} is not a judge of {ctx.cell.id}: {ctx.cell.grading.judges}")
    judge = ctx.reg.models[judge_id]
    rows = ctx.rows_by_id()
    serving = ctx.cell.grading.judge_serving or ctx.cell.eval.serving
    engine = make_engine(judge, serving=serving, stub=ctx.stub, stub_reply=_stub_reply,
                         log_path=ctx.log_path(f"vllm_judge_{judge_id}_s{seed}.log"))
    configs = []
    with engine as eng:
        grader = Grader(eng)
        for condition, data in files.items():
            def one(item):
                pid = item["prompt_id"]
                row = rows[pid]
                criteria, evaluator_met, evaluator_failed = graded_criteria(item, row)
                text = messages_to_text(row.messages)
                try:
                    rub = grade_rubric(grader, row.messages, item["response"], criteria) if criteria else None
                    parsed, raw = grade_dimensions(grader, text, item["response"])
                    harm, harm_raw = grade_harm(grader, text, item["response"])
                except Exception as e:      # one bad request must not stop the condition
                    return None, (pid, repr(e))
                ex = {"prompt_id": pid, "original_score": item.get("score"),
                      "judge_score": rub["overall_score"] if rub else None,
                      "judge_parse_failures": rub["parse_failures"] if rub else 0,
                      "judge_criteria_met": [c["criteria_met"] for c in rub["criteria_results"]] if rub else [],
                      "judge_criteria_parse_failed": [c["parse_failed"] for c in rub["criteria_results"]] if rub else [],
                      "evaluator_criteria_met": evaluator_met,
                      "evaluator_criteria_parse_failed": evaluator_failed,
                      "criteria_points": [c["points"] for c in criteria],
                      "epistemic_parse_failure": parsed is None,
                      "scores": flatten_grade(parsed) if parsed is not None else None,
                      "harm_parse_failure": harm is None, "harm": harm}
                if parsed is None:
                    ex["raw_grade_text"] = raw
                if harm is None:
                    ex["raw_harm_text"] = harm_raw
                return ex, None

            examples, errors = _parallel(one, data.get("results", []), serving.concurrency,
                                         f"{condition} [judge {judge_id}]")
            examples.sort(key=lambda e: e["prompt_id"])
            judge_scores = [e["judge_score"] for e in examples if e["judge_score"] is not None]
            original = [e["original_score"] for e in examples if e["original_score"] is not None]
            configs.append({"name": condition, "source_file": str(ctx.paths.eval_file(seed, condition)),
                            "source_sha256": mf.sha256_file(ctx.paths.eval_file(seed, condition)),
                            "n_results": len(data.get("results", [])), "n_regraded": len(examples),
                            "n_grade_errors": len(errors),
                            "rubric": {"judge_mean": float(np.mean(judge_scores)) if judge_scores else None,
                                       "original_mean": float(np.mean(original)) if original else None,
                                       "judge_parse_failures": sum(e["judge_parse_failures"] for e in examples)},
                            "epistemic": aggregate(examples),
                            "harm": aggregate_harm([e["harm"] for e in examples]),
                            "examples": examples})
    out = {"judge_model": judge.hf_id, "judge_id": judge_id, "judge_revision": judge.revision,
           "judge_family": judge.family, "judge_domain": judge.domain,
           "parts": ["rubric", "epistemic", "harm"], "seed": seed, "stub": ctx.stub, "configs": configs}
    return out, engine.settings(), judge


def run(ctx: StageContext, seed: int, judge_id: str | None = None) -> dict:
    timer = mf.Timer()
    files = load_condition_files(ctx, seed)
    if judge_id is None:
        out, settings, model = grade_dimensions_of(ctx, seed, files)
        path, role, name = ctx.paths.grade_file(seed, model.id), "grader", "epistemic"
    else:
        out, settings, model = judge_of(ctx, seed, files, judge_id)
        path, role, name = ctx.paths.judge_file(seed, judge_id), "judge", f"judge__{judge_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    errors = defaultdict(int)
    for c in out["configs"]:
        errors[c["name"]] = c["n_grade_errors"]
    counts = {c["name"]: {"n": len(c["examples"]), "n_grade_errors": c["n_grade_errors"]} for c in out["configs"]}
    ctx.write_manifest(path.parent, "grade", seed=seed, name=f"{name}.manifest.json", timer=timer,
                       models={role: model}, engines={role: settings},
                       inputs=[ctx.paths.eval_file(seed, c) for c in files], outputs=[path], counts=counts,
                       extra={"status": "complete" if not any(errors.values()) else "incomplete"})
    print(f"grade: {name} -> {path}", flush=True)
    if any(errors.values()):
        raise SystemExit(f"grade errors {dict(errors)}; run the stage again")
    return counts


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0], seed="required", judge=True).parse_args(argv)
    run(context_from_args(args), args.seed, args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
