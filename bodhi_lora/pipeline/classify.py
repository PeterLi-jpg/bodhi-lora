"""Stage classify: a judge answers two questions about every stored reply of a seed.

    python -m bodhi_lora.pipeline.classify --cell <cell id> --seed N --judge <model id>
                                           [--stub-engine]

The judge (one of grading.judges) reads each reply of every condition with the
classifier template of bodhi_lora/grading/classify.py: does the reply ask the person for
information, and does it send them to urgent care. The answers go to
seed_<n>/classify/classify__<judge>.json, one entry per condition with its rates and
answers. A reply whose answer does not parse after three tries is missing. A request
that does not fit the judge's context window is listed per condition (too_long_ids) and
skipped; any other failed request leaves the stage incomplete, and a rerun retries it.

The stage only adds files, so it also runs on recorded cells. Its rates count only once
the judge passes the hand-label gate (docs/ANALYSIS_PLAN.md section 12, 2026-10-10;
grading/classify.py gate()). The runner runs it only when --stages names it
(docs/RUN_PLAN.md R17).
"""
from __future__ import annotations

import json
import sys

from bodhi_lora import manifest as mf
from bodhi_lora.engine import make_engine
from bodhi_lora.grading.classify import STUB_CLASSIFY_REPLY, TEMPLATE_SHA256, aggregate, classify
from bodhi_lora.grading.epistemic import messages_to_text
from bodhi_lora.grading.rubric import Grader
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser
from bodhi_lora.pipeline.grade import _parallel, load_condition_files


def classify_of(ctx: StageContext, seed: int, files: dict, judge_id: str) -> tuple:
    """The judge's answers for every condition of the seed."""
    if judge_id not in ctx.cell.grading.judges:
        raise SystemExit(f"{judge_id} is not a judge of {ctx.cell.id}: {ctx.cell.grading.judges}")
    judge = ctx.reg.models[judge_id]
    rows = ctx.rows_by_id()
    serving = ctx.cell.grading.judge_serving or ctx.cell.eval.serving
    engine = make_engine(judge, serving=serving, stub=ctx.stub, stub_reply=STUB_CLASSIFY_REPLY,
                         log_path=ctx.log_path(f"vllm_classify_{judge_id}_s{seed}.log"))
    configs = []
    with engine as eng:
        grader = Grader(eng)
        for condition, data in files.items():
            def one(item):
                pid = item["prompt_id"]
                try:
                    answer, raw = classify(grader, messages_to_text(rows[pid].messages), item["response"])
                except Exception as e:      # one bad request must not stop the condition
                    return None, (pid, e)
                ex = {"prompt_id": pid, "parse_failure": answer is None, "answer": answer}
                if answer is None:
                    ex["raw_text"] = raw
                return ex, None

            examples, errors, too_long = _parallel(one, data.get("results", []), serving.concurrency,
                                                   f"{condition} [classify {judge_id}]")
            examples.sort(key=lambda e: e["prompt_id"])
            configs.append({"name": condition, "source_file": str(ctx.paths.eval_file(seed, condition)),
                            "source_sha256": mf.sha256_file(ctx.paths.eval_file(seed, condition)),
                            "n_results": len(data.get("results", [])), "n_grade_errors": len(errors),
                            "n_too_long": len(too_long), "too_long_ids": too_long,
                            "rates": aggregate([e["answer"] for e in examples]), "examples": examples})
    out = {"judge_model": judge.hf_id, "judge_id": judge_id, "judge_revision": judge.revision,
           "judge_family": judge.family, "template_sha256": TEMPLATE_SHA256, "seed": seed, "stub": ctx.stub,
           "gate": "not passed: no hand labels yet (docs/ANALYSIS_PLAN.md section 12, 2026-10-10)",
           "configs": configs}
    return out, engine.settings(), judge


def run(ctx: StageContext, seed: int, judge_id: str) -> dict:
    timer = mf.Timer()
    files = load_condition_files(ctx, seed)
    out, settings, model = classify_of(ctx, seed, files, judge_id)
    folder = ctx.paths.stage(seed, "classify")
    path = folder / f"classify__{judge_id}.json"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    counts = {c["name"]: {"n": len(c["examples"]), "n_grade_errors": c["n_grade_errors"],
                          "n_too_long": c["n_too_long"]} for c in out["configs"]}
    errors = {c["name"]: c["n_grade_errors"] for c in out["configs"] if c["n_grade_errors"]}
    ctx.write_manifest(folder, "classify", seed=seed, name=f"classify__{judge_id}.manifest.json", timer=timer,
                       models={"judge": model}, engines={"judge": settings},
                       inputs=[ctx.paths.eval_file(seed, c) for c in files], outputs=[path], counts=counts,
                       extra={"status": "incomplete" if errors else "complete", "template_sha256": TEMPLATE_SHA256})
    print(f"classify: {judge_id} -> {path}", flush=True)
    if errors:
        raise SystemExit(f"classify errors {errors}; run the stage again")
    return counts


def main(argv=None) -> int:
    ap = stage_parser(__doc__.splitlines()[0], seed="required", judge=True)
    args = ap.parse_args(argv)
    if not args.judge:
        ap.error("--judge is required: the classifier is a judge of grading.judges")
    run(context_from_args(args), args.seed, args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
