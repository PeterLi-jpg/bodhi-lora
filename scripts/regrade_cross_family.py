"""Re-grade stored eval outputs with a judge from a third model family.

The Med42-8B cells have a Llama-3 base and were graded by Llama-3.1-8B, so
the evaluator shares the base model's family (audit EVAL-7). This script
re-grades the stored responses of such a cell with a judge that is outside
the evaluator family, the filter family and the family of each graded base
model. It writes a separate scores file and never changes the original
result files.

The judge uses the same rubric anchors as the original graders:
  - HealthBench rubric items: filter_traces.GRADER_TEMPLATE through grade_trace()
  - epistemic dimensions: eval_epistemic.EPISTEMIC_GRADER_TEMPLATE through
    grade_response()
Each example keeps the original score next to the judge score, so agreement
can be computed later from this file alone.

Families come from configs/families.yaml. The default judge is the registry's
cross_family_judge (google/gemma-2-27b-it).

Usage:
    python scripts/regrade_cross_family.py \\
        --cell results_rebuttal/healthbench__m42-health_Llama3-Med42-8B \\
        --benchmark-jsonl data/raw/healthbench_hard.jsonl [--dry-run]
    python scripts/regrade_cross_family.py --response-files <seed>/eval/base_no_wrapper.json ...

Output, per eval directory: regrade_<judge>.json and regrade_<judge>.manifest.json.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _vllm_engine import DryRunEngine, VLLMEngine  # noqa: E402
import run_manifest as rmf  # noqa: E402
from scripts.filter_traces import LocalGrader, grade_trace  # noqa: E402
from scripts.eval_epistemic import (  # noqa: E402
    DRY_RUN_EPISTEMIC_REPLY, aggregate, flatten_grade, grade_response, messages_to_text,
)

PARTS = ("rubric", "epistemic")
DRY_RUN_RUBRIC_REPLY = '```json\n{"explanation": "dry run", "criteria_met": false}\n```'


def find_response_files(cell_dir):
    """The per-condition eval outputs under <cell>/seed_*/eval/.

    A result file has a "config" tag and a "results" list. Manifests, dry-run
    files, epistemic scores and earlier re-grades are skipped.
    """
    files = []
    for path in sorted(Path(cell_dir).glob("seed_*/eval/*.json")):
        name = path.name
        if name.endswith(".manifest.json") or ".dryrun" in name or name.startswith(
                ("epistemic_scores", "regrade_")):
            continue
        files.append(path)
    return files


def load_rows(paths):
    """{prompt_id: benchmark row} from one or more HealthBench-format JSONL files."""
    rows = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["prompt_id"]] = row
    return rows


def judge_family_check(judge, response_meta, allow_same_family=False):
    """Refuse a judge that shares a family with any grader or base involved."""
    judge_family = rmf.model_family(judge)
    if judge_family is None:
        raise rmf.FamilyError(f"no family in configs/families.yaml for judge {judge}")
    involved = {"filter": rmf.role_default("filter"), "evaluator": rmf.role_default("evaluator")}
    for meta in response_meta:
        involved[f"base:{meta['model']}"] = meta["model"]
        if meta.get("grader_model"):
            involved[f"grader:{meta['grader_model']}"] = meta["grader_model"]
    families = {role: rmf.model_family(m) for role, m in involved.items()}
    unknown = [f"{role}={involved[role]}" for role, fam in families.items() if fam is None]
    if unknown:
        raise rmf.FamilyError("no family in configs/families.yaml for " + ", ".join(unknown))
    clashes = sorted(role for role, fam in families.items() if fam == judge_family)
    result = {"judge": judge, "judge_family": judge_family,
              "families": {role: {"repo_id": involved[role], "family": fam}
                           for role, fam in families.items()},
              "clashes": clashes, "allow_same_family": allow_same_family,
              "passed": not clashes}
    if clashes and not allow_same_family:
        raise rmf.FamilyError(
            f"judge {judge} ({judge_family}) shares a family with: {', '.join(clashes)}. "
            "Pick another judge or pass --allow-same-family.")
    return result


def _dry_run_reply(messages):
    text = messages[-1]["content"]
    return DRY_RUN_EPISTEMIC_REPLY if "epistemic virtues" in text else DRY_RUN_RUBRIC_REPLY


def regrade_file(grader, data, rows, parts, concurrency):
    """Re-grade every stored response of one result file."""

    def _one(item):
        pid = item["prompt_id"]
        row = rows.get(pid)
        if row is None:
            return None, ("lookup", pid)
        out = {"prompt_id": pid, "original_score": item.get("score")}
        try:
            if "rubric" in parts:
                grade = grade_trace(grader, row["prompt"], item["response"], row["rubrics"])
                out["judge_score"] = grade["overall_score"]
                out["judge_parse_failures"] = grade["parse_failures"]
                out["judge_criteria_met"] = [c["criteria_met"] for c in grade["criteria_results"]]
            if "epistemic" in parts:
                parsed, raw = grade_response(grader, messages_to_text(row["prompt"]), item["response"])
                out["epistemic_parse_failure"] = parsed is None
                out["scores"] = flatten_grade(parsed) if parsed is not None else None
                if parsed is None:
                    out["raw_grade_text"] = raw
        except Exception as e:  # one bad request must not stop the file
            return None, ("grade_error", pid, repr(e))
        return out, None

    examples, errors = [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(_one, it) for it in data.get("results", [])]):
            ex, err = fut.result()
            if ex is not None:
                examples.append(ex)
            else:
                errors[err[0]].append(err[1:])
    examples.sort(key=lambda e: e["prompt_id"])

    summary = {"n_results": len(data.get("results", [])), "n_regraded": len(examples),
               "n_failed_lookups": len(errors["lookup"]), "n_grade_errors": len(errors["grade_error"])}
    if "rubric" in parts:
        judge = [e["judge_score"] for e in examples if e.get("judge_score") is not None]
        orig = [e["original_score"] for e in examples if e.get("original_score") is not None]
        summary["rubric"] = {
            "judge_mean": float(np.mean(judge)) if judge else None,
            "original_mean": float(np.mean(orig)) if orig else None,
            "judge_parse_failures": sum(e.get("judge_parse_failures", 0) for e in examples),
        }
    if "epistemic" in parts:
        summary["epistemic"] = aggregate(examples)
    return summary, examples, dict(errors)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", help="cell directory with seed_*/eval/*.json")
    src.add_argument("--response-files", nargs="+")
    ap.add_argument("--benchmark-jsonl", nargs="+", default=["data/raw/healthbench_hard.jsonl"],
                    help="prompt and rubric source, joined on prompt_id")
    ap.add_argument("--judge", default=None,
                    help="judge model. Default: roles.cross_family_judge in configs/families.yaml")
    ap.add_argument("--judge-revision", default=None)
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--allow-same-family", action="store_true")
    # Judge prompts carry the whole conversation and response, so the default
    # context is the full 8192 that gemma-2-27b-it supports.
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("EVAL_CONCURRENCY", "16")))
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true",
                    help="stub the judge, re-grade 2 responses per file, write *.dryrun.json")
    args = ap.parse_args(argv)

    judge = args.judge or rmf.role_default("cross_family_judge")
    files = ([Path(p) for p in args.response_files] if args.response_files
             else find_response_files(args.cell))
    if not files:
        raise SystemExit(f"no result files found under {args.cell}/seed_*/eval/")
    loaded = {}
    for path in files:
        with open(path) as f:
            loaded[path] = json.load(f)
    meta = [{"model": d.get("model"), "grader_model": (d.get("run_metadata") or {}).get("grader_model")}
            for d in loaded.values()]
    try:
        family = judge_family_check(judge, meta, args.allow_same_family)
    except rmf.FamilyError as e:
        raise SystemExit(f"REFUSED: {e}")
    print(f"judge {judge} ({family['judge_family']}); "
          f"{'waived: ' + ', '.join(family['clashes']) if family['clashes'] else 'family check passed'}")

    rows = load_rows(args.benchmark_jsonl)
    revision = rmf.effective_revision(judge, args.judge_revision)
    engine_kwargs = dict(revision=revision, max_model_len=args.max_model_len,
                         max_num_seqs=args.max_num_seqs, request_timeout=args.request_timeout)
    if args.dry_run:
        engine = DryRunEngine(judge, reply=_dry_run_reply, **engine_kwargs)
        for data in loaded.values():
            data["results"] = data.get("results", [])[:2]
    else:
        engine = VLLMEngine(judge, **engine_kwargs)

    by_dir = defaultdict(list)
    for path in files:
        by_dir[path.parent].append(path)
    tag = judge.replace("/", "__")
    written = []
    with engine as eng:
        grader = LocalGrader(eng)
        for eval_dir, paths in sorted(by_dir.items()):
            configs = []
            for path in paths:
                data = loaded[path]
                summary, examples, errors = regrade_file(grader, data, rows, args.parts,
                                                         args.concurrency)
                configs.append({"name": data.get("config") or path.stem,
                                "source_file": str(path),
                                "source_sha256": rmf.sha256_file(path),
                                "model": data.get("model"),
                                "original_grader": (data.get("run_metadata") or {}).get("grader_model"),
                                **summary, "errors": errors, "examples": examples})
                print(f"  {path}: {summary}")
            out = eval_dir / f"regrade_{tag}{'.dryrun' if args.dry_run else ''}.json"
            with open(out, "w") as f:
                json.dump({"judge_model": judge, "judge_revision": revision,
                           "parts": args.parts, "family_check": family,
                           "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "dry_run": args.dry_run, "configs": configs}, f, indent=2)
            rmf.write_manifest(rmf.manifest_path_for(out), rmf.build_manifest(
                "regrade_cross_family", config=vars(args),
                models={"judge": (judge, args.judge_revision)},
                engines={"judge": eng.settings()},
                inputs=[*paths, *args.benchmark_jsonl], outputs=[out],
                counts={c["name"]: {"n_results": c["n_results"], "n_regraded": c["n_regraded"]}
                        for c in configs},
                extra={"family_check": family}))
            written.append(out)
    for out in written:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
