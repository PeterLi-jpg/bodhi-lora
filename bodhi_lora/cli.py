"""The command line: python -m bodhi_lora <command>.

    list [models|benchmarks|protocols|cells] [--settings]
        Print the registry. --settings prints every setting that differs between the
        records, so a reader sees where the old runs depart from the new ones.
    fetch <benchmark> ...
        Download or build a benchmark's rows into the data root, checked against the
        record's pins.
    run <cell> [--stages traces,filter,...] [--seeds 42 7] [--gpus 0,1] [--judges a,b]
               [--stub-engine | --dry-run] [--results-root DIR] [--force]
        Run a cell (bodhi_lora/pipeline/runner.py). --stub-engine runs every stage on the
        CPU with the stub engine and trains for real; --dry-run first prints the GPU
        command of every step, then runs like --stub-engine without training. Both
        write under the scratch root (configs/project.yaml), never the results root.
        --judges runs only these judges of grading.judges in the grade stage.
    analyze, report
        The uniform analysis and the paper report; the analysis lane builds them.

Every command first loads and checks the whole registry, so a broken config fails here
with every problem listed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bodhi_lora.paths import CONFIGS_DIR, project
from bodhi_lora.registry import Registry, family_problems, load_registry

KINDS = ("models", "benchmarks", "protocols", "cells")


def _flatten(obj, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}. Lists stay whole, as their JSON text."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = json.dumps(obj) if isinstance(obj, (list, tuple)) else obj
    return out


def differing_settings(records: dict) -> dict[str, dict[str, object]]:
    """Setting -> {record id: value}, for the settings whose values are not all the same.

    A record without the setting shows "-". Ids and report fields are left out: they
    differ by design.
    """
    flat = {rid: _flatten(rec.model_dump(exclude={"id", "report"})) for rid, rec in records.items()}
    keys = sorted(set().union(*(f.keys() for f in flat.values()))) if flat else []
    out = {}
    for key in keys:
        values = {rid: f.get(key, "-") for rid, f in flat.items()}
        if len({json.dumps(v, default=str) for v in values.values()}) > 1:
            out[key] = values
    return out


def _marks(cell, reg: Registry) -> str:
    marks = []
    if cell.grading.same_family_waiver:
        marks.append("family waiver")
    if cell.report.set_aside:
        marks.append("set aside")
    if cell.train is not None and cell.train.engine == "external":
        marks.append("trained outside the package")
    if family_problems(cell, reg):
        marks.append("family rule fails")
    return ", ".join(marks)


def cmd_list(args, reg: Registry) -> int:
    kinds = [args.kind] if args.kind else list(KINDS)
    for kind in kinds:
        records = getattr(reg, kind)
        if args.settings:
            diffs = differing_settings(records)
            print(f"{kind}: {len(diffs)} settings differ between {len(records)} records")
            for key, values in diffs.items():
                print(f"  {key}")
                groups: dict[str, list[str]] = {}
                for rid, v in values.items():
                    groups.setdefault(json.dumps(v, default=str), []).append(rid)
                for v, rids in sorted(groups.items()):
                    print(f"    {v}: {', '.join(sorted(rids))}")
            continue
        print(f"{kind} ({len(records)})")
        for rid, rec in sorted(records.items()):
            if kind == "models":
                pin = rec.revision[:7] if rec.revision else "unpinned"
                line = f"{rec.hf_id}  family {rec.family}, {rec.domain}, {rec.params_b:g}B, {pin}"
            elif kind == "benchmarks":
                line = f"{rec.loader}, {rec.audience}, splits {', '.join(rec.splits)}"
            elif kind == "protocols":
                line = f"{rec.kind} {rec.impl}, {rec.passes} pass{'es' if rec.passes > 1 else ''}"
            else:
                line = (f"{rec.report.status}, seeds {rec.seeds}, "
                        f"{', '.join(rec.eval.conditions)}")
                marks = _marks(rec, reg)
                if marks:
                    line += f"  [{marks}]"
            print(f"  {rid}  {line}")
    return 0


def cmd_fetch(args, reg: Registry) -> int:
    from bodhi_lora.data import load_rows

    proj = project(args.configs)
    for bid in args.benchmarks:
        if bid not in reg.benchmarks:
            raise SystemExit(f"no benchmark {bid!r}; benchmarks are {sorted(reg.benchmarks)}")
        rows = load_rows(reg.benchmarks[bid], proj.data_root, fetch=True)
        print(f"{bid}: {len(rows)} rows in {proj.rel(proj.benchmark_dir(bid))}")
    return 0


def cmd_run(args, reg: Registry) -> int:
    from bodhi_lora.pipeline.runner import STAGES, RunError, run_cell

    stages = tuple(args.stages.split(",")) if args.stages else STAGES
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else []
    judges = args.judges.split(",") if args.judges else None
    mode = "dry_run" if args.dry_run else "stub" if args.stub_engine else "gpu"
    try:
        run_cell(reg, project(args.configs), args.cell, stages=stages, seeds=args.seeds, gpus=gpus,
                 mode=mode, configs_dir=args.configs, results_root=args.results_root, force=args.force,
                 judges=judges)
    except RunError as e:
        print(f"run refused or failed: {e}", flush=True)
        return 1
    return 0


def cmd_analysis_lane(args, reg: Registry) -> int:
    from bodhi_lora import analysis, report

    try:
        if args.command == "analyze":
            analysis.analyze_cell(reg, "")
        else:
            report.report(reg, {}, Path("paper/generated"))
    except NotImplementedError as e:
        print(f"python -m bodhi_lora {args.command}: {e}. Until then, make numbers and "
              "make figures write the paper's outputs.")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bodhi_lora", description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", type=Path, default=CONFIGS_DIR, help="the registry folder")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="print the registry")
    p.add_argument("kind", nargs="?", choices=KINDS)
    p.add_argument("--settings", action="store_true", help="print the settings that differ")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("fetch", help="download or build benchmark rows")
    p.add_argument("benchmarks", nargs="+")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("run", help="run a cell")
    p.add_argument("cell")
    p.add_argument("--stages", default=None, help="comma list; default every stage")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default every seed of the cell")
    p.add_argument("--gpus", default=None, help="comma list of GPU indices, e.g. 0,1")
    p.add_argument("--judges", default=None, help="comma list; default every judge of the cell")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stub-engine", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="rerun stages whose manifest says complete")
    p.set_defaults(func=cmd_run)

    for name in ("analyze", "report"):
        p = sub.add_parser(name, help="built by the analysis lane")
        p.set_defaults(func=cmd_analysis_lane)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reg = load_registry(args.configs)
    return args.func(args, reg)
