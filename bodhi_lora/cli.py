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
        --judges runs only these judges of grading.judges in the grade stage; --judges none
        runs the evaluator only.
    analyze (<cell> ... | --all) [--extract] [--results-root DIR]
        The uniform analysis (bodhi_lora/analysis/). --extract first writes each seed's
        response table from the raw stage outputs (local, ignored by git); without it,
        analyze reads only the tracked tables. Each cell gets analysis/stats.json; with
        --all, the cells configs/report.yaml analyzes, then the cross-cell statistics
        when every cell they need has its stats.json.
    report [--out-dir DIR] [--results-root DIR]
        Every number the paper prints (bodhi_lora/report/): the macros, tables and record
        of paper/generated/, from each cell's analysis/stats.json and the cross-cell
        statistics. make numbers runs it.
    --results-root on run, analyze and report reads and writes cells under DIR instead of
    the results root of configs/project.yaml, e.g. the scratch root of a stub run
    (outputs/stub), so a stub cell can be analyzed and reported without editing a config.

Every command first loads and checks the whole registry, so a broken config fails here
with every problem listed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
    from bodhi_lora.pipeline.runner import DEFAULT_STAGES, RunError, run_cell

    stages = tuple(args.stages.split(",")) if args.stages else DEFAULT_STAGES
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else []
    # "none" runs the grade stage with the evaluator only.
    judges = [] if args.judges == "none" else args.judges.split(",") if args.judges else None
    mode = "dry_run" if args.dry_run else "stub" if args.stub_engine else "gpu"
    try:
        run_cell(reg, project(args.configs), args.cell, stages=stages, seeds=args.seeds, gpus=gpus,
                 mode=mode, configs_dir=args.configs, results_root=args.results_root, force=args.force,
                 judges=judges)
    except RunError as e:
        print(f"run refused or failed: {e}", flush=True)
        return 1
    return 0


def _project(args):
    """The project roots, with the results root the command line names, if any."""
    proj = project(args.configs)
    if getattr(args, "results_root", None) is not None:
        proj = replace(proj, results_root=Path(args.results_root).resolve())
    return proj


def cmd_analyze(args, reg: Registry) -> int:
    from bodhi_lora import analysis
    from bodhi_lora.analysis.extract import ExtractError

    if bool(args.cells) == args.all:
        raise SystemExit("name the cells to analyze, or pass --all")
    cfg = analysis.report_config(args.configs)
    proj = _project(args)
    cells = analysis.analyzable(reg, cfg) if args.all else args.cells
    unknown = sorted(set(cells) - set(reg.cells))
    if unknown:
        raise SystemExit(f"no cells {unknown}")
    skipped = {}
    for cid in cells:
        try:
            if args.extract:
                analysis.extract_cell(reg, cid, proj, cfg)
            analysis.analyze_cell(reg, cid, proj, cfg)
            print(f"{cid}: {proj.rel(proj.cell(cid).analysis() / 'stats.json')}")
        except (ExtractError, FileNotFoundError) as e:
            # --all covers cells whose outputs are not in this checkout; a named cell must work.
            if not args.all:
                raise SystemExit(f"{cid}: {e}")
            skipped[cid] = str(e)
    for cid, why in skipped.items():
        print(f"{cid}: not analyzed: {why}")
    if args.all:
        missing = [c for c in cfg.cross_cells if not (proj.cell(c).analysis() / "stats.json").is_file()]
        if missing:
            print(f"cross-cell statistics not run: no stats.json for {missing}")
        else:
            analysis.analyze_cross(reg, cfg, proj)
            print(f"cross-cell: {proj.rel(proj.results_root / 'cross_cell' / 'stats.json')}")
    return 0


def cmd_report(args, reg: Registry) -> int:
    from bodhi_lora import analysis, report

    proj = _project(args)
    try:
        rep = report.report(reg, analysis.report_config(args.configs), proj)
    except FileNotFoundError as e:
        raise SystemExit(f"report: {e}")
    files = report.write(rep, args.out_dir)
    print(f"report: {proj.rel(args.out_dir)}: {len(rep.macros)} macros, {len(rep.tables)} tables "
          f"({len(files)} files)")
    for cid, why in rep.record["not_reported"].items():
        print(f"  not reported: {cid}: {why}")
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
    p.add_argument("--stages", default=None,
                   help="comma list; default every stage but classify, which runs only when named")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default every seed of the cell")
    p.add_argument("--gpus", default=None, help="comma list of GPU indices, e.g. 0,1")
    p.add_argument("--judges", default=None,
                   help="comma list, or none for the evaluator only; default every judge of the cell")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stub-engine", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="rerun stages whose manifest says complete")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="the uniform analysis of cells")
    p.add_argument("cells", nargs="*", help="cell ids")
    p.add_argument("--all", action="store_true", help="every cell configs/report.yaml analyzes")
    p.add_argument("--extract", action="store_true",
                   help="first write the response tables from the raw stage outputs")
    p.add_argument("--results-root", type=Path, default=None,
                   help="read and write cells here instead of the project's results root")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="the paper's macros and tables from the analysis (make numbers)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "paper" / "generated",
                   help="where to write numbers.tex, numbers.json and the tables")
    p.add_argument("--results-root", type=Path, default=None,
                   help="read the cells' analysis here instead of the project's results root")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reg = load_registry(args.configs)
    return args.func(args, reg)
