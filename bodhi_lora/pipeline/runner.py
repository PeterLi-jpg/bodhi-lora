"""Run a cell: prepare its inputs, then run each stage in order, for each seed.

    run_cell(reg, proj, cell_id, stages=..., seeds=..., gpus=[0, 1], mode="gpu")

Modes:
  gpu      every model stage runs as a subprocess in its Python environment
           (configs/project.yaml: python.infer for vLLM stages, python.train for
           training), pinned to one GPU by CUDA_VISIBLE_DEVICES, its vLLM server on
           vllm_port_base + GPU index. Seeds run in parallel, one per GPU.
  stub     the CPU stub engine answers for vLLM, and everything else runs for real,
           training included, in the current interpreter (the smoke cell).
  dry_run  prints the GPU command of every step first, then runs like stub but skips
           training.
A stub or dry run writes under <scratch_root>/<mode>/ (configs/project.yaml), or under
results_root when the caller names one, and never under the project's results root:
its stage manifests say complete, so a GPU run there would skip those stages and keep
stub outputs.

Order: prepare (pools, eval ids, cell.yaml), for an eval-only cell the adapter leakage
gate, then probe, traces, filter (grading), the leakage gate, then per seed: filter (split), train, eval (each condition), grade (the
evaluator, then each judge), bench. A stage whose manifest says complete is skipped, so
a run resumes where it stopped; prepare refuses inputs that differ from an earlier run.
`judges` limits the judge steps to some of grading.judges, so one judge panel can run
on its own.

A cell whose report.status records a past run (complete, partial, not_run, pilot) keeps
its outputs: only the stages in ADDITIVE_STAGES run on it, and they only add files (a
judge re-grade, the contamination probe). The evaluator's grades of such a cell are
never redone. A stub or dry
run of a recorded cell links its stored eval outputs into the scratch root, read-only,
so the stub judges grade the real stored replies. A new run of a recorded design is a
new cell.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import yaml
from pydantic import ValidationError

from bodhi_lora import manifest as mf
from bodhi_lora.data import (DataError, check_leakage, eval_ids, eval_population, load_rows, read_id_list, row_text,
                             split_rows)
from bodhi_lora.data.leakage import NGRAM, LeakageError, check_adapter_leakage
from bodhi_lora.data.loaders import source_hashes
from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT, CellPaths, Project
from bodhi_lora.pipeline import filter as filter_stage
from bodhi_lora.pipeline.context import POOLS_FILE, StageContext, eval_ids_file
from bodhi_lora.registry import RECORD_STATUSES, CellConfig, Registry, family_problems

STAGES = ("probe", "traces", "filter", "train", "eval", "grade", "classify", "bench")
# What a run without --stages runs. The judge classifier runs only when --stages names it
# (docs/RUN_PLAN.md R17): its rates count only after its hand-label gate, and the run
# plan's costs leave it out.
DEFAULT_STAGES = tuple(s for s in STAGES if s != "classify")
MODES = ("gpu", "stub", "dry_run")
# Stages that only add files to a recorded cell's folder: judge re-grades, the judge
# classifier and the contamination probe.
ADDITIVE_STAGES = ("probe", "grade", "classify")


class RunError(RuntimeError):
    """The cell cannot run as asked, or a stage failed."""


@dataclass
class Step:
    stage: str
    seed: Optional[int] = None
    extra: list = field(default_factory=list)   # e.g. ["--condition", "lora+bodhi"]
    env: str = "infer"                          # infer, train or cpu
    done: Optional[Path] = None                 # the manifest that marks it complete

    @property
    def label(self) -> str:
        parts = [self.stage] + ([f"s{self.seed}"] if self.seed is not None else [])
        parts += [x for x in self.extra if not x.startswith("--")]
        return "_".join(parts).replace("+", "-")


def is_complete(manifest_path: Optional[Path]) -> bool:
    if manifest_path is None or not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text()).get("status") == "complete"
    except json.JSONDecodeError:
        return False


# -- prepare --------------------------------------------------------------------------

def _write_once(path: Path, data, what: str) -> None:
    """Write a JSON input; refuse to change one an earlier run of the cell wrote."""
    text = json.dumps(data, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise RunError(f"{path} differs from what this run computes ({what}). The data, the "
                       "split or the cell record changed since the cell first ran; a changed "
                       "design is a new cell.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def same_record(cell_yaml: Path, cell: CellConfig) -> bool:
    """Whether the frozen cell.yaml holds the same record as the registry, value by value.

    It is read with the current schema, so a key the schema gained after the first run
    takes its default in both, and a resume does not stop on it (the R2 folders were
    frozen before serving.enforce_eager existed). The file keeps
    the form of the first run.
    """
    try:
        return CellConfig.model_validate(yaml.safe_load(cell_yaml.read_text())) == cell
    except (ValidationError, yaml.YAMLError):
        return False


def prepare(ctx: StageContext, fetch: bool = True) -> dict:
    """Write the cell record, the pools and every seed's eval ids into the cell's folder."""
    cell, bench = ctx.cell, ctx.bench
    inputs = ctx.paths.inputs()
    cell_yaml = ctx.paths.root / "cell.yaml"
    if cell_yaml.is_file():
        if not same_record(cell_yaml, cell):
            raise RunError(f"{cell_yaml} differs from configs/cells/{cell.id}.yaml: the record changed "
                           "since the cell first ran; a changed design is a new cell.")
    else:
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        cell_yaml.write_text(yaml.safe_dump(cell.model_dump(mode="json"), sort_keys=False))
    rows = load_rows(bench, ctx.project.data_root, fetch=fetch)
    by_id = {r.prompt_id: r for r in rows}
    rule = bench.splits[cell.split]
    fixed = None
    if cell.eval.ids.policy == "fixed_file":
        path = inputs / cell.eval.ids.file
        if not path.is_file():
            raise RunError(f"{path} is missing: a fixed_file cell keeps its eval ids there")
        fixed = read_id_list(path)
    split = split_rows(bench, cell.split, rows, fixed if rule.method == "eval_ids_only" else None)
    _write_once(inputs / POOLS_FILE, {"split": split.name, "rule": rule.model_dump(mode="json"),
                                      "train": split.train, "eval": split.eval, "probe": split.probe,
                                      "text_twins": split.text_twins},
                "pools")
    ids = {}
    for seed in cell.seeds:
        ids[seed] = eval_ids(cell, split, by_id, seed, inputs)
        record = {"seed": seed, "policy": cell.eval.ids.policy, "n": cell.eval.ids.n}
        if cell.eval.ids.unit != "row":
            # Only then, so the files of row-unit runs keep the form they had.
            record["unit"] = cell.eval.ids.unit
        _write_once(inputs / eval_ids_file(seed), {**record, "prompt_ids": ids[seed]},
                    f"eval ids of seed {seed}")
    ctx._rows = by_id
    counts = {"n_rows": len(rows), "n_train_pool": len(split.train), "n_eval_pool": len(split.eval),
              "n_probe_pool": len(split.probe), "n_train_text_twins": len(split.text_twins),
              "n_eval_ids": {str(s): len(v) for s, v in ids.items()}}
    ctx.write_manifest(inputs, "inputs", counts=counts,
                       config={"benchmark": bench.id, "split": cell.split, "eval_ids": cell.eval.ids.model_dump()},
                       outputs=[inputs / POOLS_FILE] + [inputs / eval_ids_file(s) for s in cell.seeds],
                       extra={"status": "complete", "source": bench.source.model_dump(mode="json"),
                              "source_read": source_hashes(bench, ctx.project.data_root)})
    print(f"prepare: {counts['n_rows']} rows; pools train {counts['n_train_pool']}, eval "
          f"{counts['n_eval_pool']}, probe {counts['n_probe_pool']}", flush=True)
    return counts


def leakage_gate(ctx: StageContext) -> dict:
    """No graded trace may be a held-out prompt (EVAL-12); report in shared/inputs/leakage.json."""
    graded = filter_stage.read_jsonl(ctx.paths.shared("filter") / "graded.jsonl")
    rows = ctx.rows_by_id()
    trained = {}
    for g in graded:
        last = [m for m in g["messages"] if m.get("role") == "user"]
        trained[g["prompt_id"]] = " ".join(str(last[-1]["content"]).lower().split()) if last else ""
    split = ctx.pools()
    held = set(split.eval) | set(split.probe)
    ids = {s: ctx.eval_ids(s) for s in ctx.cell.seeds}
    held |= {pid for v in ids.values() for pid in v}
    held_texts = {pid: row_text(rows[pid]) for pid in held}
    population = eval_population(ctx.cell, split, rows)
    path = ctx.paths.inputs() / "leakage.json"
    try:
        report = check_leakage(trained, split, held_texts, ids, population)
    except LeakageError as e:
        path.write_text(json.dumps(e.report, indent=2) + "\n")
        raise RunError(str(e)) from e
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"leakage gate passed: {report['n_trained_prompts']} traced prompts, "
          f"{report['n_held_out']} held out", flush=True)
    return report


def adapter_leakage_gate(ctx: StageContext) -> dict:
    """An eval-only cell's eval prompts against what its adapters trained on; report in
    shared/inputs/leakage.json."""
    kept = ctx.project.cell(ctx.cell.adapters_from).shared("filter") / "kept.jsonl"
    if not kept.is_file():
        raise RunError(f"{kept} is missing: run {ctx.cell.adapters_from} (at least its filter stage) "
                       "first; its kept traces are what the adapters trained on")
    trained = {}
    for g in filter_stage.read_jsonl(kept):
        users = [m for m in g["messages"] if m.get("role") == "user"]
        trained[g["prompt_id"]] = str(users[-1]["content"]) if users else ""
    rows = ctx.rows_by_id()
    eval_texts = {pid: row_text(rows[pid]) for s in ctx.cell.seeds for pid in ctx.eval_ids(s)}
    path = ctx.paths.inputs() / "leakage.json"
    try:
        report = check_adapter_leakage(trained, eval_texts)
    except LeakageError as e:
        path.write_text(json.dumps(e.report, indent=2) + "\n")
        raise RunError(str(e)) from e
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"adapter leakage gate passed: {report['n_eval_prompts']} eval prompts against "
          f"{report['n_trained_prompts']} trained prompts of {ctx.cell.adapters_from}; largest "
          f"{NGRAM}-word share {report['max_share']}", flush=True)
    return report


# -- the plan -------------------------------------------------------------------------

def plan(ctx: StageContext, stages: Sequence[str], seeds: Sequence[int],
         judges: Sequence[str]) -> tuple[list, dict]:
    """The shared steps, and the steps of each seed, in run order."""
    cell, p = ctx.cell, ctx.paths
    shared, per_seed = [], {s: [] for s in seeds}
    trains_here = cell.adapters_from is None and cell.train is not None
    if "probe" in stages and cell.probe:
        shared.append(probe_step(ctx))
    if "traces" in stages and trains_here:
        shared.append(Step("traces", done=p.shared("traces") / "manifest.json"))
    if "filter" in stages and trains_here:
        shared.append(Step("filter", done=p.shared("filter") / "manifest.json"))
    for s in seeds:
        steps = per_seed[s]
        if "filter" in stages and trains_here:
            steps.append(Step("filter", s, env="cpu", done=p.stage(s, "filter") / "manifest.json"))
        if "train" in stages and trains_here:
            steps.append(Step("train", s, env="train", done=p.stage(s, "train") / "manifest.json"))
        if "eval" in stages:
            for c in cell.eval.conditions:
                steps.append(Step("eval", s, ["--condition", c],
                                  done=p.stage(s, "eval") / f"{c}.manifest.json"))
        if "grade" in stages:
            steps.append(Step("grade", s, done=p.stage(s, "grade") / "epistemic.manifest.json"))
            steps += [judge_step(ctx, s, j) for j in judges]
        if "classify" in stages:
            steps += [classify_step(ctx, s, j) for j in judges]
        if "bench" in stages and cell.bench and s in cell.bench.seeds:
            steps.append(Step("bench", s, done=p.stage(s, "bench") / "manifest.json"))
    return shared, per_seed


def probe_step(ctx: StageContext) -> Step:
    return Step("probe", done=ctx.paths.shared("probe") / "manifest.json")


def fetch_benchmarks(ctx: StageContext, stages: Sequence[str], fetch: bool = True) -> None:
    """Build every benchmark the stages read in the data root, here and not in a GPU stage:
    the cell's own, and the probe's control set. With fetch False (a dry run), only check
    that they are already local."""
    benches = [ctx.bench] + ([ctx.reg.benchmarks[ctx.cell.probe.control]]
                             if "probe" in stages and ctx.cell.probe else [])
    for bench in benches:
        try:
            load_rows(bench, ctx.project.data_root, fetch=fetch)
        except DataError as e:
            if fetch:
                raise
            raise RunError(f"dry run: {e}. A dry run never downloads; after the fetch it checks the rest") from e


def judge_step(ctx: StageContext, seed: int, judge: str) -> Step:
    return Step("grade", seed, ["--judge", judge],
                done=ctx.paths.stage(seed, "grade") / f"judge__{judge}.manifest.json")


def classify_step(ctx: StageContext, seed: int, judge: str) -> Step:
    return Step("classify", seed, ["--judge", judge],
                done=ctx.paths.stage(seed, "classify") / f"classify__{judge}.manifest.json")


def gpu_of(seeds: Sequence[int], gpus: Sequence[int]) -> dict[int, int]:
    """The GPU each seed runs on: seeds go out in waves of len(gpus), one per GPU."""
    return {s: gpus[i % len(gpus)] for i, s in enumerate(seeds)}


# -- running a step ---------------------------------------------------------------------

@dataclass
class Runner:
    ctx: StageContext
    mode: str
    gpus: Sequence[int] = ()
    configs_dir: Path = CONFIGS_DIR
    # The folder the caller named, or None for the project's results root. GPU commands
    # carry it; stub commands write under ctx.project.results_root, which may be the
    # scratch root.
    results_root: Optional[Path] = None
    force: bool = False
    _print_lock: threading.Lock = field(default_factory=threading.Lock)

    def say(self, msg: str) -> None:
        with self._print_lock:
            print(msg, flush=True)

    def command(self, step: Step, gpu_python: bool) -> list[str]:
        proj: Project = self.ctx.project
        if gpu_python:
            python = str(proj.python_train if step.env == "train" else proj.python_infer)
        else:
            python = sys.executable
        cmd = [python, "-m", f"bodhi_lora.pipeline.{step.stage}", "--cell", self.ctx.cell.id]
        if step.seed is not None:
            cmd += ["--seed", str(step.seed)]
        cmd += list(step.extra)
        if Path(self.configs_dir).resolve() != CONFIGS_DIR.resolve():
            cmd += ["--configs", str(self.configs_dir)]
        root = self.results_root if gpu_python else proj.results_root
        if root is not None:
            cmd += ["--results-root", str(root)]
        return cmd

    def gpu_env(self, gpu: int) -> dict:
        return {"CUDA_VISIBLE_DEVICES": str(gpu), "BODHI_VLLM_PORT": str(self.ctx.project.vllm_port_base + gpu)}

    def print_plan(self, shared: list, per_seed: dict) -> None:
        """Every step's GPU command, on the GPU a GPU run would give it."""
        placed = [(step, self.gpus[0]) for step in shared]
        on = gpu_of(list(per_seed), self.gpus)
        placed += [(step, on[s]) for s, steps in per_seed.items() for step in steps]
        for step, gpu in placed:
            if step.env == "cpu":
                self.say(f"DRY-RUN [cpu]: {step.label} (runs in the runner)")
                continue
            prefix = " ".join(f"{k}={v}" for k, v in self.gpu_env(gpu).items())
            self.say(f"DRY-RUN [gpu {gpu}]: {prefix} {shlex.join(self.command(step, gpu_python=True))}")

    def run_step(self, step: Step, gpu: int) -> None:
        if not self.force and is_complete(step.done):
            self.say(f"[{step.label}] complete, skipping ({self.ctx.project.rel(step.done)})")
            return
        if step.env == "cpu":
            self.say(f"[{step.label}] cpu")
            filter_stage.split(self.ctx, step.seed)
            return
        if self.mode == "gpu":
            cmd, env = self.command(step, gpu_python=True), self.gpu_env(gpu)
        else:
            if self.mode == "dry_run" and step.stage == "train":
                self.say(f"[{step.label}] dry run: not trained")
                return
            cmd = self.command(step, gpu_python=False)
            if step.stage != "train":
                cmd.append("--stub-engine")
            env = {}
        log = self.ctx.log_path(f"{step.label}.log")
        self.say(f"[{step.label}] {'gpu ' + str(gpu) if self.mode == 'gpu' else self.mode} -> "
                 f"{self.ctx.project.rel(log)}")
        with open(log, "a") as fh:
            fh.write(f"$ {shlex.join(cmd)}\n")
            fh.flush()
            # The package's own checkout, so `python -m bodhi_lora...` imports this code even
            # when the registry (and so the project root) lives elsewhere.
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                                 env={**os.environ, **env})
        if res.returncode != 0:
            tail = "".join(log.read_text().splitlines(keepends=True)[-25:])
            raise RunError(f"stage {step.label} failed (exit {res.returncode}); end of {log}:\n{tail}")

    # -- pointer manifests: every seed folder gets a manifest per stage -----------------

    def write_seed_pointers(self, seed: int, stages: Sequence[str]) -> None:
        ctx, p = self.ctx, self.ctx.paths
        source = ctx.cell.traces_from or ctx.cell.id
        if "traces" in stages and ctx.cell.adapters_from is None and ctx.cell.train is not None:
            shared = ctx.project.cell(source).shared("traces") / "manifest.json"
            if is_complete(shared):
                _pointer(ctx, seed, "traces", shared)
        if ctx.cell.adapters_from:
            src = ctx.project.cell(ctx.cell.adapters_from)
            for stage in ("traces", "filter", "train"):
                m = src.stage(seed, stage) / "manifest.json"
                if m.is_file():
                    _pointer(ctx, seed, stage, m, note=f"adapters of {ctx.cell.adapters_from}")
        for stage in ("eval", "grade"):
            if stage not in stages:
                continue
            folder = p.stage(seed, stage)
            parts = sorted(folder.glob("*.manifest.json")) if folder.is_dir() else []
            done = [m for m in parts if is_complete(m)]
            expected = (len(ctx.cell.eval.conditions) if stage == "eval"
                        else 1 + len(ctx.cell.grading.judges))
            ctx.write_manifest(folder, stage, seed=seed,
                               extra={"status": "complete" if len(done) == expected else "incomplete",
                                      "parts": {m.name: mf.sha256_file(m) for m in done}})


def _pointer(ctx: StageContext, seed: int, stage: str, target: Path, note: str = "") -> None:
    ctx.write_manifest(ctx.paths.stage(seed, stage), stage, seed=seed,
                       extra={"status": "complete", "points_to": ctx.project.rel(target),
                              "points_to_sha256": mf.sha256_file(target), "note": note})


# -- the run ------------------------------------------------------------------------------

def run_cell(reg: Registry, proj: Project, cell_id: str, *, stages: Sequence[str] = DEFAULT_STAGES,
             seeds: Optional[Sequence[int]] = None, gpus: Sequence[int] = (), mode: str = "gpu",
             configs_dir: Path = CONFIGS_DIR, results_root: Optional[Path] = None,
             force: bool = False, judges: Optional[Sequence[str]] = None) -> Path:
    """Run the stages of one cell; return the cell's results folder."""
    if mode not in MODES:
        raise RunError(f"mode {mode!r}: use one of {MODES}")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise RunError(f"unknown stages {sorted(unknown)}; stages are {STAGES}")
    if cell_id not in reg.cells:
        raise RunError(f"no cell {cell_id!r}")
    cell = reg.cells[cell_id]
    judges = list(cell.grading.judges if judges is None else judges)
    not_judges = set(judges) - set(cell.grading.judges)
    if not_judges:
        raise RunError(f"{sorted(not_judges)} are not judges of {cell_id}: {cell.grading.judges}")
    stored = proj.cell(cell_id)
    if results_root is not None:
        results_root = Path(results_root).resolve()
        # The scratch rule above holds for a named root too: a stub manifest that says
        # complete under the results root would make a later GPU run keep stub outputs.
        if mode != "gpu" and results_root.is_relative_to(proj.results_root.resolve()):
            raise RunError(f"a {mode} run may not write under the results root "
                           f"({proj.rel(proj.results_root)}); leave out --results-root or name "
                           f"a folder outside it")
        proj = replace(proj, results_root=results_root)
    elif mode != "gpu":
        proj = replace(proj, results_root=(proj.scratch_root / mode).resolve())
    ctx = StageContext(reg=reg, project=proj, cell=cell, paths=proj.cell(cell_id), stub=mode != "gpu")
    seeds = list(seeds) if seeds else list(cell.seeds)
    missing = set(seeds) - set(cell.seeds)
    if missing:
        raise RunError(f"seeds {sorted(missing)} are not seeds of {cell_id}: {cell.seeds}")
    if cell.train is not None and cell.train.engine == "external" and {"traces", "filter", "train"} & set(stages):
        raise RunError(f"{cell_id} was trained outside this package ({cell.train.record}); "
                       "run --stages eval,grade on it, or make a new cell")
    recorded = cell.report.status in RECORD_STATUSES
    if recorded and not set(stages) <= set(ADDITIVE_STAGES):
        raise RunError(f"{cell_id} records a past run (status {cell.report.status}); its outputs "
                       f"stay as they are. Only --stages {','.join(ADDITIVE_STAGES)} adds files to "
                       "it (judge re-grades). A new run of this design is a new cell.")
    problems = family_problems(cell, reg)
    if problems:
        raise RunError("family rule: " + "; ".join(problems))
    if mode == "gpu":
        if not gpus:
            raise RunError("give the GPUs to use with --gpus, e.g. --gpus 0,1")
        for py in (proj.python_infer, proj.python_train):
            if not Path(py).exists():
                raise RunError(f"{py} is missing; build the environments (setup.sh) or use --stub-engine")
    gpus = list(gpus) or [0]
    runner = Runner(ctx, mode, gpus, configs_dir, results_root, force)
    if mode != "gpu":
        runner.say(f"{mode}: writing under {proj.rel(proj.results_root)}")
    if recorded:
        return _add_to_recorded(runner, stored, stages, seeds, judges)
    timer = mf.Timer()

    def write_cell_manifest(status: str) -> None:
        index = {}
        for s in cell.seeds:
            for stage in STAGES:
                m = ctx.paths.stage(s, stage) / "manifest.json"
                if m.is_file():
                    index[f"seed_{s}/{stage}"] = mf.sha256_file(m)
        models = {"base": reg.models[cell.model], "teacher": reg.teacher(cell),
                  "filter": reg.models[cell.grading.filter],
                  "evaluator": reg.models[cell.grading.evaluator],
                  **{f"judge:{j}": reg.models[j] for j in cell.grading.judges}}
        ctx.write_manifest(ctx.paths.root, "cell", timer=timer, models=models,
                           config=cell.model_dump(mode="json"),
                           extra={"status": status, "mode": mode, "seeds_run": seeds,
                                  "stages_run": list(stages), "judges_run": judges,
                                  "family": {"problems": problems,
                                             "waiver": cell.grading.same_family_waiver},
                                  "stage_manifests": index})

    shared, per_seed = plan(ctx, stages, seeds, judges)
    # A dry run prints its GPU commands first, then checks the rest on the CPU without
    # downloading anything: a benchmark that is not local stops it with the fetch command.
    fetch = mode != "dry_run"
    if not fetch:
        runner.print_plan(shared, per_seed)
    fetch_benchmarks(ctx, stages, fetch)
    prepare(ctx, fetch=fetch)
    write_cell_manifest("started")
    try:
        if cell.adapters_from and "eval" in stages:
            adapter_leakage_gate(ctx)
        for step in shared:
            runner.run_step(step, gpus[0])
            if step.stage == "filter":
                leakage_gate(ctx)
        if mode == "gpu" and len(gpus) > 1:
            _run_seeds_parallel(runner, per_seed, stages, gpus)
        else:
            for s in seeds:
                for step in per_seed[s]:
                    runner.run_step(step, gpus[0])
                runner.write_seed_pointers(s, stages)
    except Exception:
        write_cell_manifest("failed")
        raise
    write_cell_manifest("finished")
    runner.say(f"cell {cell_id}: done -> {proj.rel(ctx.paths.root)}")
    return ctx.paths.root


def link_stored_outputs(stored: CellPaths, scratch: CellPaths, cell, seeds: Sequence[int]) -> None:
    """Symlink a recorded cell's eval outputs and inputs into a scratch copy of its folder.

    A stub or dry run of a judge re-grade then reads the real stored replies and writes
    only under the scratch root. A missing file is left missing, so the stage reports it.
    """
    pairs = [(stored.eval_file(s, c), scratch.eval_file(s, c)) for s in seeds for c in cell.eval.conditions]
    if stored.inputs().is_dir():
        pairs += [(f, scratch.inputs() / f.name) for f in stored.inputs().iterdir() if f.is_file()]
    for src, dst in pairs:
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())


def _add_to_recorded(runner: Runner, stored: CellPaths, stages: Sequence[str], seeds: Sequence[int],
                     judges: Sequence[str]) -> Path:
    """Run the additive stages of a recorded cell: no prepare, no cell manifest."""
    ctx = runner.ctx
    runner.say(f"{ctx.cell.id}: recorded run; adding files only ({', '.join(stages)})")
    if runner.mode != "gpu" and stored.root != ctx.paths.root:
        link_stored_outputs(stored, ctx.paths, ctx.cell, seeds)
    shared = [probe_step(ctx)] if "probe" in stages and ctx.cell.probe else []
    steps = [judge_step(ctx, s, j) for s in seeds for j in judges] if "grade" in stages else []
    steps += [classify_step(ctx, s, j) for s in seeds for j in judges] if "classify" in stages else []
    if runner.mode == "dry_run":
        runner.print_plan(shared, {s: [st for st in steps if st.seed == s] for s in seeds})
    # After the plan, as in run_cell: a dry run prints its GPU commands even when a
    # benchmark is not local, and never downloads one.
    fetch_benchmarks(ctx, stages, fetch=runner.mode != "dry_run")
    for step in shared + steps:
        runner.run_step(step, runner.gpus[0])
    return ctx.paths.root


def _run_seeds_parallel(runner: Runner, per_seed: dict, stages, gpus) -> None:
    """One seed per GPU at a time; a failure stops the run after the current wave."""
    seeds = list(per_seed)
    for i in range(0, len(seeds), len(gpus)):
        wave = list(zip(seeds[i:i + len(gpus)], gpus))
        errors = []

        def work(seed, gpu):
            try:
                for step in per_seed[seed]:
                    runner.run_step(step, gpu)
                runner.write_seed_pointers(seed, stages)
            except Exception as e:      # reported after the wave
                errors.append(e)

        threads = [threading.Thread(target=work, args=w) for w in wave]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
