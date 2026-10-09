"""Run a cell: prepare its inputs, then run each stage in order, for each seed.

    run_cell(reg, proj, cell_id, stages=..., seeds=..., gpus=[0, 1], mode="gpu")

Modes:
  gpu      every model stage runs as a subprocess in its Python environment
           (configs/project.yaml: python.infer for vLLM stages, python.train for
           training), pinned to one GPU by CUDA_VISIBLE_DEVICES, its vLLM server on
           vllm_port_base + GPU index. Seeds run in parallel, one per GPU.
  stub     the CPU stub engine answers for vLLM, and everything else runs for real,
           training included, in the current interpreter (the smoke cell).
  dry_run  prints the GPU command of every stage, then runs like stub but prints the
           training command instead of training.

Order: prepare (pools, eval ids, cell.yaml), probe, traces, filter (grading), the
leakage gate, then per seed: filter (split), train, eval (each condition), grade (the
evaluator, then each judge), bench. A stage whose manifest says complete is skipped, so
a run resumes where it stopped; prepare refuses inputs that differ from an earlier run.

A cell whose report.status records a past run (complete, partial, not_run, pilot) is
refused, so new outputs never mix with recorded ones; a new run of the same design is
a new cell. The exception is a judge re-grade, which only adds files. With
results_root set to another folder, any cell runs there (tests, dry runs).
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

from bodhi_lora import manifest as mf
from bodhi_lora.data import check_leakage, eval_ids, eval_population, load_rows, read_id_list, row_text, split_rows
from bodhi_lora.data.leakage import LeakageError
from bodhi_lora.paths import CONFIGS_DIR, Project
from bodhi_lora.pipeline import filter as filter_stage
from bodhi_lora.pipeline.context import POOLS_FILE, StageContext, eval_ids_file
from bodhi_lora.registry import RECORD_STATUSES, Registry, family_problems

STAGES = ("probe", "traces", "filter", "train", "eval", "grade", "bench")
MODES = ("gpu", "stub", "dry_run")


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


def prepare(ctx: StageContext, fetch: bool = True) -> dict:
    """Write the cell record, the pools and every seed's eval ids into the cell's folder."""
    cell, bench = ctx.cell, ctx.bench
    inputs = ctx.paths.inputs()
    record = yaml.safe_dump(cell.model_dump(mode="json"), sort_keys=False)
    cell_yaml = ctx.paths.root / "cell.yaml"
    if cell_yaml.is_file() and cell_yaml.read_text() != record:
        raise RunError(f"{cell_yaml} differs from configs/cells/{cell.id}.yaml: the record changed "
                       "since the cell first ran; a changed design is a new cell.")
    ctx.paths.root.mkdir(parents=True, exist_ok=True)
    cell_yaml.write_text(record)
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
                                      "train": split.train, "eval": split.eval, "probe": split.probe},
                "pools")
    ids = {}
    for seed in cell.seeds:
        ids[seed] = eval_ids(cell, split, by_id, seed, inputs)
        _write_once(inputs / eval_ids_file(seed), {"seed": seed, "policy": cell.eval.ids.policy,
                                                   "n": cell.eval.ids.n, "prompt_ids": ids[seed]},
                    f"eval ids of seed {seed}")
    ctx._rows = by_id
    counts = {"n_rows": len(rows), "n_train_pool": len(split.train), "n_eval_pool": len(split.eval),
              "n_probe_pool": len(split.probe), "n_eval_ids": {str(s): len(v) for s, v in ids.items()}}
    ctx.write_manifest(inputs, "inputs", counts=counts,
                       config={"benchmark": bench.id, "split": cell.split, "eval_ids": cell.eval.ids.model_dump()},
                       outputs=[inputs / POOLS_FILE] + [inputs / eval_ids_file(s) for s in cell.seeds],
                       extra={"status": "complete", "source": bench.source.model_dump(mode="json")})
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


# -- the plan -------------------------------------------------------------------------

def plan(ctx: StageContext, stages: Sequence[str], seeds: Sequence[int]) -> tuple[list, dict]:
    """The shared steps, and the steps of each seed, in run order."""
    cell, p = ctx.cell, ctx.paths
    shared, per_seed = [], {s: [] for s in seeds}
    trains_here = cell.adapters_from is None and cell.train is not None
    if "probe" in stages and cell.probe:
        shared.append(Step("probe", done=p.shared("probe") / "manifest.json"))
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
            for j in cell.grading.judges:
                steps.append(Step("grade", s, ["--judge", j],
                                  done=p.stage(s, "grade") / f"judge__{j}.manifest.json"))
        if "bench" in stages and cell.bench:
            steps.append(Step("bench", s, done=p.stage(s, "bench") / "manifest.json"))
    return shared, per_seed


# -- running a step ---------------------------------------------------------------------

@dataclass
class Runner:
    ctx: StageContext
    mode: str
    gpus: Sequence[int] = ()
    configs_dir: Path = CONFIGS_DIR
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
        if self.results_root is not None:
            cmd += ["--results-root", str(self.results_root)]
        return cmd

    def gpu_env(self, gpu: int) -> dict:
        return {"CUDA_VISIBLE_DEVICES": str(gpu), "BODHI_VLLM_PORT": str(self.ctx.project.vllm_port_base + gpu)}

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
            if self.mode == "dry_run":
                gpu_cmd = self.command(step, gpu_python=True)
                prefix = " ".join(f"{k}={v}" for k, v in self.gpu_env(gpu).items())
                self.say(f"DRY-RUN [gpu {gpu}]: {prefix} {shlex.join(gpu_cmd)}")
                if step.stage == "train":
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
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=self.ctx.project.repo_root,
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

def run_cell(reg: Registry, proj: Project, cell_id: str, *, stages: Sequence[str] = STAGES,
             seeds: Optional[Sequence[int]] = None, gpus: Sequence[int] = (), mode: str = "gpu",
             configs_dir: Path = CONFIGS_DIR, results_root: Optional[Path] = None,
             force: bool = False) -> Path:
    """Run the stages of one cell; return the cell's results folder."""
    if mode not in MODES:
        raise RunError(f"mode {mode!r}: use one of {MODES}")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise RunError(f"unknown stages {sorted(unknown)}; stages are {STAGES}")
    if cell_id not in reg.cells:
        raise RunError(f"no cell {cell_id!r}")
    cell = reg.cells[cell_id]
    if results_root is not None:
        proj = replace(proj, results_root=Path(results_root).resolve())
    ctx = StageContext(reg=reg, project=proj, cell=cell, paths=proj.cell(cell_id), stub=mode != "gpu")
    seeds = list(seeds) if seeds else list(cell.seeds)
    missing = set(seeds) - set(cell.seeds)
    if missing:
        raise RunError(f"seeds {sorted(missing)} are not seeds of {cell_id}: {cell.seeds}")
    judges_only = False
    if cell.report.status in RECORD_STATUSES and results_root is None:
        if set(stages) == {"grade"} and cell.grading.judges:
            judges_only = True
        else:
            raise RunError(f"{cell_id} records a past run (status {cell.report.status}); its outputs "
                           "stay as they are. Only --stages grade adds judge re-grades to it. A new "
                           "run of this design is a new cell.")
    if cell.train is not None and cell.train.engine == "external" and {"traces", "filter", "train"} & set(stages):
        raise RunError(f"{cell_id} was trained outside this package ({cell.train.record}); "
                       "run --stages eval,grade on it, or make a new cell")
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
                                  "stages_run": list(stages),
                                  "family": {"problems": problems,
                                             "waiver": cell.grading.same_family_waiver},
                                  "stage_manifests": index})

    if judges_only:
        runner.say(f"{cell_id}: recorded run; adding judge re-grades only")
        # The grade stage reads the prompts from the data root; build it once here.
        load_rows(ctx.bench, proj.data_root, fetch=True)
        for s in seeds:
            for j in cell.grading.judges:
                runner.run_step(Step("grade", s, ["--judge", j],
                                     done=ctx.paths.stage(s, "grade") / f"judge__{j}.manifest.json"), gpus[0])
        return ctx.paths.root

    prepare(ctx, fetch=True)
    write_cell_manifest("started")
    shared, per_seed = plan(ctx, stages, seeds)
    try:
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
