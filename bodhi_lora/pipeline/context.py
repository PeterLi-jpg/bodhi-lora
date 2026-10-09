"""What every stage of a cell run needs: the registry, the cell, its folders, its inputs.

Each stage is a module with a command line of the same shape,

    python -m bodhi_lora.pipeline.<stage> --cell <cell id> [--seed N] [...] [--stub-engine]

so the runner (pipeline/runner.py) can start any stage in either Python environment, on
any GPU, and a stage can be rerun by hand. The stage reads every setting from the
registry; the runner's prepare step has already written the cell's pools and eval ids
to shared/inputs/, so every stage of a run sees the same ones.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from bodhi_lora import manifest as mf
from bodhi_lora.data import Row, Split, load_rows
from bodhi_lora.paths import CONFIGS_DIR, CellPaths, Project, project
from bodhi_lora.registry import CellConfig, Registry, load_registry, parse_condition
from bodhi_lora.registry.schema import ModelConfig, ProtocolConfig, Train

POOLS_FILE = "pools.json"


def eval_ids_file(seed: int) -> str:
    return f"eval_ids_seed_{seed}.json"


@dataclass
class StageContext:
    reg: Registry
    project: Project
    cell: CellConfig
    paths: CellPaths
    stub: bool = False
    _rows: Optional[dict] = field(default=None, repr=False)

    # -- records ------------------------------------------------------------------

    @property
    def bench(self):
        return self.reg.benchmarks[self.cell.benchmark]

    @property
    def model(self) -> ModelConfig:
        return self.reg.models[self.cell.model]

    @property
    def protocol(self) -> ProtocolConfig:
        return self.reg.protocols[self.cell.protocol]

    def condition_protocol(self, condition: str) -> tuple[str, Optional[ProtocolConfig]]:
        """(weights, protocol record or None) of a condition such as "lora+bodhi"."""
        weights, proto = parse_condition(condition)
        return weights, (self.reg.protocols[proto] if proto else None)

    def train_cell(self) -> CellConfig:
        """The cell whose adapters this cell evaluates: itself, or adapters_from."""
        return self.reg.cells[self.cell.adapters_from] if self.cell.adapters_from else self.cell

    def train_block(self) -> Optional[Train]:
        return self.train_cell().train

    def quantized_base(self) -> bool:
        """Serve the base in 4-bit: the cell serves it as trained, and training was 4-bit."""
        train = self.train_block()
        return (self.cell.eval.base_precision == "train" and train is not None
                and train.quantization == "4bit")

    # -- paths --------------------------------------------------------------------

    def adapter_dir(self, seed: int) -> Path:
        return self.project.cell(self.train_cell().id).adapter_dir(seed)

    def traces_file(self) -> Path:
        """The teacher traces the cell trains on: its own, or those of traces_from."""
        source = self.cell.traces_from or self.cell.id
        return self.project.cell(source).shared("traces") / "traces.jsonl"

    def log_path(self, name: str) -> Path:
        self.paths.logs().mkdir(parents=True, exist_ok=True)
        return self.paths.logs() / name

    # -- inputs written by the runner's prepare step -----------------------------------

    def rows_by_id(self) -> dict[str, Row]:
        if self._rows is None:
            self._rows = {r.prompt_id: r for r in load_rows(self.bench, self.project.data_root,
                                                            fetch=False)}
        return self._rows

    def pools(self) -> Split:
        data = json.loads((self.paths.inputs() / POOLS_FILE).read_text())
        return Split(name=data["split"], train=data["train"], eval=data["eval"], probe=data["probe"])

    def eval_ids(self, seed: int) -> list[str]:
        data = json.loads((self.paths.inputs() / eval_ids_file(seed)).read_text())
        return data["prompt_ids"]

    # -- manifests ----------------------------------------------------------------

    def write_manifest(self, folder: Path, stage: str, *, seed: Optional[int] = None,
                       name: str = "manifest.json", **kwargs) -> Path:
        return mf.write_manifest(Path(folder) / name,
                                 mf.build_manifest(stage, cell=self.cell.id, seed=seed, **kwargs))


def stage_parser(description: str, *, seed: str = "none", condition: bool = False,
                 judge: bool = False) -> argparse.ArgumentParser:
    """The common command line. seed: "none", "optional" or "required"."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--cell", required=True)
    if seed != "none":
        ap.add_argument("--seed", type=int, required=seed == "required")
    if condition:
        ap.add_argument("--condition", required=True)
    if judge:
        ap.add_argument("--judge", default=None,
                        help="re-grade with this judge (a model id in grading.judges)")
    ap.add_argument("--stub-engine", action="store_true",
                    help="answer with the CPU stub engine instead of starting vLLM")
    ap.add_argument("--configs", type=Path, default=CONFIGS_DIR)
    ap.add_argument("--results-root", type=Path, default=None,
                    help="write under this folder instead of the project's results root")
    return ap


def context_from_args(args) -> StageContext:
    reg = load_registry(args.configs)
    proj = project(args.configs)
    if args.results_root is not None:
        proj = replace(proj, results_root=Path(args.results_root).resolve())
    if args.cell not in reg.cells:
        raise SystemExit(f"no cell {args.cell!r} in {args.configs}/cells")
    return StageContext(reg=reg, project=proj, cell=reg.cells[args.cell],
                        paths=proj.cell(args.cell), stub=args.stub_engine)
