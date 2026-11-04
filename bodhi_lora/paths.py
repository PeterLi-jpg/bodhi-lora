"""Where a run reads and writes: the project roots and the folders of one cell.

configs/project.yaml names the results root and the data root. Every other path is
derived here, so no other module builds a results or data path by hand:

    <results_root>/<cell_id>/
        shared/<stage>/       output that every seed shares (inputs, traces, filter, probe)
        seed_<n>/<stage>/     one folder per seed and stage, each with a manifest.json
        seed_<n>/table.csv    the per-response table the analysis reads
        analysis/             the cell's statistics
        validation/           physician grading sheets and agreement, where a cell has them
        logs/                 stage logs (ignored by git)
    <data_root>/<benchmark_id>/   downloaded and built benchmark files
    <scratch_root>/<mode>/        stub and dry runs, laid out like the results root
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from bodhi_lora.registry.schema import ProjectConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

STAGES = ("probe", "traces", "filter", "train", "eval", "grade", "classify", "bench")


@dataclass(frozen=True)
class Project:
    repo_root: Path
    results_root: Path
    data_root: Path
    scratch_root: Path
    python_train: Path
    python_infer: Path
    vllm_port_base: int

    def cell(self, cell_id: str) -> "CellPaths":
        return CellPaths(self.results_root / cell_id)

    def benchmark_dir(self, benchmark_id: str) -> Path:
        return self.data_root / benchmark_id

    def rel(self, path: Path) -> str:
        """A path relative to the repository root when it is inside it, for records."""
        path = Path(path)
        try:
            return str(path.resolve().relative_to(self.repo_root.resolve()))
        except ValueError:
            return str(path)


def project(configs_dir: Path = CONFIGS_DIR) -> Project:
    """Read configs/project.yaml. Its paths are relative to the folder above configs/."""
    configs_dir = Path(configs_dir)
    root = configs_dir.resolve().parent
    cfg = ProjectConfig.model_validate(
        yaml.safe_load((configs_dir / "project.yaml").read_text(encoding="utf-8")))
    return Project(repo_root=root, results_root=root / cfg.results_root,
                   data_root=root / cfg.data_root, scratch_root=root / cfg.scratch_root,
                   python_train=root / cfg.python.train,
                   python_infer=root / cfg.python.infer, vllm_port_base=cfg.vllm_port_base)


@dataclass(frozen=True)
class CellPaths:
    """The folders of one cell under the results root."""

    root: Path

    def shared(self, stage: str) -> Path:
        return self.root / "shared" / stage

    def inputs(self) -> Path:
        """Pool ids, eval ids per seed and the leakage report."""
        return self.shared("inputs")

    def seed(self, seed: int) -> Path:
        return self.root / f"seed_{seed}"

    def stage(self, seed: int, stage: str) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
        return self.seed(seed) / stage

    def table(self, seed: int) -> Path:
        return self.seed(seed) / "table.csv"

    def analysis(self) -> Path:
        return self.root / "analysis"

    def validation(self) -> Path:
        return self.root / "validation"

    def logs(self) -> Path:
        return self.root / "logs"

    def eval_file(self, seed: int, condition: str) -> Path:
        """The responses and rubric grades of one condition."""
        return self.stage(seed, "eval") / f"{condition}.json"

    def grade_file(self, seed: int, evaluator_id: str) -> Path:
        """The evaluator's seven-dimension grades of every condition."""
        return self.stage(seed, "grade") / f"epistemic__{evaluator_id}.json"

    def judge_file(self, seed: int, judge_id: str) -> Path:
        """A judge's re-grade (rubric and dimensions) of every condition."""
        return self.stage(seed, "grade") / f"judge__{judge_id}.json"

    def adapter_dir(self, seed: int) -> Path:
        """The seed's adapter: the train folder itself, with checkpoint-<k>/ inside it."""
        return self.stage(seed, "train")
