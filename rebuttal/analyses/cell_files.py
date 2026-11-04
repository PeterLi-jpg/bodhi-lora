"""Where the three discussion scripts find a cell's raw outputs.

scripts/paper_stats.py held these two helpers until the report moved into
bodhi_lora/report/ (2026-10-10). The scripts are kept as the record of how the posted
values were made, so they keep reading the same files the same way.
"""
from pathlib import Path

from bodhi_lora.registry import condition_from_legacy


def eval_path(cell_dir: Path, seed: int, cfg: str) -> Path:
    """The eval output of one condition, named by its old name (base_no_wrapper, ...)."""
    return cell_dir / f"seed_{seed}" / "eval" / f"{condition_from_legacy(cfg)}.json"


def grade_path(cell_dir: Path, seed: int) -> Path:
    """The evaluator's dimension grades of a seed (one epistemic__<evaluator>.json)."""
    found = sorted((cell_dir / f"seed_{seed}" / "grade").glob("epistemic__*.json"))
    if len(found) != 1:
        raise SystemExit(f"expected one evaluator grade file in {cell_dir}/seed_{seed}/grade, found {len(found)}")
    return found[0]
