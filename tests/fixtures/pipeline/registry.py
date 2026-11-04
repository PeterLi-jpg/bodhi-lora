"""A registry for pipeline tests: every record of configs/ plus the test records here.

The test records (tests/fixtures/pipeline/registry/<kind>/*.yaml) are tiny twins of the
experiment cells: the tiny random model, fixture benchmarks, a few rows. They live here
and not in configs/, so `python -m bodhi_lora list` shows only real cells.

    configs = fixture_configs(tmp_path)          # tmp_path/configs
    reg, proj = load_registry(configs), project(configs)

project(configs) puts the results, data and scratch roots under tmp_path, so a test
never writes into the repository.
"""
import shutil
from pathlib import Path

from bodhi_lora.paths import CONFIGS_DIR

RECORDS = Path(__file__).resolve().parent / "registry"
KINDS = ("models", "benchmarks", "protocols", "cells")


def fixture_configs(tmp_path: Path) -> Path:
    out = Path(tmp_path) / "configs"
    shutil.copytree(CONFIGS_DIR, out)
    for kind in KINDS:
        folder = RECORDS / kind
        for path in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
            shutil.copy(path, out / kind / path.name)
    return out
