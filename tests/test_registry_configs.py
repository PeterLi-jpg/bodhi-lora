"""The registry in configs/: every record loads, and the docs' examples match the schema."""
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pydantic")

from bodhi_lora import cli  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT, project  # noqa: E402
from bodhi_lora.registry import (  # noqa: E402
    BenchmarkConfig,
    CellConfig,
    ModelConfig,
    ProtocolConfig,
    family_problems,
    load_registry,
)

RECORD = {"models": ModelConfig, "benchmarks": BenchmarkConfig,
          "protocols": ProtocolConfig, "cells": CellConfig}


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


def test_registry_loads_and_every_cell_passes_the_family_rule(reg):
    assert reg.cells, "no cell records"
    for cell in reg.cells.values():
        assert family_problems(cell, reg) == [], cell.id


def test_cells_that_break_the_family_rule_say_why(reg):
    waived = {cid for cid, c in reg.cells.items() if c.grading.same_family_waiver}
    # The Med42-8B cells share the evaluator's family; the pilot graded with its filter.
    assert {"healthbench__med42-8b__bodhi__2026-07", "chatdoctor__med42-8b__bodhi__2026-07",
            "healthbench__medgemma-27b-text__bodhi__2026-04"} == waived


def test_project_roots_are_inside_the_repo():
    proj = project(CONFIGS_DIR)
    for root in (proj.results_root, proj.data_root):
        assert root.resolve().is_relative_to(REPO_ROOT.resolve())


def test_list_prints_every_cell(reg, capsys):
    assert cli.main(["list", "cells"]) == 0
    out = capsys.readouterr().out
    for cid in reg.cells:
        assert cid in out


def test_list_settings_names_what_differs(reg, capsys):
    assert cli.main(["list", "cells", "--settings"]) == 0
    out = capsys.readouterr().out
    # The 2026-05 cell differs from the 2026-07 cells in its evaluator repo and context.
    assert "grading.evaluator" in out and "eval.serving.max_model_len" in out


EXAMPLE = re.compile(r"```yaml\n# configs/(\w+)/([^\n]+)\.yaml\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("doc", ["docs/ARCHITECTURE.md", "docs/EXPERIMENTS.md"])
def test_doc_examples_match_the_schema(doc):
    """Each example record in the design docs validates against its record class.

    The examples may name records that a build step has not written yet, so only the
    shape is checked here, not the references.
    """
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    found = EXAMPLE.findall(text)
    assert found, f"no example records in {doc}"
    for kind, name, body in found:
        rec = RECORD[kind].model_validate(yaml.safe_load(body))
        assert rec.id == name


def _settings(obj, prefix=""):
    """{"a": {"b": 1}} -> {"a.b": 1}."""
    if not isinstance(obj, dict):
        return {prefix[:-1]: obj}
    out = {}
    for k, v in obj.items():
        out.update(_settings(v, f"{prefix}{k}."))
    return out


@pytest.mark.parametrize("doc", ["docs/ARCHITECTURE.md", "docs/EXPERIMENTS.md"])
def test_doc_examples_show_the_values_of_their_records(doc):
    """An example of a record that exists may leave settings out, but each setting it
    shows has the record's value, so the docs cannot drift from configs/."""
    for kind, name, body in EXAMPLE.findall((REPO_ROOT / doc).read_text(encoding="utf-8")):
        path = CONFIGS_DIR / kind / f"{name}.yaml"
        if not path.is_file():
            continue
        shown = _settings(yaml.safe_load(body))
        record = _settings(yaml.safe_load(path.read_text(encoding="utf-8")))
        differ = {k: (v, record.get(k, "<not in the record>")) for k, v in shown.items() if record.get(k) != v}
        assert not differ, f"{doc}, {kind}/{name}: {differ}"


def test_every_record_file_is_named_after_its_id():
    for kind in RECORD:
        for path in sorted((CONFIGS_DIR / kind).glob("*.yaml")):
            assert yaml.safe_load(path.read_text(encoding="utf-8"))["id"] == path.stem
    assert not list(Path(CONFIGS_DIR).glob("*/*.yml")), "use the .yaml suffix"


def test_run_names_the_install_it_needs_without_bodhi_llm(monkeypatch, capsys):
    """On the CPU install (.[dev]) bodhi-llm is missing; run stops with the install to use
    instead of a traceback from inside a stage (docs/INSTALL.md)."""
    monkeypatch.setattr(cli, "_has_module", lambda name: name != "bodhi")
    assert cli.main(["run", "tiny-rubric__tiny-random-llama__bodhi", "--dry-run"]) == 1
    assert "pip install -e '.[train,dev]'" in capsys.readouterr().out
