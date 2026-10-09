"""The package names no model, benchmark, results path or run (docs/ARCHITECTURE.md section 9).

Every string constant in bodhi_lora/**/*.py is read with ast: plain strings, docstrings
and the literal parts of f-strings. Comments are not read, so a comment may still say
why a line exists. The names come from the registry, so a new record extends the check.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pydantic")

from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

PACKAGE = REPO_ROOT / "bodhi_lora"
# Rule 4: words that tie a module to one run, venue or machine.
RUN_WORDS = ("modal", "tpu", "gcs", "rebuttal", "submitted", "posted", "neurips", "reviewer")
# Rule 5: the five seeds of the 2026-05 and 2026-07 cells.
PAPER_SEEDS = [7, 13, 42, 99, 101]


def _is_seed_list(node) -> bool:
    """A list, tuple or set literal of exactly the paper's five seeds, in any order."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return False
    values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
    return (len(values) == len(node.elts) == len(PAPER_SEEDS)
            and all(type(v) is int for v in values) and sorted(values) == PAPER_SEEDS)


def _token(name: str) -> re.Pattern:
    """A whole-token match: no letter or digit directly before or after the name."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _forbidden() -> dict[str, re.Pattern]:
    reg = load_registry(CONFIGS_DIR)
    names = {}
    for m in reg.models.values():
        names[f"model {m.id}"] = _token(m.id)
        names[f"model repo {m.hf_id}"] = _token(m.hf_id)
    for b in reg.benchmarks.values():
        names[f"benchmark {b.id}"] = _token(b.id)
        src = b.source
        if src.kind == "hf":
            names[f"dataset {src.dataset}"] = _token(src.dataset)
        elif src.kind == "url":
            for f in src.files.values():
                names[f"url {f.url}"] = re.compile(re.escape(f.url))
    project = yaml.safe_load((CONFIGS_DIR / "project.yaml").read_text())
    # results_root is the config key that names the root, not a results folder.
    names["a results_<word> folder"] = re.compile(r"(?<![A-Za-z0-9_])results_(?!root\b)[a-z]+")
    for key in ("results_root", "data_root"):
        root = project[key].rstrip("/") + "/"
        names[f"{key} path {root}"] = re.compile(r"(?<![A-Za-z0-9_/])" + re.escape(root))
    return names


def _strings(path: Path):
    """(line, text) of every string constant in a module, f-string parts included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _modules():
    return sorted(PACKAGE.rglob("*.py"))


def test_no_model_benchmark_or_results_path_in_package_strings():
    forbidden = _forbidden()
    hits = []
    for path in _modules():
        rel = path.relative_to(REPO_ROOT)
        for line, text in _strings(path):
            for what, pat in forbidden.items():
                if pat.search(text):
                    hits.append(f"{rel}:{line}: {what}: {text.strip()[:80]!r}")
    assert hits == [], "package strings name a record or a results path:\n" + "\n".join(hits)


def test_no_run_word_in_package_file_names():
    bad = [str(p.relative_to(REPO_ROOT)) for p in PACKAGE.rglob("*")
           if "__pycache__" not in p.parts
           and any(w in part.lower() for part in p.relative_to(PACKAGE).parts for w in RUN_WORDS)]
    assert bad == []


def test_no_paper_seed_list_literal():
    hits = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits += [f"{path.relative_to(REPO_ROOT)}:{n.lineno}" for n in ast.walk(tree) if _is_seed_list(n)]
    assert hits == []


def test_the_check_catches_what_it_should(tmp_path):
    """The rules fire on a module that breaks each one, so a passing run means something."""
    forbidden = _forbidden()
    bad = tmp_path / "bad.py"
    bad.write_text('"""Grades HealthBench."""\n'
                   'MODEL = "google/medgemma-27b-text-it"\n'
                   'OUT = f"results/{cell}"\n'
                   'OLD = "results_modal"\n'
                   'SEEDS = (42, 7, 13, 99, 101)\n')
    texts = [t for _, t in _strings(bad)]
    fired = {what.split()[0] for what, pat in forbidden.items() for t in texts if pat.search(t)}
    assert {"benchmark", "model", "results_root", "a"} <= fired
    assert any(_is_seed_list(n) for n in ast.walk(ast.parse(bad.read_text())))
    assert not any(_is_seed_list(n) for n in ast.walk(ast.parse("S = (42, 7, 13)\nT = ('a', None)")))
