"""No measurement decides what text means by a keyword list or a regular expression
(docs/CONVENTIONS.md rule 15).

Keyword rules miss negation, hypotheticals and quotes, so a reply that says "no need to
call 911" counts as an emergency referral, and a quoted question counts as asking. What a
reply or a prompt means comes from structured fields that ship with a dataset (tags), from
the evaluator's grades, or from a judge with a semantic rubric that passed a hand-label
gate. The one kind of text pattern a config may hold is a parser of a format the protocol
itself defines: a leak rule whose markers are the protocol's own section headers or field
names.

Three checks:
  1. No config file has a key that holds text patterns or word lists, except the markers
     of a protocol record's leak rules.
  2. Every leak marker is one of the protocol's own format fields (Protocol.format_fields),
     matched whole up to a plural ending of at most three letters, so it is a header and
     not a word inside one, and never a free word.
  3. The analysis and label code compiles no regular expression from a string literal; the
     only patterns it compiles are the leak markers of the protocol records.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pydantic")

from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

# Config keys that would hold a text pattern or a word list. Exact key names: a key such
# as fix_mistral_regex (a tokenizer switch) is not one.
PATTERN_KEYS = {"pattern", "patterns", "regex", "regexes", "rubric_pattern", "keyword", "keywords",
                "lexicon", "word_list", "words", "phrases", "markers", "text_features"}
# The code that measures: the uniform analysis and the benchmark labels.
MEASURING_CODE = sorted((REPO_ROOT / "bodhi_lora" / "analysis").glob("*.py")) + [
    REPO_ROOT / "bodhi_lora" / "data" / "labels.py"]
RE_FUNCTIONS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"}
# A marker may stop short of its header by a plural ending ("KEY UNCERTAINT" for
# "KEY UNCERTAINTIES:"), and the header's colon is optional in the marker.
HEADER_TAIL = r"[A-Z]{0,3}\s*:?"


def _keys(obj, path=()):
    """(path, key) for every mapping key in a loaded YAML document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k
            yield from _keys(v, path + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v, path)


def _is_leak_markers(rel: str, path: tuple) -> bool:
    return rel.startswith("protocols/") and len(path) == 2 and path[0] == "leak_rules"


def test_no_config_holds_a_text_pattern_or_word_list():
    found = []
    for f in sorted(CONFIGS_DIR.rglob("*.yaml")):
        rel = f.relative_to(CONFIGS_DIR).as_posix()
        for path, key in _keys(yaml.safe_load(f.read_text(encoding="utf-8"))):
            if key in PATTERN_KEYS and not (key == "markers" and _is_leak_markers(rel, path)):
                found.append(f"configs/{rel}: {'.'.join(map(str, path + (key,)))}")
    assert not found, "a config holds a text pattern or word list:\n" + "\n".join(found)


def test_the_schemas_take_no_text_pattern():
    """A label rule and the report config have no field for one, and unknown keys are errors."""
    from bodhi_lora.analysis.config import ReportConfig
    from bodhi_lora.registry.schema import BenchmarkConfig, LabelRule

    for model in (LabelRule, ReportConfig, BenchmarkConfig):
        assert not set(model.model_fields) & PATTERN_KEYS, model.__name__
        assert model.model_config.get("extra") == "forbid", model.__name__


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


def test_leak_markers_are_the_protocols_own_format_fields(reg):
    pytest.importorskip("bodhi")
    from bodhi_lora.protocols import build

    # Building a protocol reads only its record; any model record will do.
    model = next(iter(reg.models.values()))
    problems = []
    for pid, proto in sorted(reg.protocols.items()):
        if not proto.leak_rules:
            continue
        fields = build(proto, model).format_fields()
        for rid, rule in sorted(proto.leak_rules.items()):
            for marker in rule.markers:
                if not any(re.fullmatch(marker + HEADER_TAIL, f, re.IGNORECASE) for f in fields):
                    problems.append(f"{pid}.{rid}: {marker!r} is none of the protocol's fields {fields}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("marker, ok", [
    ("KEY UNCERTAINT", True),            # a header up to its plural ending
    ("RED FLAGS?\\s*:", True),
    ("DIFFERENTIAL", False),             # a free word (the withdrawn nine-marker rule had it)
    ("SCOPE (OF|BOUNDAR)", False),
    ("RED", False),                      # a word inside a header
    ("FLAGS", False),
])
def test_the_header_check_tells_a_header_from_a_word(marker, ok):
    fields = ("TASK TYPE:", "KEY UNCERTAINTIES:", "RED FLAGS:")
    assert any(re.fullmatch(marker + HEADER_TAIL, f, re.IGNORECASE) for f in fields) is ok


def _literal_patterns(path: Path) -> list[str]:
    """re.<function>("literal", ...) calls in one module."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in RE_FUNCTIONS and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re" and node.args and isinstance(node.args[0], ast.Constant)):
            out.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.args[0].value!r}")
    return out


def test_the_measuring_code_compiles_no_literal_pattern():
    found = [hit for path in MEASURING_CODE for hit in _literal_patterns(path)]
    assert not found, "a literal regular expression in the analysis or label code:\n" + "\n".join(found)
