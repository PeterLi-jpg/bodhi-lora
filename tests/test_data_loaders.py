"""Benchmark loaders (bodhi_lora/data/loaders.py): config-driven rows, pins and caches.

The question-answer fixtures were made by the dataset-specific builders that built the
2026-07 cells' files (scripts/build_benchmark_jsonl.py at 02ac9a3, and its MedQA
template before 035d85f). The config-driven loader must give the same rows, byte for
byte, so the eval ids of those cells still name the same prompts.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bodhi_lora.data import DataError, load_rows, source_file  # noqa: E402
from bodhi_lora.data.loaders import build_hf_qa_rows, hf_qa_build_key, load_hf_qa  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "data"
SOURCES = json.loads((FIXTURES / "hf_qa_sources.json").read_text())
EXPECTED = json.loads((FIXTURES / "hf_qa_expected.json").read_text())
SOURCE_KEY = {"chatdoctor": "chatdoctor", "medquad": "medquad", "medqa-open": "medqa",
              "medqa-open-preamble": "medqa", "medicationqa": "medicationqa", "medmcqa-open": "medmcqa"}


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


@pytest.mark.parametrize("bench_id", sorted(SOURCE_KEY))
def test_records_rebuild_the_old_builders_rows(reg, bench_id):
    rows, counts = build_hf_qa_rows(reg.benchmarks[bench_id], SOURCES[SOURCE_KEY[bench_id]])
    assert rows == EXPECTED[bench_id]["rows"]
    assert counts == EXPECTED[bench_id]["counts"]


def test_the_two_medqa_templates_share_prompt_ids(reg):
    a = EXPECTED["medqa-open"]["rows"]
    b = EXPECTED["medqa-open-preamble"]["rows"]
    assert [r["prompt_id"] for r in a] == [r["prompt_id"] for r in b]
    assert "Ask for any information" in b[0]["prompt"][0]["content"]
    assert "Ask for any information" not in a[0]["prompt"][0]["content"]


def test_hf_cache_is_built_once_and_rebuilt_when_the_record_changes(reg, tmp_path):
    bench = reg.benchmarks["chatdoctor"]
    calls = []

    def fake_source(src):
        calls.append(src.revision)
        return SOURCES["chatdoctor"]

    rows = load_hf_qa(bench, tmp_path, source_rows=fake_source)
    assert calls == [bench.source.revision]
    assert [r.prompt_id for r in rows] == [r["prompt_id"] for r in EXPECTED["chatdoctor"]["rows"]]
    assert [r.positions["source"] for r in rows] == list(range(len(rows)))
    meta = json.loads((tmp_path / "chatdoctor" / "rows.manifest.json").read_text())
    assert meta["build_key"] == hf_qa_build_key(bench) and meta["counts"]["n_written"] == len(rows)
    load_hf_qa(bench, tmp_path, source_rows=fake_source)
    assert len(calls) == 1, "a valid cache is read, not rebuilt"
    changed = bench.model_copy(update={"tags": ["theme:other"]})
    load_hf_qa(changed, tmp_path, source_rows=fake_source)
    assert len(calls) == 2, "a record change rebuilds the cache"
    with pytest.raises(DataError, match="fetch"):
        load_hf_qa(changed.model_copy(update={"id_prefix": "x"}), tmp_path, fetch=False)


def test_local_rubric_rows_carry_labels_and_positions(reg):
    rows = load_rows(reg.benchmarks["tiny-rubric"], Path("/nonexistent"), fetch=False)
    assert len(rows) == 24 and rows[0].prompt_id == "tiny-001"
    assert [r.positions["rows"] for r in rows] == list(range(24))
    assert rows[0].labels["needs-info"] is True and rows[1].labels["needs-info"] is False
    assert rows[0].to_json()["prompt"] == rows[0].messages


def test_url_source_is_checked_against_its_sha256(reg, tmp_path):
    bench = reg.benchmarks["healthbench"]
    hard = tmp_path / "healthbench" / "hard.jsonl"
    hard.parent.mkdir(parents=True)
    hard.write_text('{"prompt_id": "x", "prompt": [], "rubrics": [], "example_tags": []}\n')
    with pytest.raises(DataError, match="source changed"):
        source_file(bench, "hard", tmp_path, fetch=False)
    with pytest.raises(DataError, match="fetch"):
        source_file(bench, "full", tmp_path, fetch=False)


HB = CONFIGS_DIR.parent / "data" / "raw" / "healthbench"


@pytest.mark.skipif(not (HB / "hard.jsonl").is_file(), reason="HealthBench is not downloaded")
def test_healthbench_rows_and_physician_labels(reg):
    rows = load_rows(reg.benchmarks["healthbench"], HB.parent, fetch=False)
    by_id = {r.prompt_id: r for r in rows}
    assert len(rows) == 5000
    hard = [r for r in rows if "hard" in r.positions]
    assert len(hard) == 1000 and all("full" in r.positions for r in hard)
    # The label counts quoted in docs/EXPERIMENTS.md (E02, E18, E19).
    def count(label, value, subset):
        return sum(1 for r in subset if r.labels[label] is value)
    assert (count("needs-info-physician", True, hard), count("needs-info-physician", False, hard)) == (204, 113)
    assert count("emergent-physician", False, rows) == 134
    assert count("emergent-physician", True, rows) == 139 + 180
    assert (count("clinician-user", True, rows), count("clinician-user", True, hard)) == (322, 40)
    assert by_id[hard[0].prompt_id].messages[-1]["role"] == "user"
