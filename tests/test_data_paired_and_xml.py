"""The paired-context and XML-record loaders (bodhi_lora/data/loaders.py), on fixtures in
the formats of the MediQ release and the NLM Clinical Questions Collection files, and the
pair draws of the eval ids (bodhi_lora/data/splits.py)."""
import pytest

pytest.importorskip("pydantic")

from bodhi_lora.data import load_rows, split_rows  # noqa: E402
from bodhi_lora.data.loaders import parse_xml_records  # noqa: E402
from bodhi_lora.data.splits import SplitError, eval_ids  # noqa: E402
from bodhi_lora.paths import project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from tests.fixtures.pipeline.registry import fixture_configs  # noqa: E402


@pytest.fixture(scope="module")
def reg_proj(tmp_path_factory):
    configs = fixture_configs(tmp_path_factory.mktemp("reg"))
    return load_registry(configs), project(configs)


def test_paired_rows_differ_only_in_the_context_they_keep(reg_proj):
    reg, proj = reg_proj
    rows = load_rows(reg.benchmarks["tiny-paired"], proj.data_root)
    # 8 cases: one has a single context sentence and one has no answer, so 6 pairs.
    assert len(rows) == 12
    pairs = {}
    for r in rows:
        pairs.setdefault(r.pair_id, {})[r.labels["incomplete"]] = r
    assert len(pairs) == 6 and all(set(p) == {True, False} for p in pairs.values())
    complete, incomplete = pairs[rows[0].pair_id][False], pairs[rows[0].pair_id][True]
    full, short = complete.messages[0]["content"], incomplete.messages[0]["content"]
    assert full.startswith("You are advising a clinician colleague.")
    assert "A 34-year-old woman comes to the clinic with a cough for three weeks." in short
    assert "night sweats" in full and "night sweats" not in short
    assert full.endswith("Which test should be done next?") and short.endswith("Which test should be done next?")
    assert complete.rubrics == incomplete.rubrics
    assert '"Sputum acid-fast bacilli smear and culture"' in complete.rubrics[0]["criterion"]
    assert "variant:incomplete" in incomplete.tags and complete.prompt_id.endswith("-complete")


def test_xml_records_read_one_document_per_record(reg_proj):
    reg, proj = reg_proj
    rows = load_rows(reg.benchmarks["tiny-questions"], proj.data_root)
    # 8 records: one has no question and one repeats an id.
    assert [r.prompt_id for r in rows] == ["NQ000001", "NQ000002", "NQ000003", "NQ000005",
                                           "NQ000006", "NQ000007"]
    assert all(r.rubrics == [] for r in rows)
    first = rows[0]
    assert first.messages[0]["content"].endswith(
        "\n\nWhat is the first-line treatment for hypertension in a 50-year-old man with diabetes?")
    assert "asker:Medicine" in first.tags and "year:2001" in first.tags
    # Inline markup inside a question keeps its text.
    assert rows[-1].messages[0]["content"].endswith("What causes a sudden loss of hearing in one ear?")


def test_parse_xml_records_takes_a_single_document_too():
    text = '<?xml version="1.0"?><root><r><id>1</id></r><r><id>2</id></r></root>'
    assert [r.findtext("id") for r in parse_xml_records(text, "r")] == ["1", "2"]


def test_pair_draws_keep_whole_pairs_and_differ_by_seed(reg_proj):
    reg, proj = reg_proj
    bench = reg.benchmarks["tiny-paired"]
    rows = load_rows(bench, proj.data_root)
    by_id = {r.prompt_id: r for r in rows}
    split = split_rows(bench, "default", rows)
    cell = reg.cells["tiny-paired__tiny-random-llama__bodhi"]
    a = eval_ids(cell, split, by_id, 1, proj.results_root)
    b = eval_ids(cell, split, by_id, 2, proj.results_root)
    assert len(a) == 2 * cell.eval.ids.n and a != b
    for ids in (a, b):
        pairs = {}
        for pid in ids:
            pairs.setdefault(by_id[pid].pair_id, []).append(pid)
        assert all(len(v) == 2 for v in pairs.values())
    too_many = cell.model_copy(update={"eval": cell.eval.model_copy(
        update={"ids": cell.eval.ids.model_copy(update={"n": 7})})})
    with pytest.raises(SplitError, match="cannot draw 7 eval pairs from 6"):
        eval_ids(too_many, split, by_id, 1, proj.results_root)
