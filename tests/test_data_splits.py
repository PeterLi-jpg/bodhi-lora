"""Pools and eval-id draws (bodhi_lora/data/splits.py)."""
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bodhi_lora.data import SplitError, eval_ids, load_rows, split_rows  # noqa: E402
from bodhi_lora.data.loaders import build_hf_qa_rows, load_hf_qa  # noqa: E402
from bodhi_lora.data.splits import draw  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from bodhi_lora.registry.schema import Draw, LabelGroups  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "data"
SPLITS = json.loads((FIXTURES / "hf_qa_splits_expected.json").read_text())


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


@pytest.mark.parametrize("case", ["head", "seeded", "seeded_s1"])
def test_pool_and_holdout_draws_equal_the_old_builder(reg, case, tmp_path):
    """The old builder drew the pool and then the eval holdout; draw() gives the same ids."""
    bench = reg.benchmarks["chatdoctor"]
    rows, _ = build_hf_qa_rows(bench, SPLITS["source"])
    order = [r["prompt_id"] for r in rows]
    exp = SPLITS[case]
    pool = draw(order, Draw(**exp["pool"]), short_ok=True)
    assert pool == exp["pool_ids"]
    assert draw(pool, Draw(**exp["holdout"]), short_ok=False) == exp["holdout_ids"]


def test_holdout_split_keeps_the_head_rule_of_the_2026_07_cells(reg, tmp_path):
    bench = reg.benchmarks["chatdoctor"]
    rows = load_hf_qa(bench, tmp_path, source_rows=lambda src: SPLITS["source"])
    head = bench.model_copy(update={"splits": {**bench.splits, "head": bench.splits["head"].model_copy(
        update={"pool": Draw(select="head", size=30), "holdout": Draw(select="head", size=8)})}})
    split = split_rows(head, "head", rows)
    assert split.eval == SPLITS["head"]["holdout_ids"]
    assert split.train == [p for p in SPLITS["head"]["pool_ids"] if p not in set(split.eval)]
    assert split.probe == []
    with pytest.raises(SplitError):
        draw(split.eval, Draw(select="head", size=9), short_ok=False)


def test_label_hold_out_moves_rows_to_the_probe_pool(reg):
    bench = reg.benchmarks["tiny-rubric"]
    rows = load_rows(bench, REPO_ROOT, fetch=False)
    rule = bench.splits["default"].model_copy(
        update={"hold_out": [LabelGroups(label="needs-info", groups=["positive"])]})
    held = bench.model_copy(update={"splits": {"default": rule}})
    split = split_rows(held, "default", rows)
    by_id = {r.prompt_id: r for r in rows}
    assert split.eval == [f"tiny-{i:03d}" for i in range(1, 7)]
    assert split.probe and all(by_id[p].labels["needs-info"] is True for p in split.probe)
    assert all(by_id[p].labels["needs-info"] is False for p in split.train)
    assert set(split.train) | set(split.probe) == {f"tiny-{i:03d}" for i in range(7, 25)}


def test_a_train_row_with_a_held_out_rows_text_leaves_the_train_pool(reg):
    """MedQuAD repeats a question under several ids. The copy in the train pool goes, so
    the leakage gate, which compares the same texts, cannot fail on it after the traces
    are made."""
    from dataclasses import replace

    from bodhi_lora.data.loaders import row_text

    bench = reg.benchmarks["tiny-rubric"]
    rows = load_rows(bench, REPO_ROOT, fetch=False)
    by_id = {r.prompt_id: r for r in rows}
    held = by_id["tiny-001"]
    # tiny-010 asks tiny-001's question again, in other case and spacing.
    text = held.messages[-1]["content"]
    twin = replace(by_id["tiny-010"], messages=[{"role": "user", "content": "  " + text.upper() + " "}])
    rows = [twin if r.prompt_id == "tiny-010" else r for r in rows]
    split = split_rows(bench, "default", rows)
    assert row_text(twin) == row_text(held) and "tiny-001" in split.eval
    assert split.text_twins == ["tiny-010"] and "tiny-010" not in split.train + split.probe
    assert split_rows(bench, "default", load_rows(bench, REPO_ROOT, fetch=False)).text_twins == []


def test_per_seed_draw_is_deterministic_sorted_and_inside_the_pool(reg):
    cell = reg.cells["tiny-rubric__tiny-random-llama__bodhi"]
    bench = reg.benchmarks[cell.benchmark]
    rows = load_rows(bench, REPO_ROOT, fetch=False)
    by_id = {r.prompt_id: r for r in rows}
    split = split_rows(bench, cell.split, rows)
    a = eval_ids(cell, split, by_id, 1, None)
    assert a == eval_ids(cell, split, by_id, 1, None) == sorted(a)
    assert set(a) <= set(split.eval) and len(a) == cell.eval.ids.n
    draws = {tuple(eval_ids(cell, split, by_id, seed, None)) for seed in range(1, 8)}
    assert len(draws) > 1, "the seed must change the draw"


def test_fixed_file_ids_must_lie_in_the_eval_pool(reg, tmp_path):
    cell = reg.cells["tiny-rubric__tiny-random-llama__bodhi"]
    fixed = cell.model_copy(update={"eval": cell.eval.model_copy(update={"ids": cell.eval.ids.model_copy(
        update={"policy": "fixed_file", "file": "ids.json", "n": 2})})})
    bench = reg.benchmarks[cell.benchmark]
    rows = load_rows(bench, REPO_ROOT, fetch=False)
    split = split_rows(bench, cell.split, rows)
    by_id = {r.prompt_id: r for r in rows}
    (tmp_path / "ids.json").write_text(json.dumps({"prompt_ids": ["tiny-001", "tiny-002"]}))
    assert eval_ids(fixed, split, by_id, 1, tmp_path) == ["tiny-001", "tiny-002"]
    (tmp_path / "ids.json").write_text(json.dumps(["tiny-001", "tiny-020"]))
    with pytest.raises(SplitError, match="outside"):
        eval_ids(fixed, split, by_id, 1, tmp_path)


HB = REPO_ROOT / "data" / "raw"
# sha256 of the sorted prompt ids each seed of the 2026-05 cell evaluated (its
# base_no_wrapper.json); the per-seed draw must name the same prompts.
SEED_SETS_2026_05 = {
    42: "08f6fa3c2d456851777accb2f235b402041d8899c465f1383d41c0bb4d1f2f50",
    7: "7c0642532ca0b4e6cfc9af09e98b34651850886b10f572e9245a0c887ca1e866",
    13: "34933786953704624cc1bc2a7d86e93dc516ba429c71a03847aee8f482c00215",
    99: "453bfc1051b4d4d6b9b3d00ba0a7cff47c4f22e7eb43a28c9205ed86552384e9",
    101: "1032f7c7cb13a97e72782ab16a7287a17244bdd261708c0216cf305eabeed810",
}


@pytest.mark.skipif(not (HB / "healthbench" / "hard.jsonl").is_file(), reason="HealthBench is not downloaded")
def test_healthbench_splits_and_the_2026_05_draws(reg):
    bench = reg.benchmarks["healthbench"]
    rows = load_rows(bench, HB, fetch=False)
    by_id = {r.prompt_id: r for r in rows}
    cell = reg.cells["healthbench__medgemma-27b-text__bodhi__2026-05"]
    split = split_rows(bench, cell.split, rows)
    assert (len(split.train), len(split.eval), len(split.probe)) == (4000, 1000, 0)
    for seed, digest in SEED_SETS_2026_05.items():
        ids = eval_ids(cell, split, by_id, seed, None)
        assert hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest() == digest, seed
    # The proposed default split holds the emergency and clinician-user rows out of training.
    default = split_rows(bench, "default", rows)
    assert (len(default.train), len(default.eval), len(default.probe)) == (3321, 1000, 679)
    # The pilot split: only its fixed 200 eval ids leave the train pool.
    pilot_ids = REPO_ROOT / "results" / "healthbench__medgemma-27b-text__bodhi__2026-04" / "shared" / "inputs" / "eval_ids.json"
    fixed = json.loads(pilot_ids.read_text())["prompt_ids"]
    pilot = split_rows(bench, "pilot", rows, fixed)
    assert len(pilot.eval) == 200 and len(pilot.train) == 4800
