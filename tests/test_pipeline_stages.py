"""Stage logic that a stub run does not reach: the tau rules, the resume guard of the
traces, and the eval stage's refusal to rewrite a finished condition."""
import json
import random
from dataclasses import replace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("transformers")

from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.pipeline import filter as flt  # noqa: E402
from bodhi_lora.pipeline import traces  # noqa: E402
from bodhi_lora.pipeline.context import StageContext  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from bodhi_lora.registry.schema import Tau  # noqa: E402


def graded(scores):
    return [{"prompt_id": f"p{i:03d}", "grade": {"overall_score": s}} for i, s in enumerate(scores)]


SCORES = [round(0.01 * i, 2) for i in range(200)]   # 0.00 .. 1.99, two per hundredth band


def test_percentile_tau_keeps_the_rank_rule_of_the_2026_07_launcher():
    # Sorted index int(p / 100 * n) - 1, minus 1e-9: with p = 22 and n = 200, index 43.
    assert flt.percentile_tau(SCORES, 22) == pytest.approx(0.43 - 1e-9, abs=1e-12)
    with pytest.raises(ValueError):
        flt.percentile_tau([], 22)


def test_fixed_rule_keeps_scores_at_or_above_value():
    kept, rec = flt.apply_tau(graded(SCORES), Tau(rule="fixed", value=0.4), 0.1, 200)
    assert len(kept) == 160 and rec["tau"] == 0.4 and rec["rule"] == "fixed"


def test_fixed_then_percentile_switches_only_when_the_train_share_is_low():
    rows = graded(SCORES)
    rule = Tau(rule="fixed_then_percentile", value=0.4, percentile=22, min_kept_share=0.4)
    kept, rec = flt.apply_tau(rows, rule, 0.1, 200)
    assert not rec["triggered"] and rec["tau"] == 0.4 and len(kept) == 160
    high = Tau(rule="fixed_then_percentile", value=1.5, percentile=22, min_kept_share=0.4)
    kept, rec = flt.apply_tau(rows, high, 0.1, 200)
    assert rec["triggered"] and rec["tau"] == pytest.approx(0.43 - 1e-9) and len(kept) == 157


def test_percentile_needs_enough_grades():
    with pytest.raises(SystemExit, match="needs"):
        flt.apply_tau(graded(SCORES[:50]), Tau(rule="percentile", percentile=22), 0.1, 50)


def test_none_and_random_same_count_rules():
    rows = graded(SCORES)
    kept, rec = flt.apply_tau(rows, Tau(rule="none"), 0.1, 200)
    assert len(kept) == 200 and rec["tau"] is None
    a, _ = flt.apply_tau(rows, Tau(rule="random_same_count", value=0.4, seed=3), 0.1, 200)
    b, _ = flt.apply_tau(list(reversed(rows)), Tau(rule="random_same_count", value=0.4, seed=3), 0.1, 200)
    assert len(a) == 160 and [g["prompt_id"] for g in a] == [g["prompt_id"] for g in b]
    assert any(g["grade"]["overall_score"] < 0.4 for g in a)


def test_split_ignores_the_order_grades_finished_in():
    rows = graded(SCORES[:20])
    shuffled = rows[:]
    random.Random(5).shuffle(shuffled)
    assert flt.split_kept(rows, 0.1, 42) == flt.split_kept(shuffled, 0.1, 42)
    train, val = flt.split_kept(rows, 0.1, 42)
    assert (len(train), len(val)) == (18, 2)
    assert flt.split_kept(graded([0.5]), 0.1, 1) == (graded([0.5]), [])


def test_traces_resume_refuses_another_model_or_protocol(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"prompt_id": "a", "model": "org/m", "protocol": "bodhi",
                                "ablate_component": "none"}) + "\n" + '{"prompt_id": "b", "mod')
    done, bad = traces.read_existing(path)
    assert set(done) == {"a"} and bad == 1
    traces.check_resume(done, "org/m", "bodhi", "none")
    with pytest.raises(SystemExit, match="mixing"):
        traces.check_resume(done, "org/other", "bodhi", "none")
    with pytest.raises(SystemExit, match="mixing"):
        traces.check_resume(done, "org/m", "ask-first", "none")


def test_eval_never_rewrites_a_finished_condition(tmp_path):
    from bodhi_lora.pipeline import eval as eval_stage

    reg = load_registry(CONFIGS_DIR)
    cell_id = "tiny-rubric__tiny-random-llama__bodhi"
    proj = replace(project(CONFIGS_DIR), results_root=tmp_path)
    ctx = StageContext(reg=reg, project=proj, cell=reg.cells[cell_id], paths=proj.cell(cell_id), stub=True)
    out = ctx.paths.eval_file(1, "base")
    out.parent.mkdir(parents=True)
    out.write_text('{"already": "graded"}')
    assert eval_stage.run(ctx, 1, "base") == {}
    assert out.read_text() == '{"already": "graded"}'
    with pytest.raises(SystemExit, match="not a condition"):
        eval_stage.run(ctx, 1, "lora+clarify")
