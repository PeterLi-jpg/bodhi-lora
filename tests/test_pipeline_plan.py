"""Every cell record in configs/ is one runnable command: its plan of GPU steps builds on
the CPU with no data and no model, every protocol it names builds, and its judges keep
the panel rule of EXPERIMENTS.md E10."""
import io
from contextlib import redirect_stdout

import pytest

pytest.importorskip("bodhi")

from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.pipeline.context import StageContext  # noqa: E402
from bodhi_lora.pipeline.runner import ADDITIVE_STAGES, STAGES, Runner, plan  # noqa: E402
from bodhi_lora.protocols import build  # noqa: E402
from bodhi_lora.registry import RECORD_STATUSES, load_registry, parse_condition  # noqa: E402

REG = load_registry(CONFIGS_DIR)
PROJ = project(CONFIGS_DIR)
PLANNED = sorted(cid for cid, c in REG.cells.items() if c.report.status == "planned")
RECORDED = sorted(cid for cid, c in REG.cells.items() if c.report.status in RECORD_STATUSES)


def _printed_plan(cell_id, stages, gpus=(0, 1, 2, 3, 4), shared_eval=True):
    cell = REG.cells[cell_id]
    ctx = StageContext(reg=REG, project=PROJ, cell=cell, paths=PROJ.cell(cell_id))
    shared, per_seed = plan(ctx, stages, cell.seeds, cell.grading.judges, shared_eval)
    buf = io.StringIO()
    with redirect_stdout(buf):
        Runner(ctx, "dry_run", list(gpus)).print_plan(shared, per_seed)
    return shared, per_seed, buf.getvalue()


@pytest.mark.parametrize("cell_id", PLANNED)
def test_every_planned_cell_prints_one_gpu_command_per_step(cell_id):
    cell = REG.cells[cell_id]
    shared, per_seed, out = _printed_plan(cell_id, STAGES)
    gpu_lines = [line for line in out.splitlines() if line.startswith("DRY-RUN [gpu")]
    steps = shared + [s for steps in per_seed.values() for s in steps]
    assert len(gpu_lines) == sum(s.env != "cpu" for s in steps)
    assert all(f"--cell {cell_id}" in line for line in gpu_lines)
    count = lambda stage: sum(f"bodhi_lora.pipeline.{stage} " in line for line in gpu_lines)  # noqa: E731
    trains = cell.adapters_from is None and cell.train is not None
    assert count("train") == (len(cell.seeds) if trains else 0)
    # One shared eval server per seed, each on its own GPU (eval.shared_server).
    assert count("eval") == len(cell.seeds) * (1 if cell.eval.shared_server else len(cell.eval.conditions))
    assert count("grade") == len(cell.seeds) * (1 + len(cell.grading.judges))
    assert count("classify") == len(cell.seeds) * len(cell.grading.judges)
    assert count("bench") == (len(cell.bench.seeds) if cell.bench else 0)
    assert count("probe") == (1 if cell.probe else 0)
    assert count("traces") == (1 if trains else 0)


@pytest.mark.parametrize("cell_id", PLANNED)
def test_on_one_gpu_one_eval_step_serves_every_seed_after_every_training(cell_id):
    """R3 to R7 run on one GPU: the eval of all seeds is one server holding every seed's
    adapter, after the last training and before the first grade."""
    cell = REG.cells[cell_id]
    _, _, out = _printed_plan(cell_id, STAGES, gpus=(3,))
    lines = [line for line in out.splitlines() if line.startswith("DRY-RUN [gpu 3]")]
    evals = [i for i, line in enumerate(lines) if "bodhi_lora.pipeline.eval " in line]
    assert len(evals) == 1
    assert lines[evals[0]].endswith("--seeds " + ",".join(str(s) for s in cell.seeds))
    trains = [i for i, line in enumerate(lines) if "bodhi_lora.pipeline.train " in line]
    grades = [i for i, line in enumerate(lines) if "bodhi_lora.pipeline.grade " in line]
    assert all(i < evals[0] for i in trains) and all(i > evals[0] for i in grades)
    # The fallback: one step per condition and seed.
    _, _, out = _printed_plan(cell_id, STAGES, gpus=(3,), shared_eval=False)
    per_condition = [line for line in out.splitlines() if "bodhi_lora.pipeline.eval " in line]
    assert len(per_condition) == len(cell.seeds) * len(cell.eval.conditions)
    assert all("--condition" in line for line in per_condition)


@pytest.mark.parametrize("cell_id", RECORDED)
def test_recorded_cells_plan_only_additive_stages(cell_id):
    cell = REG.cells[cell_id]
    _, per_seed, _ = _printed_plan(cell_id, ADDITIVE_STAGES)
    steps = [s for steps in per_seed.values() for s in steps]
    assert {s.stage for s in steps} <= set(ADDITIVE_STAGES)
    for stage in ("grade", "classify"):
        judge_steps = [s for s in steps if s.stage == stage and "--judge" in s.extra]
        assert len(judge_steps) == len(cell.seeds) * len(cell.grading.judges), stage


@pytest.mark.parametrize("cell_id", PLANNED)
def test_every_protocol_of_a_planned_cell_builds_for_its_model(cell_id):
    cell = REG.cells[cell_id]
    model = REG.models[cell.model]
    names = {cell.protocol} | {p for c in cell.eval.conditions for p in [parse_condition(c)[1]] if p}
    for name in names:
        build(REG.protocols[name], model)


def test_judges_sit_outside_the_filter_evaluator_and_base_families():
    """E10: a judge is not Qwen or Llama (the graders' families) and not the base's family."""
    fam = {mid: m.family for mid, m in REG.models.items()}
    for cid, cell in REG.cells.items():
        barred = {fam[cell.grading.filter], fam[cell.grading.evaluator], fam[cell.model],
                  REG.teacher(cell).family}
        for judge in cell.grading.judges:
            assert fam[judge] not in barred, (cid, judge)
