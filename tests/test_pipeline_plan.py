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


def _printed_plan(cell_id, stages):
    cell = REG.cells[cell_id]
    ctx = StageContext(reg=REG, project=PROJ, cell=cell, paths=PROJ.cell(cell_id))
    shared, per_seed = plan(ctx, stages, cell.seeds, cell.grading.judges)
    buf = io.StringIO()
    with redirect_stdout(buf):
        Runner(ctx, "dry_run", [0, 1, 2, 3, 4]).print_plan(shared, per_seed)
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
    assert count("eval") == len(cell.seeds) * len(cell.eval.conditions)
    assert count("grade") == len(cell.seeds) * (1 + len(cell.grading.judges))
    assert count("bench") == (len(cell.bench.seeds) if cell.bench else 0)
    assert count("probe") == (1 if cell.probe else 0)
    assert count("traces") == (1 if trains else 0)


@pytest.mark.parametrize("cell_id", RECORDED)
def test_recorded_cells_plan_only_additive_stages(cell_id):
    cell = REG.cells[cell_id]
    _, per_seed, _ = _printed_plan(cell_id, ADDITIVE_STAGES)
    judge_steps = [s for steps in per_seed.values() for s in steps if "--judge" in s.extra]
    assert len(judge_steps) == len(cell.seeds) * len(cell.grading.judges)


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
