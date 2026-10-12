"""The eval stage's shared server (bodhi_lora/pipeline/eval.py, run_shared) on the stub
engine: every request reaches the weights of its condition and seed, the files are those
of the one-server-per-condition path, a crash keeps the replies made so far, and with
fixed eval ids each base condition runs once for every seed."""
import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("bodhi")

from bodhi_lora.engine import StubEngine  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.pipeline import eval as eval_stage  # noqa: E402
from bodhi_lora.pipeline.context import StageContext  # noqa: E402
from bodhi_lora.pipeline.runner import prepare  # noqa: E402
from bodhi_lora.registry import load_registry, parse_condition  # noqa: E402
from bodhi_lora.registry.schema import EvalIds  # noqa: E402

SMOKE = "tiny-rubric__tiny-random-llama__bodhi"


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


def _ctx(reg, root, cell=None):
    cell = cell or reg.cells[SMOKE]
    proj = replace(project(CONFIGS_DIR), results_root=root)
    ctx = StageContext(reg=reg, project=proj, cell=cell, paths=proj.cell(cell.id), stub=True)
    prepare(ctx, fetch=False)
    return ctx


@pytest.fixture
def engines(monkeypatch):
    """Every engine the eval stage builds, in order: the policy server, then the grader."""
    made = []
    real = eval_stage.make_engine

    def keep(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(eval_stage, "make_engine", keep)
    return made


@pytest.fixture
def tagged(monkeypatch):
    """The stub policy server ends each reply with the served name the request asked for."""
    plain = StubEngine.complete

    def complete(self, messages, max_new_tokens, temperature=0.0, top_p=1.0, seed=None, logprobs=False,
                 model=None):
        done = plain(self, messages, max_new_tokens, temperature, top_p, seed, logprobs, model=model)
        if not self.lora_modules:
            return done
        return replace(done, text=f"{done.text} [{self.routes[-1]}]")

    monkeypatch.setattr(StubEngine, "complete", complete)


def _results(ctx, seed, condition):
    return json.loads(ctx.paths.eval_file(seed, condition).read_text())


def test_each_request_reaches_the_adapter_of_its_seed_or_the_base(reg, tmp_path, engines, tagged):
    ctx = _ctx(reg, tmp_path)
    cell = ctx.cell
    eval_stage.run_shared(ctx, [1, 2])
    policy, grader = engines
    assert set(policy.lora_modules) == {"adapter-s1", "adapter-s2"}
    assert policy.settings()["enforce_eager"] is True and grader.settings()["enforce_eager"] is False
    assert set(grader.routes) == {grader.model}
    expected_calls = {}
    for seed in cell.seeds:
        for condition in cell.eval.conditions:
            weights, _ = parse_condition(condition)
            name = f"adapter-s{seed}" if weights == "lora" else policy.model
            out = _results(ctx, seed, condition)
            assert len(out["results"]) == cell.eval.ids.n
            for r in out["results"]:
                expected_calls[name] = expected_calls.get(name, 0) + r["calls"]
                # The reply carries the name of the last call that made it, when the
                # protocol passes the reply text through (the identity protocol does).
                if "[" in r["response"]:
                    assert r["response"].endswith(f"[{name}]"), (seed, condition, r["response"])
                elif weights == "base" and condition == "base":
                    pytest.fail("the plain base reply lost its route tag")
            manifest = json.loads((ctx.paths.stage(seed, "eval") / f"{condition}.manifest.json").read_text())
            assert manifest["status"] == "complete" and manifest["shared_server"]["route"] == name
            assert manifest["engines"]["policy"]["lora_modules"] == policy.settings()["lora_modules"]
            assert "time" not in manifest and "base_evaluated_once" not in manifest
    # Every call of every reply went to its condition's weights, and no other call was made.
    assert {n: policy.routes.count(n) for n in set(policy.routes)} == expected_calls
    runs = list(ctx.paths.shared("eval").glob("*.manifest.json"))
    assert len(runs) == 1
    run = json.loads(runs[0].read_text())
    assert run["status"] == "complete" and run["time"]["seconds"] >= 0
    assert run["counts"]["n_files"] == len(cell.seeds) * len(cell.eval.conditions)
    assert run["shared_server"]["client_threads"] == max(cell.eval.serving.concurrency,
                                                         cell.eval.serving.max_num_seqs)


def test_a_rerun_skips_finished_files_and_serves_only_what_is_left(reg, tmp_path, engines):
    ctx = _ctx(reg, tmp_path)
    eval_stage.run_shared(ctx, [1])
    first = ctx.paths.eval_file(1, "lora").read_text()
    ctx.paths.eval_file(2, "lora").parent.mkdir(parents=True, exist_ok=True)
    engines.clear()
    eval_stage.run_shared(ctx, [1, 2])
    assert ctx.paths.eval_file(1, "lora").read_text() == first
    policy = engines[0]
    assert set(policy.lora_modules) == {"adapter-s2"}
    engines.clear()
    assert eval_stage.run_shared(ctx, [1, 2]) == {} and engines == []
    with pytest.raises(SystemExit, match="not seeds"):
        eval_stage.run_shared(ctx, [3])


def test_a_crash_after_the_replies_keeps_them_and_the_rerun_makes_none_again(reg, tmp_path, engines,
                                                                             monkeypatch):
    """One server makes every reply of a run, so the replies go to disk as they come; a
    grader that fails then costs only the grading."""
    ctx = _ctx(reg, tmp_path / "crashed")
    cell = ctx.cell
    real_grade = eval_stage.grade_replies

    def grader_down(*args, **kwargs):
        raise RuntimeError("grader server did not start")

    monkeypatch.setattr(eval_stage, "grade_replies", grader_down)
    with pytest.raises(RuntimeError, match="grader"):
        eval_stage.run_shared(ctx, [1])
    jobs = [eval_stage.make_job(ctx, 1, c) for c in cell.eval.conditions]
    assert all(len(eval_stage.saved_replies(j)) == cell.eval.ids.n for j in jobs)
    assert not any(j.out_path.exists() for j in jobs)
    made = len(engines[0].calls)

    monkeypatch.setattr(eval_stage, "grade_replies", real_grade)
    engines.clear()
    eval_stage.run_shared(ctx, [1])
    policy, _ = engines
    assert policy.calls == [] and made > 0
    assert not any(eval_stage.replies_file(j).exists() for j in jobs)
    manifest = json.loads((ctx.paths.stage(1, "eval") / "lora.manifest.json").read_text())
    assert manifest["shared_server"]["replies_kept_from_an_earlier_run"] == len(jobs) * cell.eval.ids.n
    clean = _ctx(reg, tmp_path / "clean")
    eval_stage.run_shared(clean, [1])
    for c in cell.eval.conditions:
        assert (_normalized(ctx.paths.eval_file(1, c), tmp_path / "crashed")
                == _normalized(clean.paths.eval_file(1, c), tmp_path / "clean")), c


def _fixed_ids_cell(reg):
    """The smoke cell with every seed on the whole eval pool, as the labelled cells are."""
    cell = reg.cells[SMOKE]
    ids = EvalIds(policy="fixed_pool", n=6)
    return cell.model_copy(update={"eval": cell.eval.model_copy(update={"ids": ids})})


def test_with_fixed_ids_a_base_condition_is_evaluated_once_for_every_seed(reg, tmp_path, engines):
    ctx = _ctx(reg, tmp_path, _fixed_ids_cell(reg))
    cell = ctx.cell
    assert ctx.eval_ids(1) == ctx.eval_ids(2)
    eval_stage.run_shared(ctx, [1, 2])
    policy, _ = engines
    base = [c for c in cell.eval.conditions if parse_condition(c)[0] == "base"]
    lora = [c for c in cell.eval.conditions if parse_condition(c)[0] == "lora"]
    # The base server got the calls of seed 1's base files only; each adapter its own.
    base_calls = sum(r["calls"] for c in base for r in _results(ctx, 1, c)["results"])
    assert policy.routes.count(policy.model) == base_calls
    for seed in cell.seeds:
        assert policy.routes.count(f"adapter-s{seed}") == sum(
            r["calls"] for c in lora for r in _results(ctx, seed, c)["results"])
    for c in base:
        one, two = _results(ctx, 1, c), _results(ctx, 2, c)
        assert one["results"] == two["results"]
        assert (one["run_metadata"]["seed"], two["run_metadata"]["seed"]) == (1, 2)
        for seed in cell.seeds:
            m = json.loads((ctx.paths.stage(seed, "eval") / f"{c}.manifest.json").read_text())
            assert m["base_evaluated_once"] == {"evaluated_with_seed": 1, "written_for_seeds": [1, 2]}
            assert m["eval_ids"]["n"] == 6
    for c in lora:
        m = json.loads((ctx.paths.stage(2, "eval") / f"{c}.manifest.json").read_text())
        assert "base_evaluated_once" not in m
    run = json.loads(next(ctx.paths.shared("eval").glob("*.manifest.json")).read_text())
    assert {x["condition"] for x in run["base_evaluated_once"]} == set(base)
    assert run["counts"]["n_files"] == 2 * len(cell.eval.conditions)
    assert run["counts"]["n_conditions_run"] == len(base) + 2 * len(lora)


def test_a_per_seed_draw_cell_evaluates_base_for_every_seed(reg, tmp_path, engines):
    ctx = _ctx(reg, tmp_path)
    eval_stage.run_shared(ctx, [1, 2])
    policy, _ = engines
    base = [c for c in ctx.cell.eval.conditions if parse_condition(c)[0] == "base"]
    assert policy.routes.count(policy.model) == sum(
        r["calls"] for s in (1, 2) for c in base for r in _results(ctx, s, c)["results"])
    assert not eval_stage.seed_invariant_ids(ctx, [1, 2])


def _normalized(path, root):
    """The file without what is a clock reading: the time stamp, and bodhi-llm's own
    pass timings in each reply's protocol metadata."""
    data = json.loads(path.read_text().replace(str(root), "<root>"))
    data["run_metadata"].pop("timestamp_utc")
    for r in data["results"]:
        for key in ("pass1_time", "total_time"):
            r["protocol_metadata"].pop(key, None)
    return data


def test_the_shared_server_writes_the_files_of_the_per_condition_path(reg, tmp_path):
    """Same keys and the same rows, condition by condition; only clock readings differ.
    The manifests have the same fields, except that the per-condition one has its own time
    and the shared one points to the run's manifest instead."""
    per_condition, shared = tmp_path / "per_condition", tmp_path / "shared"
    cell = reg.cells[SMOKE]
    a = _ctx(reg, per_condition)
    for seed in cell.seeds:
        for condition in cell.eval.conditions:
            eval_stage.run(a, seed, condition)
    b = _ctx(reg, shared)
    eval_stage.run_shared(b, list(cell.seeds))
    for seed in cell.seeds:
        for condition in cell.eval.conditions:
            fa, fb = a.paths.eval_file(seed, condition), b.paths.eval_file(seed, condition)
            assert _normalized(fa, per_condition) == _normalized(fb, shared), (seed, condition)
            ma = json.loads((fa.parent / f"{condition}.manifest.json").read_text())
            mb = json.loads((fb.parent / f"{condition}.manifest.json").read_text())
            assert set(ma) - {"time"} == set(mb) - {"shared_server"}
            assert (ma["config"], ma["counts"], ma["status"]) == (mb["config"], mb["counts"], mb["status"])
            assert set(ma["engines"]) == set(mb["engines"]) == {"policy", "grader"}
