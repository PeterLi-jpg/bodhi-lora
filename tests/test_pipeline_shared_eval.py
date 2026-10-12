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
    # Seed 1 has nothing left, but the server of seeds 1 and 2 holds both adapters.
    assert set(policy.lora_modules) == {"adapter-s1", "adapter-s2"}
    assert {r for r in policy.routes if r != policy.model} == {"adapter-s2"}
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


def _manifest(ctx, seed, condition):
    return json.loads((ctx.paths.stage(seed, "eval") / f"{condition}.manifest.json").read_text())


def _policy_dies_after(monkeypatch, n):
    """The stub policy server answers n requests, then refuses every connection, as a
    vLLM server does once its engine died; the grader stub keeps answering."""
    import urllib.error

    plain, made = StubEngine.complete, [0]

    def complete(self, *args, **kwargs):
        if self.lora_modules:
            made[0] += 1
            if made[0] > n:
                raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        return plain(self, *args, **kwargs)

    monkeypatch.setattr(StubEngine, "complete", complete)


def test_failed_replies_leave_their_conditions_incomplete_and_stop_the_stage(reg, tmp_path, monkeypatch):
    """A server that dies mid-run must not make its conditions look finished: they get no
    result file and an incomplete manifest, the runner does not count them complete, and
    the stage exits with an error, so a queue stops instead of grading on."""
    from bodhi_lora.pipeline.runner import eval_step, is_complete

    ctx = _ctx(reg, tmp_path)
    cell = ctx.cell
    _policy_dies_after(monkeypatch, 3)
    with pytest.raises(SystemExit, match="incomplete"):
        eval_stage.run_shared(ctx, list(cell.seeds))
    pairs = [(s, c) for s in cell.seeds for c in cell.eval.conditions]
    broken = [(s, c) for s, c in pairs if _manifest(ctx, s, c)["counts"]["n_failed_inference"]]
    assert broken
    for s, c in pairs:
        m = _manifest(ctx, s, c)
        if (s, c) in broken:
            assert m["status"] == "incomplete" and "outputs" not in m
            assert not ctx.paths.eval_file(s, c).exists()
        else:
            assert m["status"] == "complete" and ctx.paths.eval_file(s, c).is_file()
    assert not is_complete(eval_step(ctx, list(cell.seeds)).done)
    run = json.loads(next(ctx.paths.shared("eval").glob("*.manifest.json")).read_text())
    assert run["status"] == "incomplete" and run["counts"]["n_incomplete"] == len(broken)
    kept = sum(len(eval_stage.saved_replies(eval_stage.make_job(ctx, s, c))) for s, c in broken)

    # A rerun with a working server makes only the replies that are missing.
    monkeypatch.undo()
    eval_stage.run_shared(ctx, list(cell.seeds))
    assert all(_manifest(ctx, s, c)["status"] == "complete" for s, c in pairs)
    assert all(ctx.paths.eval_file(s, c).is_file() for s, c in pairs)
    assert is_complete(eval_step(ctx, list(cell.seeds)).done)
    rerun = max(ctx.paths.shared("eval").glob("*.manifest.json"), key=lambda p: p.stat().st_mtime)
    assert json.loads(rerun.read_text())["shared_server"]["replies_kept_from_an_earlier_run"] == kept
    assert not any(eval_stage.replies_file(eval_stage.make_job(ctx, s, c)).exists() for s, c in pairs)


def test_failed_grades_keep_the_replies_and_the_condition_incomplete(reg, tmp_path, monkeypatch):
    ctx = _ctx(reg, tmp_path)
    plain = StubEngine.complete

    def grader_refuses(self, *args, **kwargs):
        if not self.lora_modules:
            raise TimeoutError("timed out")
        return plain(self, *args, **kwargs)

    # The policy stub of this cell holds adapters, so it answers; the grader holds none.
    monkeypatch.setattr(StubEngine, "complete", grader_refuses)
    with pytest.raises(SystemExit, match="incomplete"):
        eval_stage.run_shared(ctx, [1])
    for c in ctx.cell.eval.conditions:
        m = _manifest(ctx, 1, c)
        assert m["status"] == "incomplete" and m["counts"]["n_failed_grading"] > 0
        assert m["counts"]["n_failed_inference"] == 0
        job = eval_stage.make_job(ctx, 1, c)
        assert not job.out_path.exists() and len(eval_stage.saved_replies(job)) == ctx.cell.eval.ids.n


def test_a_reply_too_long_for_the_context_is_listed_and_the_condition_completes(reg, tmp_path, monkeypatch):
    from bodhi_lora.engine import ContextLengthError

    ctx = _ctx(reg, tmp_path)
    long_id = ctx.eval_ids(1)[0]
    long_text = ctx.rows_by_id()[long_id].messages[-1]["content"]
    plain = StubEngine.complete

    def complete(self, messages, *args, **kwargs):
        if self.lora_modules and messages[-1]["content"] == long_text:
            raise ContextLengthError("This model's maximum context length is 4096 tokens.")
        return plain(self, messages, *args, **kwargs)

    monkeypatch.setattr(StubEngine, "complete", complete)
    eval_stage.run_shared(ctx, [1])
    m = _manifest(ctx, 1, "base")
    assert m["status"] == "complete" and m["too_long_ids"] == [long_id] and m["counts"]["n_too_long"] == 1
    assert long_id not in {r["prompt_id"] for r in _results(ctx, 1, "base")["results"]}


def test_the_per_condition_path_stops_on_a_failed_reply_too(reg, tmp_path, monkeypatch):
    ctx = _ctx(reg, tmp_path)
    _policy_dies_after(monkeypatch, 1)
    with pytest.raises(SystemExit, match="incomplete"):
        eval_stage.run(ctx, 1, "lora")
    assert _manifest(ctx, 1, "lora")["status"] == "incomplete"
    assert not ctx.paths.eval_file(1, "lora").exists()


def test_a_rerun_of_base_files_only_uses_the_server_of_the_first_run(reg, tmp_path, engines):
    """With only base conditions left (a base file deleted, or a crash while the files
    were written), the server still holds every adapter of the run and stays eager, so a
    4-bit base never runs with CUDA graphs here and every base file of the cell comes
    from one serving setup."""
    ctx = _ctx(reg, tmp_path)
    eval_stage.run_shared(ctx, [1, 2])
    first = engines[0].settings()
    base = [c for c in ctx.cell.eval.conditions if parse_condition(c)[0] == "base"]
    for c in base:
        ctx.paths.eval_file(2, c).unlink()
    engines.clear()
    eval_stage.run_shared(ctx, [1, 2])
    policy = engines[0]
    assert policy.settings() == first
    assert policy.settings()["enforce_eager"] is True and set(policy.lora_modules) == {"adapter-s1", "adapter-s2"}
    assert set(policy.routes) == {policy.model}
    for c in base:
        assert _manifest(ctx, 2, c)["engines"]["policy"] == first
