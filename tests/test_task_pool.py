"""scripts/task_pool.py on the CPU (docs/NRP.md): workers on one state folder, with short
shell commands as tasks. The last tests check configs/queues/eos_rerun_nrp.yaml."""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "scripts" / "task_pool.py"
EOS_NRP = ROOT / "configs" / "queues" / "eos_rerun_nrp.yaml"
FAST = {"TASK_POOL_POLL_S": "0.2", "TASK_POOL_HEARTBEAT_S": "0.2", "TASK_POOL_STALE_S": "3",
        "TASK_POOL_STOP_GRACE_S": "5", "TASK_POOL_CARD_WAIT_S": "0"}


class Pool:
    def __init__(self, tmp: Path):
        self.tmp, self.state, self.repo = tmp, tmp / "state", tmp / "repo"
        self.repo.mkdir()
        self.trace = tmp / "trace.txt"
        self.env = {**os.environ, **FAST, "PATH": os.environ["PATH"]}

    def dag(self, tasks) -> Path:
        path = self.tmp / "dag.yaml"
        path.write_text(yaml.safe_dump({"tasks": tasks}, sort_keys=False))
        return path

    def step(self, name, seconds=0.0, rc=0) -> str:
        """A task command that records its start and end in the trace file."""
        return (f"echo start {name} $(date +%s.%N) >> {self.trace}; sleep {seconds}; "
                f"echo end {name} $(date +%s.%N) >> {self.trace}; exit {rc}")

    def args(self, dag, worker):
        return [sys.executable, str(POOL), "--dag", str(dag), "--state", str(self.state),
                "--repo", str(self.repo), "--worker", worker]

    def start(self, dag, worker, env=None):
        return subprocess.Popen(self.args(dag, worker), env={**self.env, **(env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def run(self, dag, worker="w1", env=None, timeout=60):
        return subprocess.run(self.args(dag, worker), env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=timeout)

    def events(self):
        out = []
        for line in self.trace.read_text().splitlines():
            kind, name, t = line.split()
            out.append((kind, name, float(t)))
        return out

    def times(self, name):
        ev = {(k, n): t for k, n, t in self.events()}
        return ev[("start", name)], ev[("end", name)]


@pytest.fixture
def pool(tmp_path):
    return Pool(tmp_path)


def test_two_workers_run_each_task_once_after_its_dependencies(pool):
    dag = pool.dag([
        {"id": "a", "hours": 1, "run": pool.step("a", 0.6)},
        {"id": "b", "hours": 1, "run": pool.step("b", 0.6)},
        {"id": "c", "hours": 1, "after": ["a", "b"], "run": pool.step("c", 0.2)},
        {"id": "d", "hours": 1, "after": ["a"], "run": pool.step("d", 0.2)},
    ])
    workers = [pool.start(dag, f"w{i}") for i in (1, 2)]
    for w in workers:
        assert w.wait(timeout=60) == 0, w.stdout.read()
    starts = [n for k, n, _ in pool.events() if k == "start"]
    assert sorted(starts) == ["a", "b", "c", "d"]            # each exactly once
    assert pool.times("c")[0] >= max(pool.times("a")[1], pool.times("b")[1])
    assert pool.times("d")[0] >= pool.times("a")[1]
    # a and b ran at the same time, one on each worker
    assert pool.times("b")[0] < pool.times("a")[1] and pool.times("a")[0] < pool.times("b")[1]
    assert {p.stem for p in (pool.state / "done").glob("*.json")} == {"a", "b", "c", "d"}
    assert [p.name for p in (pool.state / "claims").iterdir()] == [".stale"]


def test_the_longest_chain_goes_first(pool):
    dag = pool.dag([
        {"id": "short", "hours": 5, "run": pool.step("short")},
        {"id": "head", "hours": 1, "run": pool.step("head")},
        {"id": "tail", "hours": 9, "after": ["head"], "run": pool.step("tail")},
    ])
    assert pool.run(dag).returncode == 0
    assert [n for k, n, _ in pool.events() if k == "start"] == ["head", "tail", "short"]


def test_a_failed_task_runs_again_then_blocks_what_comes_after_it(pool):
    flag = pool.tmp / "flaky"
    dag = pool.dag([
        # fails the first time only
        {"id": "flaky", "run": f"if [ -e {flag} ]; then exit 0; fi; touch {flag}; exit 3"},
        {"id": "bad", "run": pool.step("bad", rc=4)},
        {"id": "after_bad", "after": ["bad"], "run": pool.step("after_bad")},
        {"id": "free", "run": pool.step("free")},
    ])
    proc = pool.run(dag)
    # exit 0, so the cluster does not replace the pod with one that finds the same
    assert proc.returncode == 0, proc.stdout
    done = {p.stem for p in (pool.state / "done").glob("*.json")}
    assert done == {"flaky", "free"}
    summary = json.loads((pool.state / "summary.json").read_text())
    assert summary["failed"] == ["bad"] and summary["blocked"] == ["after_bad"]
    assert pool.run(dag, worker="w2").returncode == 0               # a later pod stops at once
    assert [n for k, n, _ in pool.events() if k == "start"].count("bad") == 2   # MAX_ATTEMPTS
    assert "after_bad" not in [n for _, n, _ in pool.events()]
    assert json.loads((pool.state / "failed" / "bad.json").read_text())["exit"] == 4
    assert "FAILED or blocked" in proc.stdout


def test_a_stale_claim_is_taken_again(pool):
    dag = pool.dag([{"id": "a", "run": pool.step("a")}])
    claim = pool.state / "claims" / "a"
    claim.mkdir(parents=True)
    (claim / "heartbeat").touch()
    old = time.time() - 60
    os.utime(claim / "heartbeat", (old, old))
    assert pool.run(dag).returncode == 0
    assert (pool.state / "done" / "a.json").is_file()
    assert any(p.name.startswith("a.") for p in (pool.state / "claims" / ".stale").iterdir())


def test_a_live_claim_of_another_worker_is_left_alone(pool):
    """With the only task claimed by a live worker, this worker exits at once and leaves
    the card free: the other worker covers what is left."""
    dag = pool.dag([{"id": "a", "run": pool.step("a")}])
    claim = pool.state / "claims" / "a"
    claim.mkdir(parents=True)
    (claim / "heartbeat").touch()
    proc = pool.run(dag)
    assert proc.returncode == 0 and "exiting" in proc.stdout
    assert not pool.trace.exists()


def test_a_worker_waits_while_more_tasks_are_left_than_are_running(pool):
    dag = pool.dag([
        {"id": "long", "hours": 1, "run": pool.step("long", 2.0)},
        {"id": "x", "after": ["long"], "run": pool.step("x", 0.8)},
        {"id": "y", "after": ["long"], "run": pool.step("y", 0.8)},
    ])
    first = pool.start(dag, "w1")
    time.sleep(0.8)
    second = pool.start(dag, "w2")
    assert first.wait(timeout=60) == 0 and second.wait(timeout=60) == 0
    out = second.stdout.read()
    assert "waiting for ['long']" in out
    # x and y ran at the same time, one on each worker
    assert pool.times("x")[0] < pool.times("y")[1] and pool.times("y")[0] < pool.times("x")[1]


def test_sigterm_stops_the_task_with_sigint_and_keeps_its_claim(pool):
    seen = pool.tmp / "seen"
    dag = pool.dag([{"id": "a", "run": f"trap 'echo int > {seen}; exit 130' INT; sleep 30 & wait"}])
    proc = pool.start(dag, "w1")
    claim = pool.state / "claims" / "a"
    end = time.time() + 20
    while not (claim / "heartbeat").exists() and time.time() < end:
        time.sleep(0.05)
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    assert seen.read_text().strip() == "int"
    assert claim.is_dir() and not (pool.state / "done" / "a.json").exists()
    assert not (pool.state / "failed" / "a.json").exists()


def test_a_card_that_holds_memory_stops_the_worker_before_the_task(pool, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/usr/bin/env bash\necho 41000\n")
    (fake / "nvidia-smi").chmod(0o755)
    dag = pool.dag([{"id": "a", "run": pool.step("a")}])
    proc = pool.run(dag, env={"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"})
    assert proc.returncode == 75
    assert not pool.trace.exists()
    assert not (pool.state / "claims" / "a").exists()          # the task is free for another pod


def test_has_work_tells_a_new_pod_whether_to_start(pool):
    dag = pool.dag([{"id": "a", "run": pool.step("a")}, {"id": "b", "after": ["a"], "run": pool.step("b")}])
    has = [sys.executable, str(POOL), "--dag", str(dag), "--state", str(pool.state), "--repo", str(pool.repo),
           "--has-work"]
    assert subprocess.run(has, env=pool.env).returncode == 0          # a is ready
    claim = pool.state / "claims" / "a"
    claim.mkdir(parents=True)
    (claim / "heartbeat").touch()
    assert subprocess.run(has, env=pool.env).returncode == 3          # b waits on a, whose worker takes it
    import shutil
    shutil.rmtree(claim)
    assert pool.run(dag).returncode == 0
    assert subprocess.run(has, env=pool.env).returncode == 3          # all done


@pytest.mark.parametrize("tasks, message", [
    ([{"id": "a", "run": "true"}, {"id": "a", "run": "true"}], "twice"),
    ([{"id": "a", "after": ["zz"], "run": "true"}], "unknown"),
    ([{"id": "a", "after": ["b"], "run": "true"}, {"id": "b", "after": ["a"], "run": "true"}], "cycle"),
    ([{"id": "a/b", "run": "true"}], "needs an id"),
])
def test_a_bad_task_file_is_refused(pool, tasks, message):
    proc = pool.run(pool.dag(tasks))
    assert proc.returncode != 0 and message in proc.stderr + proc.stdout


def test_the_eos_rerun_tasks_name_real_cells_on_card_0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import task_pool
    tasks = task_pool.load_dag(EOS_NRP)
    cells = set()
    for t in tasks:
        argv = shlex.split(t["run"])
        assert argv[:4] == ["python", "-m", "bodhi_lora", "run"], t
        assert (ROOT / "configs" / "cells" / f"{argv[4]}.yaml").is_file(), t
        assert argv[argv.index("--gpus") + 1] == "0", t
        cells.add(argv[4])
    # the cells of the reruns (RUN_PLAN.md section 4.4)
    plan = {shlex.split(line)[4] for line in (ROOT / "configs" / "queues" / "eos_rerun.txt").read_text().splitlines()
            if line.startswith("python ")}
    assert cells == plan
    # every cell that reads another's adapters comes after all of that cell's training
    for t in tasks:
        cell = shlex.split(t["run"])[4]
        source = yaml.safe_load((ROOT / "configs" / "cells" / f"{cell}.yaml").read_text()).get("adapters_from")
        if source and "--stages" in t["run"]:
            trains = {u["id"] for u in tasks if shlex.split(u["run"])[4] == source and "--stages train" in u["run"]}
            assert trains and trains <= set(t["after"]), t
