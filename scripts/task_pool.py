#!/usr/bin/env python3
"""Run a graph of shell tasks with workers that share one state folder (docs/NRP.md).

    python scripts/task_pool.py --dag configs/queues/eos_rerun_nrp.yaml \\
        --state /data/pool/eos_rerun --repo /data/bohdi [--worker NAME]

Several workers, each a pod with one GPU, run this on the same state folder on a shared
volume. A worker takes the ready task (every task it comes after is done) with the most
estimated hours on its longest chain of later tasks, runs it in the repo, records the
result, and takes the next. It exits when no task is ready and the tasks the other
workers are running cover what is left, so no worker holds a card idle while it waits.
The cluster's scheduler queues the worker pods until a card is free.

The state folder:
  claims/<task>/        a worker's claim (mkdir is atomic on CephFS): claim.json and a
                        heartbeat file the worker touches every HEARTBEAT_S
  done/<task>.json      the attempt that exited 0
  failed/<task>.json    the last attempt that did not; attempts/<task> counts them
  logs/<task>.log       the task's output, every attempt appended
  workers/<worker>.log  this script's lines

A claim whose heartbeat is older than STALE_S (its pod died) moves to claims/.stale/ and
the task is ready again. A task that fails runs again until it has failed MAX_ATTEMPTS
times; then every task after it is blocked, and the workers stop once the rest is done.
On SIGTERM or SIGINT (the pod is stopped) the running task's process group gets SIGINT,
so a stage stops its vLLM server; its claim stays and goes stale, and another worker runs
the task again. The pipeline runner skips every stage whose manifest is complete.

When no task is left that this worker could run, it writes summary.json (done, failed,
blocked) and exits 0, failed tasks or not: a pod that failed would only be replaced by
one that finds the same. The summary and the worker logs say what failed. --has-work
exits 0 if a new worker would find a task to run and 3 if not, so a pod can check
before it downloads its models.

Exit codes: 0 no task left for this worker (summary.json says whether all are done);
2 bad arguments or task file; 3 (--has-work) nothing for a new worker; 75 the card still
holds memory before a task (the pod should be replaced); 128+N a signal.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

POLL_S = float(os.environ.get("TASK_POOL_POLL_S", "60"))
HEARTBEAT_S = float(os.environ.get("TASK_POOL_HEARTBEAT_S", "30"))
STALE_S = float(os.environ.get("TASK_POOL_STALE_S", "600"))
# For the exit rule a claim counts as running only with a heartbeat this recent: a pod
# that died leaves a claim that is not yet stale, and the workers must not all leave on
# the strength of it.
LIVE_S = float(os.environ.get("TASK_POOL_LIVE_S", str(3 * HEARTBEAT_S)))
MAX_ATTEMPTS = int(os.environ.get("TASK_POOL_MAX_ATTEMPTS", "2"))
STOP_GRACE_S = float(os.environ.get("TASK_POOL_STOP_GRACE_S", "120"))
# A card that uses at least this many MiB before a task holds a process that outlived
# the last one (a vLLM server starts in a session of its own). The lease tool of the
# shared host uses the same limit.
BUSY_MIB = 2000
CARD_WAIT_S = float(os.environ.get("TASK_POOL_CARD_WAIT_S", "300"))


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Pool:
    def __init__(self, dag: Path, state: Path, repo: Path, worker: str):
        self.state, self.repo, self.worker = state, repo, worker
        self.tasks = load_dag(dag)
        self.order = {t["id"]: i for i, t in enumerate(self.tasks)}
        self.by_id = {t["id"]: t for t in self.tasks}
        self.rank = chain_hours(self.tasks)
        for sub in ("claims/.stale", "done", "failed", "attempts", "logs", "workers"):
            (state / sub).mkdir(parents=True, exist_ok=True)
        self.log_path = state / "workers" / f"{worker}.log"
        self.child: subprocess.Popen | None = None
        self.stopping = False

    # -- output -------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{utc()} {self.worker} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    # -- state ----------------------------------------------------------------------------

    def done(self) -> set:
        return {p.stem for p in (self.state / "done").glob("*.json")}

    def attempts(self, tid: str) -> int:
        try:
            return int((self.state / "attempts" / tid).read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def failed(self) -> set:
        """Tasks that failed MAX_ATTEMPTS times and stay failed."""
        return {p.stem for p in (self.state / "failed").glob("*.json")
                if self.attempts(p.stem) >= MAX_ATTEMPTS and p.stem not in self.done()}

    def claim_dir(self, tid: str) -> Path:
        return self.state / "claims" / tid

    def claim_age(self, tid: str) -> float:
        """Seconds since the claim's last heartbeat (or since the claim, before the first)."""
        d = self.claim_dir(tid)
        for p in (d / "heartbeat", d):
            try:
                return time.time() - p.stat().st_mtime
            except OSError:
                continue
        return float("inf")

    def running(self, within: float = STALE_S) -> set:
        """Tasks with a claim whose heartbeat is at most `within` seconds old."""
        out = set()
        for d in (self.state / "claims").iterdir():
            if d.name.startswith(".") or not d.is_dir():
                continue
            if self.claim_age(d.name) <= within:
                out.add(d.name)
        return out

    def reclaim_stale(self) -> None:
        """Move each claim whose heartbeat stopped, so its task is ready again."""
        done = self.done()
        for d in list((self.state / "claims").iterdir()):
            if d.name.startswith(".") or not d.is_dir():
                continue
            if d.name in done:
                shutil.rmtree(d, ignore_errors=True)
                continue
            age = self.claim_age(d.name)
            if age > STALE_S:
                dest = self.state / "claims" / ".stale" / f"{d.name}.{int(time.time())}.{self.worker}"
                try:
                    os.rename(d, dest)          # one worker wins; the others get an error
                    self.log(f"claim of {d.name} was stale ({int(age)} s without a heartbeat); ready again")
                except OSError:
                    pass

    def blocked(self, failed: set) -> set:
        """Every task that comes after a failed one."""
        out, changed = set(failed), True
        while changed:
            changed = False
            for t in self.tasks:
                if t["id"] not in out and any(a in out for a in t["after"]):
                    out.add(t["id"])
                    changed = True
        return out - failed

    def plan(self):
        """(ready, pending, live, failed): ready is in the order to take; live are the
        claims with a recent heartbeat (claimed but not live tasks are neither pending
        nor live until their claim goes stale)."""
        self.reclaim_stale()
        done, failed, claimed = self.done(), self.failed(), self.running()
        blocked = self.blocked(failed)
        pending = [t["id"] for t in self.tasks
                   if t["id"] not in done | failed | claimed | blocked]
        ready = [tid for tid in pending if all(a in done for a in self.by_id[tid]["after"])]
        ready.sort(key=lambda tid: (-self.rank[tid], self.order[tid]))
        live = self.running(LIVE_S)
        # a claim whose heartbeat stopped is work no live worker will finish
        pending += sorted(claimed - live)
        return ready, pending, live, failed | blocked

    def mine(self, tid: str) -> bool:
        """Whether the claim of tid is still this worker's (another worker may have taken
        it over after its heartbeat looked stale)."""
        try:
            return json.loads((self.claim_dir(tid) / "claim.json").read_text()).get("worker") == self.worker
        except (OSError, ValueError):
            return False

    def drop_claim(self, tid: str) -> None:
        if self.mine(tid):
            shutil.rmtree(self.claim_dir(tid), ignore_errors=True)

    def claim(self, tid: str) -> bool:
        try:
            os.mkdir(self.claim_dir(tid))
        except FileExistsError:
            return False
        (self.claim_dir(tid) / "claim.json").write_text(json.dumps(
            {"worker": self.worker, "host": socket.gethostname(), "start": utc()}) + "\n")
        (self.claim_dir(tid) / "heartbeat").touch()
        # Another worker may have finished it between its plan and this claim.
        if tid in self.done():
            self.drop_claim(tid)
            return False
        return True

    # -- running a task -----------------------------------------------------------------------

    def card_used_mib(self) -> int:
        if shutil.which("nvidia-smi") is None:
            return 0
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout
            return max(int(x) for x in out.split()) if out.split() else 0
        except (subprocess.SubprocessError, ValueError):
            return 10 ** 6

    def card_empty(self) -> bool:
        waited = 0.0
        while self.card_used_mib() >= BUSY_MIB:
            if waited >= CARD_WAIT_S:
                return False
            time.sleep(10)
            waited += 10
        return True

    def run(self, tid: str) -> int:
        task = self.by_id[tid]
        start = utc()
        attempt = self.attempts(tid) + 1
        self.log(f"start {tid} (attempt {attempt}): {task['run']}")
        log = open(self.state / "logs" / f"{tid}.log", "a")
        log.write(f"=== {start} {self.worker} attempt {attempt}: {task['run']}\n")
        log.flush()
        stop_beat = threading.Event()
        self.lost = False

        def beat():
            while not stop_beat.wait(HEARTBEAT_S):
                if not self.mine(tid):
                    # Another worker took the task over: stop this copy, so one task never
                    # runs twice (two trainers on one adapter folder).
                    self.lost = True
                    if self.child is not None:
                        try:
                            os.killpg(self.child.pid, signal.SIGINT)
                        except OSError:
                            pass
                    return
                try:
                    (self.claim_dir(tid) / "heartbeat").touch()
                except OSError:
                    pass

        threading.Thread(target=beat, daemon=True).start()
        t0 = time.time()
        # Its own session, so a signal reaches every process of the task and nothing else.
        self.child = subprocess.Popen(["bash", "-o", "pipefail", "-c", task["run"]], cwd=self.repo,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        rc = self.child.wait()
        self.child = None
        stop_beat.set()
        log.close()
        hours = (time.time() - t0) / 3600
        record = {"task": tid, "worker": self.worker, "host": socket.gethostname(), "start": start,
                  "end": utc(), "hours": round(hours, 3), "exit": rc, "attempt": attempt}
        if self.stopping:
            self.log(f"stopped during {tid} (exit {rc}); its claim stays until it goes stale")
            return rc
        if self.lost:
            self.log(f"lost the claim of {tid} to another worker; stopped this copy (exit {rc}), no record")
            return rc
        if rc == 0:
            # done first, then the claim goes: no moment when neither marks the task.
            (self.state / "done" / f"{tid}.json").write_text(json.dumps(record) + "\n")
            self.log(f"done {tid} after {hours:.2f} h")
        else:
            (self.state / "attempts" / tid).write_text(f"{attempt}\n")
            (self.state / "failed" / f"{tid}.json").write_text(json.dumps(record) + "\n")
            left = MAX_ATTEMPTS - attempt
            self.log(f"failed {tid} with exit {rc} after {hours:.2f} h; "
                     + (f"{left} attempt(s) left" if left > 0 else "no attempts left, tasks after it are blocked"))
        self.drop_claim(tid)
        return rc

    def has_work(self) -> bool:
        """Whether a worker that starts now would find a task: one is ready, or more are
        left than the running workers will take."""
        ready, pending, running, _ = self.plan()
        return bool(ready) or len(pending) > len(running)

    def write_summary(self, done: set, stuck: set) -> None:
        failed = self.failed()
        out = {"time": utc(), "tasks": len(self.tasks), "done": sorted(done),
               "failed": sorted(failed), "blocked": sorted(stuck - failed)}
        tmp = self.state / f"summary.json.{self.worker}.tmp"
        tmp.write_text(json.dumps(out, indent=1) + "\n")
        os.replace(tmp, self.state / "summary.json")

    def stop(self, signum, _frame) -> None:
        self.stopping = True
        self.log(f"got signal {signum}")
        child = self.child
        if child is not None and not exited(child.pid):
            os.killpg(child.pid, signal.SIGINT)
            end = time.time() + STOP_GRACE_S
            while not exited(child.pid) and time.time() < end:
                time.sleep(1)
            if not exited(child.pid):
                os.killpg(child.pid, signal.SIGKILL)
        sys.exit(128 + signum)

    # -- the loop -------------------------------------------------------------------------------

    def loop(self) -> int:
        self.log(f"start: {len(self.tasks)} tasks, state {self.state}, repo {self.repo}")
        said_wait = False
        while True:
            ready, pending, running, stuck = self.plan()
            # The card first, before any claim: a claim must not wait without a heartbeat.
            if ready and not self.card_empty():
                self.log(f"the card holds {self.card_used_mib()} MiB before a task; stopping so the pod is replaced")
                return 75
            taken = None
            for tid in ready:
                if self.claim(tid):
                    taken = tid
                    break
            if taken:
                said_wait = False
                self.run(taken)
                continue
            if not pending and not running:
                done = self.done()
                self.write_summary(done, stuck)
                self.log(f"finished: {len(done)} of {len(self.tasks)} tasks done"
                         + (f"; FAILED or blocked: {sorted(stuck)}" if stuck else ""))
                return 0
            # Nothing ready: the tasks left wait on running ones. When the other workers'
            # tasks are as many as the tasks left, those workers take them, and this card
            # is better given back than held empty.
            if len(pending) <= len(running):
                self.log(f"nothing ready; {len(pending)} task(s) left, {len(running)} running on other "
                         "workers, which will take them; exiting")
                return 0
            if not said_wait:
                self.log(f"nothing ready; waiting for {sorted(running)} ({len(pending)} task(s) left)")
                said_wait = True
            time.sleep(POLL_S)


def exited(pid: int) -> bool:
    """Whether the child has exited. The signal handler runs inside child.wait(), which
    holds the Popen lock that poll() needs, so poll() would always say "running"; waitpid
    itself answers."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return done == pid


def load_dag(path: Path) -> list[dict]:
    """The tasks of a task file, checked: unique ids, known and acyclic dependencies."""
    data = yaml.safe_load(Path(path).read_text())
    tasks = data["tasks"] if isinstance(data, dict) else None
    if not tasks:
        raise SystemExit(f"{path}: no tasks")
    out, ids = [], set()
    for t in tasks:
        tid, run = t.get("id"), t.get("run")
        if not tid or not isinstance(tid, str) or "/" in tid or tid.startswith(".") or not run:
            raise SystemExit(f"{path}: a task needs an id (no '/', not starting with '.') and a run: {t}")
        if tid in ids:
            raise SystemExit(f"{path}: task {tid} appears twice")
        ids.add(tid)
        out.append({"id": tid, "run": run, "after": list(t.get("after") or []),
                    "hours": float(t.get("hours", 1.0))})
    for t in out:
        unknown = [a for a in t["after"] if a not in ids]
        if unknown:
            raise SystemExit(f"{path}: task {t['id']} comes after unknown tasks {unknown}")
    chain_hours(out)                            # raises on a cycle
    return out


def chain_hours(tasks: list[dict]) -> dict:
    """For each task, its hours plus the longest chain of hours of the tasks after it."""
    after = {t["id"]: [] for t in tasks}
    for t in tasks:
        for a in t["after"]:
            after[a].append(t["id"])
    hours = {t["id"]: t["hours"] for t in tasks}
    memo, visiting = {}, set()

    def rank(tid):
        if tid in memo:
            return memo[tid]
        if tid in visiting:
            raise SystemExit(f"the tasks form a cycle through {tid}")
        visiting.add(tid)
        memo[tid] = hours[tid] + max((rank(n) for n in after[tid]), default=0.0)
        visiting.discard(tid)
        return memo[tid]

    for tid in hours:
        rank(tid)
    return memo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dag", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--worker", default=os.environ.get("POD_NAME") or socket.gethostname())
    ap.add_argument("--list", action="store_true", help="print the tasks in the order of their rank and exit")
    ap.add_argument("--has-work", action="store_true", help="exit 0 if a new worker would find a task, else 3")
    args = ap.parse_args(argv)
    dag = args.dag if args.dag.is_absolute() else args.repo / args.dag
    pool = Pool(dag, args.state, args.repo, args.worker)
    if args.list:
        for t in sorted(pool.tasks, key=lambda t: (-pool.rank[t["id"]], pool.order[t["id"]])):
            print(f"{pool.rank[t['id']]:6.1f} h  {t['id']:28s} after {','.join(t['after']) or '-'}: "
                  f"{shlex.join(shlex.split(t['run']))}")
        return 0
    if args.has_work:
        return 0 if pool.has_work() else 3
    signal.signal(signal.SIGTERM, pool.stop)
    signal.signal(signal.SIGINT, pool.stop)
    return pool.loop()


if __name__ == "__main__":
    sys.exit(main())
