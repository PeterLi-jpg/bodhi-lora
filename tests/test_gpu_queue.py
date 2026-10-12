"""scripts/gpu_queue.sh on the CPU (docs/GPU_QUEUE.md).

Each test runs the script with bash, with a fake lease tool, a fake tmux and a fake
nvidia-smi first on PATH, and small shell commands as the queue. One test runs the
node's lease tool itself, from the commit that last had it, to check the lease file
format the script reads.
"""
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUEUE_SH = ROOT / "scripts" / "gpu_queue.sh"
# The last commit with the node's lease tool in the repo (docs/RUN_PLAN.md section 1).
LEASE_TOOL_AT = "7e02348:scripts/gpu_lease.sh"
GPU = "3"

# A stand-in for the node's lease tool: the same commands and the same table, without
# the lock and the nvidia-smi checks. Each call is logged with the CLAIM_PID it saw.
FAKE_LEASE = r"""#!/usr/bin/env bash
f="$FAKE_LEASE_FILE"
echo "$1 $2 CLAIM_PID=${CLAIM_PID:-}" >>"$FAKE_LEASE_CALLS"
row() { awk -v i="$1" -v s="$2" -v o="$3" -v p="$4" -v t="$5" -v u="$6" '
    $1 == i { printf "%-4s %-6s %-10s %-9s %-20s %s\n", i, s, o, p, t, u; next } { print }' "$f" >"$f.tmp" && mv "$f.tmp" "$f"; }
case $1 in
    claim)
        if [ "$(awk -v i="$2" '$1 == i { print $2 }' "$f")" = busy ]; then
            echo "GPU $2 already claimed" >&2
            exit 1
        fi
        row "$2" busy "$3" "${CLAIM_PID:-$PPID}" 2026-10-12T00:00:00Z "$4"
        echo "claimed GPU $2 for $3" ;;
    release)
        row "$2" free - - - -
        echo "released GPU $2" ;;
    *) exit 2 ;;
esac
"""

# has-session -t =NAME succeeds while the file $FAKE_TMUX_DIR/NAME exists.
FAKE_TMUX = r"""#!/usr/bin/env bash
[ "$1" = has-session ] || exit 2
[ -e "$FAKE_TMUX_DIR/${3#=}" ]
"""

# The memory query prints $FAKE_GPU_MIB; any other call prints a line for the log.
FAKE_NVIDIA_SMI = r"""#!/usr/bin/env bash
case "$*" in
    *--query-gpu=memory.used*) echo "${FAKE_GPU_MIB:-0}" ;;
    *) echo "fake nvidia-smi $*" ;;
esac
"""

TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ")


def _write_exec(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


def _lease_table(owner_of_3=None) -> str:
    lines = ["# GPU leases for this node.", "#idx state  owner      pid       since                purpose"]
    for i in range(8):
        if i == 3 and owner_of_3:
            lines.append(f"{i:<4} busy   {owner_of_3[0]:<10} {owner_of_3[1]:<9} 2026-10-11T00:00:00Z  other job")
        else:
            lines.append(f"{i:<4} free   -          -         -                    -")
    return "\n".join(lines) + "\n"


def _row(lease_file: Path, gpu: str = GPU) -> list:
    for line in lease_file.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == gpu:
            return fields
    raise AssertionError(f"no row for GPU {gpu}")


class Node:
    """A fake shared node in tmp_path: tools on PATH, a lease file, a log folder and a
    working folder for the commands."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.lease_tool = _write_exec(self.bin / "gpu_lease.sh", FAKE_LEASE)
        _write_exec(self.bin / "tmux", FAKE_TMUX)
        _write_exec(self.bin / "nvidia-smi", FAKE_NVIDIA_SMI)
        self.lease_file = tmp / "GPUS.txt"
        self.lease_file.write_text(_lease_table())
        self.calls = tmp / "lease_calls.txt"
        self.calls.touch()
        self.sessions = tmp / "sessions"
        self.sessions.mkdir()
        self.logs = tmp / "logs"
        self.repo = tmp / "work"
        self.repo.mkdir()
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_LEASE_FILE": str(self.lease_file),
            "FAKE_LEASE_CALLS": str(self.calls),
            "FAKE_TMUX_DIR": str(self.sessions),
            "GPU_QUEUE_POLL_S": "1",
            "GPU_QUEUE_STOP_GRACE_S": "10",
            "GPU_QUEUE_MEM_WAIT_S": "0",
        }
        # A GPU job of the test machine must not leak into the queue's environment.
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPU_LEASE_FILE"):
            self.env.pop(key, None)

    def queue_file(self, lines) -> Path:
        path = self.tmp / "q.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def args(self, queue, *extra) -> list:
        return ["bash", str(QUEUE_SH), "--gpu", GPU, "--owner", "tester", "--purpose", "test run",
                "--queue", str(queue), "--name", "q", "--log-dir", str(self.logs),
                "--repo", str(self.repo), "--lease-tool", str(self.lease_tool),
                "--lease-file", str(self.lease_file), *extra]

    def run(self, lines, *extra, env=None, timeout=60):
        queue = lines if isinstance(lines, (str, Path)) else self.queue_file(lines)
        return subprocess.run(self.args(queue, *extra), env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=timeout)

    def start(self, lines, *extra, env=None):
        return subprocess.Popen(self.args(self.queue_file(lines), *extra), env={**self.env, **(env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def marker(self, name="q") -> list:
        return (self.logs / f"{name}.exit").read_text().splitlines()

    def queue_log(self, name="q") -> str:
        return (self.logs / f"{name}.log").read_text()

    def lease_calls(self) -> list:
        return self.calls.read_text().splitlines()


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def _wait_for(path: Path, timeout=20.0):
    end = time.time() + timeout
    while not path.exists():
        if time.time() > end:
            raise AssertionError(f"{path} did not appear")
        time.sleep(0.05)


def test_runs_each_line_in_order_on_the_claimed_gpu_and_releases_it(node):
    out = node.tmp / "out.txt"
    proc = node.run([
        "# a comment line",
        f'echo "one $CUDA_VISIBLE_DEVICES $CUDA_DEVICE_ORDER $PWD" >> {out}',
        "",
        # The eval stage prints scores; they belong in the step log only. The printed
        # text "score 77" is not in the command line, which the queue log does show.
        f"printf 'score %s\\n' $((70 + 7)); echo two >> {out}",
    ])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.read_text().splitlines() == [f"one {GPU} PCI_BUS_ID {node.repo}", "two"]
    assert node.marker()[0] == "0"
    assert _row(node.lease_file)[1] == "free"
    # The lease names the queue's own shell, the PID of the process the test started.
    calls = node.lease_calls()
    assert [c.split()[0] for c in calls] == ["claim", "release"]
    log = node.queue_log()
    pid = re.search(r"pid (\d+)", log).group(1)
    assert calls[0] == f"claim {GPU} CLAIM_PID={pid}"
    assert all(TIMESTAMP.match(line) for line in log.splitlines())
    # Step output goes to the step logs, not to the queue log or the terminal.
    assert "score 77" in (node.logs / "q.step02.log").read_text()
    assert "score 77" not in log and "score 77" not in proc.stdout
    assert (node.logs / "q.queue.txt").is_file()


def test_stops_at_the_first_failing_line(node):
    out = node.tmp / "out.txt"
    proc = node.run([f"echo a >> {out}", "exit 7", f"echo c >> {out}"])
    assert proc.returncode == 7
    assert out.read_text() == "a\n"
    marker = node.marker()
    assert marker[0] == "7" and "step 2/3" in marker[1]
    assert _row(node.lease_file)[1] == "free"


def test_refuses_a_gpu_that_another_job_holds(node):
    node.lease_file.write_text(_lease_table(owner_of_3=("other", "4242")))
    out = node.tmp / "out.txt"
    proc = node.run([f"echo ran >> {out}"])
    assert proc.returncode == 4
    assert not out.exists()
    assert _row(node.lease_file)[1:4] == ["busy", "other", "4242"]
    # The queue never releases a lease it does not hold.
    assert [c.split()[0] for c in node.lease_calls()] == ["claim"]
    assert node.marker()[0] == "4"


def test_waits_for_the_session_before_it_then_starts(node):
    (node.sessions / "prev").touch()
    node.logs.mkdir()
    (node.logs / "prev.exit").write_text("running\n")
    out = node.tmp / "out.txt"
    proc = node.start([f"echo ran >> {out}"], "--after", "prev")
    try:
        time.sleep(2.5)
        assert proc.poll() is None
        assert node.lease_calls() == [] and not out.exists()
        (node.logs / "prev.exit").write_text("0\nall 3 steps finished\n")
        (node.sessions / "prev").unlink()
        assert proc.wait(timeout=30) == 0
    finally:
        proc.kill()
    assert out.read_text() == "ran\n"


@pytest.mark.parametrize("marker", ["3\nprevious queue stopped\n", "running\n", None])
def test_does_not_start_when_the_queue_before_it_did_not_finish(node, marker):
    node.logs.mkdir()
    prev = node.tmp / "prev-marker"
    if marker is not None:
        prev.write_text(marker)
    out = node.tmp / "out.txt"
    proc = node.run([f"echo ran >> {out}"], "--after", "prev", "--after-marker", str(prev))
    assert proc.returncode == 3
    assert not out.exists()
    assert node.lease_calls() == []
    assert node.marker()[0] == "3"


@pytest.mark.parametrize("line", ["true --gpus 0", "true --gpus=3,4", "true --gpus 3 --gpus 2", "true --gpus"])
def test_refuses_a_line_for_another_gpu_before_any_line_runs(node, line):
    out = node.tmp / "out.txt"
    proc = node.run([f"echo ran >> {out}", line])
    assert proc.returncode == 7
    assert not out.exists()
    assert _row(node.lease_file)[1] == "free"


def test_accepts_lines_for_its_own_gpu(node):
    proc = node.run(["true --gpus 3", "true --gpus=3"])
    assert proc.returncode == 0, proc.stdout


def test_steps_get_the_default_signal_handling(node):
    """The queue catches SIGPIPE for itself, but a command must see the default, or a
    pipe in a queue line would fail where it works by hand."""
    out = node.tmp / "out.txt"
    proc = node.run([f"trap -p PIPE INT TERM > {out}"])
    assert proc.returncode == 0, proc.stdout
    assert out.read_text() == ""


def test_sources_the_env_file_before_the_gpu_pin(node):
    env_file = node.tmp / "run.env"
    env_file.write_text("FROM_ENV=yes\nCUDA_VISIBLE_DEVICES=7\n")
    out = node.tmp / "out.txt"
    proc = node.run([f'echo "$FROM_ENV $CUDA_VISIBLE_DEVICES" > {out}'], "--env", str(env_file))
    assert proc.returncode == 0, proc.stdout
    assert out.read_text() == f"yes {GPU}\n"


@pytest.mark.parametrize("sig,code", [(signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)])
def test_a_signal_stops_the_step_with_sigint_and_releases_the_gpu(node, sig, code):
    started, stopped, out = node.tmp / "started", node.tmp / "stopped", node.tmp / "out.txt"
    proc = node.start([
        f'trap "echo int > {stopped}; exit 130" INT; touch {started}; while :; do sleep 0.1; done',
        f"echo never >> {out}",
    ])
    try:
        _wait_for(started)
        proc.send_signal(sig)
        assert proc.wait(timeout=30) == code
    finally:
        proc.kill()
    assert stopped.read_text() == "int\n"
    assert not out.exists()
    assert node.marker()[0] == str(code)
    assert _row(node.lease_file)[1] == "free"


def test_keeps_the_lease_while_the_card_still_holds_memory(node):
    proc = node.run(["true"], env={"FAKE_GPU_MIB": "41000"})
    assert proc.returncode == 0
    row = _row(node.lease_file)
    assert row[1] == "busy" and row[2] == "tester"
    assert [c.split()[0] for c in node.lease_calls()] == ["claim"]
    assert "lease of GPU 3 kept" in node.marker()[1]


def test_works_with_the_node_lease_tool(node, tmp_path):
    """The node's tool (the last committed copy) and this script agree on the lease file:
    during a run the row names the queue's PID, and the end frees it."""
    try:
        text = subprocess.run(["git", "-C", str(ROOT), "show", LEASE_TOOL_AT], capture_output=True,
                              text=True, check=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip(f"{LEASE_TOOL_AT} is not in this clone's history")
    tool = _write_exec(tmp_path / "real_gpu_lease.sh", text)
    if subprocess.run(["bash", "-c", "command -v flock"], env=node.env, capture_output=True).returncode:
        # macOS has no flock; the lock is the tool's own business, not the queue's.
        _write_exec(node.bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    lease_file = tmp_path / "real_GPUS.txt"
    seen = tmp_path / "seen.txt"
    queue = node.queue_file([f"cp {lease_file} {seen}"])
    args = node.args(queue)
    args[args.index("--lease-tool") + 1] = str(tool)
    args[args.index("--lease-file") + 1] = str(lease_file)
    proc = subprocess.run(args, env={**node.env, "GPU_LEASE_FILE": str(lease_file)},
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    during = _row(seen)
    assert during[1:4] == ["busy", "tester", str(proc_pid(node))]
    assert _row(lease_file)[1] == "free"


def proc_pid(node) -> str:
    return re.search(r"pid (\d+)", node.queue_log()).group(1)


# -- the checkout switch -------------------------------------------------------------

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args], check=True,
                          capture_output=True, text=True, env={**os.environ, **GIT_ENV}).stdout.strip()


class Repos:
    """An upstream repo and a node checkout of its first commit, in which the lease tool
    is a local copy marked skip-worktree. The bundle brings two later commits, each with
    a queue file: one drops the tool, stops tracking a stored eval file (as the new code
    does with the 2026-05 cell's) and changes a manifest; the other changes the tool."""

    def __init__(self, node: Node):
        up = node.tmp / "upstream"
        up.mkdir()
        _git(up, "init", "-q")
        (up / "scripts").mkdir()
        (up / "scripts" / "gpu_lease.sh").write_text("committed tool\n")
        (up / "a.txt").write_text("a\n")
        self.stored = Path("results") / "cell" / "seed_42" / "eval" / "base.json"
        self.manifest = Path("results") / "cell" / "manifest.json"
        (up / self.stored).parent.mkdir(parents=True)
        (up / self.stored).write_text("stored replies\n")
        (up / self.manifest).write_text("old manifest\n")
        _git(up, "add", "-A")
        _git(up, "commit", "-q", "-m", "one")
        self.c1 = _git(up, "rev-parse", "HEAD")
        _git(up, "branch", "-M", "main")
        # The node clones before the later commits exist, so only the bundle has them.
        self.node = node.repo = node.tmp / "checkout"
        _git(node.tmp, "clone", "-q", str(up), str(self.node))
        self.out = node.tmp / "out.txt"
        queue = f"git rev-parse HEAD >> {self.out}\n"
        (up / "configs" / "queues").mkdir(parents=True)
        (up / "configs" / "queues" / "t.txt").write_text(queue)
        _git(up, "rm", "-q", "scripts/gpu_lease.sh")
        _git(up, "rm", "-q", "--cached", str(self.stored))
        (up / ".gitignore").write_text("results/**/eval/*.json\n")
        (up / self.manifest).write_text("new manifest\n")
        _git(up, "add", "-A")
        _git(up, "commit", "-q", "-m", "drop the tool")
        self.drops = _git(up, "rev-parse", "HEAD")
        _git(up, "checkout", "-q", "-b", "keeps", self.c1)
        (up / "configs" / "queues").mkdir(parents=True)
        (up / "configs" / "queues" / "t.txt").write_text(queue)
        (up / "scripts" / "gpu_lease.sh").write_text("committed tool, version 2\n")
        _git(up, "add", "-A")
        _git(up, "commit", "-q", "-m", "change the tool")
        self.changes = _git(up, "rev-parse", "HEAD")
        self.bundle = node.tmp / "work.bundle"
        _git(up, "bundle", "create", "-q", str(self.bundle), f"{self.c1}..main", f"{self.c1}..keeps")
        _git(self.node, "checkout", "-q", "--detach", self.c1)
        (self.node / "scripts" / "gpu_lease.sh").write_text("node copy\n")
        _git(self.node, "update-index", "--skip-worktree", "scripts/gpu_lease.sh")
        assert _git(self.node, "status", "--porcelain") == ""
        self.refs = _git(self.node, "for-each-ref")

    def head(self) -> str:
        return _git(self.node, "rev-parse", "HEAD")

    def tool(self) -> str:
        return (self.node / "scripts" / "gpu_lease.sh").read_text()

    def status(self) -> str:
        return _git(self.node, "status", "--porcelain", "--untracked-files=all")


@pytest.fixture
def repos(node):
    return Repos(node)


def _switch(node, repos, commit, *extra):
    return node.run("configs/queues/t.txt", "--bundle", str(repos.bundle), "--commit", commit, *extra)


def test_switch_keeps_the_node_copy_when_the_commit_drops_the_file(node, repos):
    proc = _switch(node, repos, repos.drops, "--keep-local", "scripts/gpu_lease.sh")
    assert proc.returncode == 0, node.queue_log()
    assert repos.head() == repos.drops
    assert repos.out.read_text() == repos.drops + "\n"
    assert repos.tool() == "node copy\n"
    assert repos.status() == ""
    assert "/scripts/gpu_lease.sh" in (repos.node / ".git" / "info" / "exclude").read_text().splitlines()
    # The fetch touched no branch or remote ref.
    assert _git(repos.node, "for-each-ref") == repos.refs


def test_switch_brings_back_committed_files_that_the_commit_drops(node, repos):
    """A folder keeps every file that the switch deletes, as the host needs the 2026-05
    cell's eval files for R6; a file that the commit still tracks takes its new content."""
    proc = _switch(node, repos, repos.drops, "--keep-local", "scripts/gpu_lease.sh",
                   "--keep-local", "results/cell")
    assert proc.returncode == 0, node.queue_log()
    assert (repos.node / repos.stored).read_text() == "stored replies\n"
    assert (repos.node / repos.manifest).read_text() == "new manifest\n"
    assert repos.status() == ""
    # The commit's own ignore rule covers the eval file, so only the tool is excluded.
    assert (repos.node / ".git" / "info" / "exclude").read_text().splitlines()[-1] == "/scripts/gpu_lease.sh"


def test_switch_without_keep_local_for_a_folder_loses_its_dropped_files(node, repos):
    proc = _switch(node, repos, repos.drops, "--keep-local", "scripts/gpu_lease.sh")
    assert proc.returncode == 0, node.queue_log()
    assert not (repos.node / repos.stored).exists()


def test_switch_sets_skip_worktree_again_when_the_commit_tracks_the_file(node, repos):
    # An absolute --keep-local path inside the checkout works as a relative one.
    proc = _switch(node, repos, repos.changes[:10], "--keep-local", str(repos.node / "scripts" / "gpu_lease.sh"))
    assert proc.returncode == 0, node.queue_log()
    assert repos.head() == repos.changes
    assert repos.tool() == "node copy\n"
    assert _git(repos.node, "ls-files", "-v", "scripts/gpu_lease.sh").startswith("S ")
    assert repos.status() == ""


def test_switch_refuses_untracked_files_at_the_start(node, repos):
    """The check runs before the wait and the claim, so the mistake shows at once."""
    (repos.node / "stray.txt").write_text("x\n")
    proc = _switch(node, repos, repos.drops, "--keep-local", "scripts/gpu_lease.sh",
                   "--allow-untracked", "results/")
    assert proc.returncode == 5
    assert "stray.txt" in node.queue_log()
    assert repos.head() == repos.c1 and repos.tool() == "node copy\n"
    assert not repos.out.exists()
    assert node.lease_calls() == []


def test_switch_allows_untracked_run_outputs_under_the_given_path(node, repos):
    # A run writes manifests and tables that git does not ignore (results/ of R1 and R2).
    manifest = repos.node / "results" / "r1-cell" / "seed_42" / "grade" / "judge__x.manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    proc = _switch(node, repos, repos.drops, "--keep-local", "scripts/gpu_lease.sh",
                   "--allow-untracked", "results/")
    assert proc.returncode == 0, node.queue_log()
    assert repos.head() == repos.drops and manifest.read_text() == "{}\n"


def test_switch_without_keep_local_leaves_the_checkout_as_it_was(node, repos):
    # git refuses to overwrite the changed skip-worktree file; nothing moves.
    proc = _switch(node, repos, repos.changes)
    assert proc.returncode == 5
    assert repos.head() == repos.c1 and repos.tool() == "node copy\n"
    assert _git(repos.node, "ls-files", "-v", "scripts/gpu_lease.sh").startswith("S ")
    assert not repos.out.exists()


def test_switch_stops_on_a_commit_the_bundle_does_not_have(node, repos):
    proc = _switch(node, repos, "0123456789abcdef0123456789abcdef01234567", "--keep-local", "scripts/gpu_lease.sh")
    assert proc.returncode == 5
    assert repos.head() == repos.c1 and repos.tool() == "node copy\n"

