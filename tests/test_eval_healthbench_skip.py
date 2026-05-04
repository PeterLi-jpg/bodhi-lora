"""Tests that scripts/eval_healthbench.py is idempotent on existing --output.

The pipeline's stage-4 wrapper run_eval() in tpu/run_pipeline.sh already
short-circuits when the output JSON exists, but a recent preempt-resume
incident showed that direct invocations (or future wrapper changes) could
re-run a 10-15 min eval whose output already landed. The script now does
its own idempotent check immediately after expanding --output. These tests
pin that contract:

  * non-empty output file present  -> exit 0, file untouched
  * empty (size 0) output file     -> NOT skipped (treated as a partial
    write from a crashed run, allowed to be overwritten by the real run)

We invoke the script via subprocess so we exercise the real argparse +
sys.exit path that the launcher hits at runtime. The script's import chain
(eval_healthbench -> filter_traces -> generate_traces -> _bodhi_ablation
-> bodhi, plus torch/transformers/tqdm) reaches into heavy first-party and
ML deps that aren't installed in dev/CI. We stub those packages onto
PYTHONPATH so the import chain succeeds and main() runs to the skip check.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "eval_healthbench.py"


def _write_stub_pkg(root: Path, name: str, body: str = "") -> None:
    """Write a stub package directory with __init__.py at root/name."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(body)


def _make_stub_path(tmp_path: Path) -> Path:
    """Build a directory of stub modules sufficient to import eval_healthbench.

    Only `bodhi` and its submodules need to be stubbed at the package level
    because they're checked-out as an external project that isn't installed
    in dev/CI. Everything else (torch, transformers, numpy, tqdm) is either
    already installed in the preflight venv or shadowed by an entry here.
    """
    stub = tmp_path / "stubs"
    stub.mkdir()

    # bodhi: top-level + .prompts + .constants. _bodhi_ablation does
    # `from bodhi import BODHIConfig` and `from bodhi.prompts import
    # render_analysis_prompt`. Stub just enough symbols to satisfy the
    # import; we never actually call them because we exit before
    # reaching the eval body.
    _write_stub_pkg(
        stub,
        "bodhi",
        body="class BODHIConfig:\n    pass\n",
    )
    _write_stub_pkg(
        stub,
        "bodhi/prompts",
        body="def render_analysis_prompt(*a, **kw):\n    return ''\n",
    )
    # Some downstream imports reference bodhi.constants; stub it too.
    _write_stub_pkg(stub, "bodhi/constants", body="")

    return stub


def _run_script(output_path: Path, *, stub_path: Path) -> subprocess.CompletedProcess:
    """Invoke eval_healthbench.py with --output and minimum-required args.

    Returns the CompletedProcess so callers can assert on returncode and
    captured streams. We pipe a fake --sample-ids file that doesn't have
    to exist because the skip check fires before sample_ids is read.
    """
    env = os.environ.copy()
    # Prepend stub_path so our shims win over any partial install.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{stub_path}{os.pathsep}{REPO_ROOT}"
        + (os.pathsep + existing if existing else "")
    )

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--model", "google/gemma-3-4b-it",
        "--output", str(output_path),
        "--sample-ids", "/tmp/does-not-need-to-exist.json",
    ]
    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _imports_available() -> bool:
    """The skip test only matters when the script's import chain actually
    runs. On a bare dev box without numpy/transformers/etc., the subprocess
    can't even reach the skip block. Skip rather than fail in that case."""
    try:
        import numpy  # noqa: F401
        import transformers  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _imports_available(),
    reason="numpy/transformers/tqdm not installed in this environment",
)


def test_skips_when_output_already_exists_nonempty(tmp_path):
    """Non-empty output file -> script exits 0 without rewriting it."""
    stub_path = _make_stub_path(tmp_path)
    out = tmp_path / "result.json"
    out.write_text('{"already": "graded"}')

    original_mtime_ns = out.stat().st_mtime_ns
    original_bytes = out.read_bytes()

    # Sleep a hair so any accidental rewrite would bump mtime visibly.
    time.sleep(0.05)

    result = _run_script(out, stub_path=stub_path)

    assert result.returncode == 0, (
        f"expected clean exit on existing file, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "skipping" in result.stderr.lower(), (
        f"expected skip notice on stderr, got:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert out.stat().st_mtime_ns == original_mtime_ns, (
        "output file was rewritten despite being non-empty"
    )
    assert out.read_bytes() == original_bytes, (
        "output file content changed despite skip"
    )


def test_does_not_skip_when_output_is_empty(tmp_path):
    """Empty (size 0) output file -> script does NOT skip; it proceeds and
    fails downstream because we passed a fake --sample-ids. The point is
    that we don't see the skip-message and rc != 0 (the real run would
    overwrite the empty file). Crashed-run partial writes must not
    poison resume."""
    stub_path = _make_stub_path(tmp_path)
    out = tmp_path / "empty.json"
    out.touch()  # size-0 file

    assert out.stat().st_size == 0

    result = _run_script(out, stub_path=stub_path)

    # We should NOT see the skip message when the file is empty.
    assert "skipping" not in result.stderr.lower(), (
        f"empty file was incorrectly treated as skip-worthy:\n"
        f"stderr: {result.stderr}"
    )
    # We don't pin returncode further: the real eval needs sample_ids /
    # GPU / etc. and will fail downstream. The only contract this test
    # enforces is "didn't take the skip path on empty file."
