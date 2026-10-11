"""The move into the results root (scripts/migrate_results.py, results/PROVENANCE.md).

Every file git tracked in an old location at the migration base has an entry in
results/provenance.json, the entry's sha256 is the file's hash at that commit, and every
moved file has the same hash where it landed. Ignored files are checked where this
checkout has them; a clean checkout has only the tracked ones. Markdown files are docs and
may change after the move (the validation README now names the new paths), so only data
files are held to their hash.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "results" / "provenance.json"
# The last commit before the move. Its tree holds every old tracked path.
MIGRATION_BASE = "4e3edce"
# The commit that made the move. A file edited after it says when and why in its entry's
# "edited" field, and is held to its bytes at this commit instead of in the checkout.
MOVE_COMMIT = "408e42c"
OLD_TRACKED_ROOTS = ("results_modal", "results_rebuttal", "rebuttal/results", "data/derived",
                     "data/rebuttal_cells", "data/irr_instructions", "data/raw/hard_200_sample_ids.json",
                     "data/sft")


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text, check=True).stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bytes_at_move(path):
    """The file as the move commit wrote it, under the name it had then (it may have moved
    again since)."""
    try:
        _git("cat-file", "-e", f"{MOVE_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        pytest.skip(f"{MOVE_COMMIT} is not in this clone (shallow checkout)")
    log = _git("log", "--follow", "--format=%h", "--name-only", f"{MOVE_COMMIT}^..HEAD", "--", path)
    name = [line for line in log.splitlines() if line.strip()][-1]
    return _git("show", f"{MOVE_COMMIT}:{name}", text=False)


@pytest.fixture(scope="module")
def entries():
    return json.loads(PROVENANCE.read_text())


@pytest.fixture(scope="module")
def base_files():
    try:
        _git("cat-file", "-e", f"{MIGRATION_BASE}^{{commit}}")
    except subprocess.CalledProcessError:
        pytest.skip(f"{MIGRATION_BASE} is not in this clone (shallow checkout)")
    return _git("ls-tree", "-r", "--name-only", MIGRATION_BASE, "--", *OLD_TRACKED_ROOTS).split()


def test_every_old_tracked_file_has_an_entry(entries, base_files):
    by_old = {e["old"]: e for e in entries}
    assert base_files, "no old tracked files found at the migration base"
    missing = [p for p in base_files if p not in by_old]
    assert missing == []
    assert all(by_old[p]["tracked_before"] for p in base_files)


def test_tracked_entries_hash_as_at_the_migration_base(entries, base_files):
    for e in entries:
        if e["old"] in base_files:
            blob = _git("show", f"{MIGRATION_BASE}:{e['old']}", text=False)
            assert _sha256(blob) == e["sha256"], e["old"]


def test_moved_files_keep_their_bytes(entries):
    tracked = set(_git("ls-files").splitlines())
    checked = 0
    for e in entries:
        if e["kind"] != "moved" or e["old"].endswith(".md"):
            continue
        for new in e["new"]:
            path = ROOT / new
            if new in tracked:
                assert path.is_file(), new
            if "edited" in e:
                assert _sha256(_bytes_at_move(new)) == e["sha256"], f"{e['old']} -> {new} at {MOVE_COMMIT}"
                checked += 1
            elif path.is_file():
                assert _sha256(path.read_bytes()) == e["sha256"], f"{e['old']} -> {new}"
                checked += 1
    assert checked > 0


def test_tracked_after_matches_git(entries):
    tracked = set(_git("ls-files").splitlines())
    for e in entries:
        if e["kind"] == "moved":
            assert e["tracked_after"] == all(n in tracked for n in e["new"]), e["old"]


def test_no_old_location_is_tracked_any_more():
    left = _git("ls-files", "--", *OLD_TRACKED_ROOTS).split()
    assert left == []


def test_every_entry_says_where_it_went(entries):
    kinds = {"moved", "archived", "merged"}
    for e in entries:
        assert e["kind"] in kinds and e["new"], e["old"]
        if e["kind"] == "archived":
            assert all(n.startswith("_archive/") for n in e["new"]), e["old"]
