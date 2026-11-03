"""The repository tools around the paper's numbers: the macro check, the tracked score
copies and the doc path check (scripts/check_paper_macros.py,
scripts/export_derived_inputs.py, scripts/check_doc_paths.py)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
# The submitted cell and the 2026-04 pilot (results/PROVENANCE.md).
M = RESULTS / "healthbench__medgemma-27b-text__bodhi__2026-05"
PILOT = RESULTS / "healthbench__medgemma-27b-text__bodhi__2026-04"
FILTER_SCORES = M / "seed_7" / "filter" / "filter_scores.json"
QWEN_SCORES = PILOT / "analysis" / "qwen_grader_scores.json"


def _without_source_paths(obj):
    """The copy minus its "source" path labels, which name where the export read."""
    if isinstance(obj, dict):
        return {k: _without_source_paths(v) for k, v in obj.items() if k != "source"}
    return obj


@pytest.mark.parametrize("which", ["filter", "qwen"])
def test_derived_inputs_match_their_sources(tmp_path, which):
    """The tracked score copies equal a fresh export from the ignored source files:
    the same rows and the same source sha256."""
    from scripts import export_derived_inputs as ex
    src = {"filter": M / "seed_7" / "filter" / "train.jsonl",
           "qwen": PILOT / "seed_42" / "eval" / "base.json"}[which]
    if not src.is_file():
        pytest.skip("ignored source file not present")
    tracked = FILTER_SCORES if which == "filter" else QWEN_SCORES
    fresh = ex.export_filter_scores(src) if which == "filter" else ex.export_qwen_scores(src.parent)
    ex._write(tmp_path / "copy.json", fresh)
    assert (_without_source_paths(json.loads((tmp_path / "copy.json").read_text()))
            == _without_source_paths(json.loads(tracked.read_text())))


def test_check_paper_macros_names_missing(tmp_path, monkeypatch, capsys):
    from scripts import check_paper_macros as cm
    parts = sorted((ROOT / "paper" / "generated").glob("numbers_*.tex"))
    full = "".join(p.read_text() for p in parts)
    assert parts and cm.main() == 0
    # Drop one macro the paper uses: the check must fail and name it, and only it.
    name = "statHBMedGemmaMayFilterTrainSplit"
    assert f"\\newcommand{{\\{name}}}" in full
    cut = "\n".join(line for line in full.splitlines() if not line.startswith(f"\\newcommand{{\\{name}}}"))
    (tmp_path / "numbers.tex").write_text(cut)
    monkeypatch.setattr(cm, "NUMBERS", tmp_path / "numbers.tex")
    monkeypatch.setattr(cm, "ROOT", ROOT)
    capsys.readouterr()
    assert cm.main() == 1
    out = capsys.readouterr().out
    assert name in out and "1 macro(s) not in" in out


def test_check_doc_paths_accepts_only_the_paths_the_copy_withheld(tmp_path):
    """The anonymous copy drops internal files that its docs still name. The list of them
    excuses those paths and no others."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "check_doc_paths.py", tmp_path / "scripts")
    (tmp_path / "README.md").write_text("See `docs/GONE.md`, `hooks/run.sh` and `scripts/check_doc_paths.py`.\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    def check():
        return subprocess.run([sys.executable, "scripts/check_doc_paths.py"], cwd=tmp_path,
                              capture_output=True, text=True)

    first = check()
    assert first.returncode == 1 and "docs/GONE.md" in first.stdout and "hooks/run.sh" in first.stdout
    (tmp_path / "WITHHELD_PATHS.txt").write_text("docs/GONE.md\n")
    second = check()
    assert second.returncode == 1 and "docs/GONE.md" not in second.stdout and "hooks/run.sh" in second.stdout
    (tmp_path / "WITHHELD_PATHS.txt").write_text("docs/GONE.md\nhooks/\n")
    assert check().returncode == 0
