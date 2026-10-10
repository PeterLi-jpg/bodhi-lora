"""The repository tools around the paper's numbers: the macro check and the tracked
score copies (scripts/check_paper_macros.py, scripts/export_derived_inputs.py)."""
import json
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
    full = (ROOT / "paper" / "generated" / "numbers.tex").read_text()
    assert cm.main() == 0
    # Drop one macro the paper uses: the check must fail and name it.
    name = "statHBMedGemmaMayFilterTrainSplit"
    cut = "\n".join(line for line in full.splitlines() if not line.startswith(f"\\newcommand{{\\{name}}}"))
    (tmp_path / "numbers.tex").write_text(cut)
    monkeypatch.setattr(cm, "NUMBERS", tmp_path / "numbers.tex")
    monkeypatch.setattr(cm, "ROOT", ROOT)
    capsys.readouterr()
    assert cm.main() == 1
    assert name in capsys.readouterr().out
