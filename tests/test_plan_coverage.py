"""The plans and the registry name the same cells (docs/CONVENTIONS.md, convention 3).

Every planned cell record is named in the dated analysis plan, so no cell runs without
a test fixed in advance. Every cell id the plans name has a record, so a renamed cell
cannot leave a plan pointing at nothing.
"""
import re

import pytest

pytest.importorskip("pydantic")

from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

PLANS = ("docs/ANALYSIS_PLAN.md", "docs/RUN_PLAN.md", "docs/EXPERIMENTS.md")
# A full cell id in a code span: <benchmark>__<model>__<protocol>[__<tag>].
CELL_ID = re.compile(r"`([a-z0-9][a-z0-9.-]*__[a-z0-9][a-z0-9.-]*__[a-z0-9][a-z0-9-]*(?:__[a-z0-9][a-z0-9-]*)?)`")


@pytest.fixture(scope="module")
def reg():
    return load_registry(CONFIGS_DIR)


def _named(doc: str) -> set[str]:
    return set(CELL_ID.findall((REPO_ROOT / doc).read_text(encoding="utf-8")))


def test_every_planned_cell_is_named_in_the_analysis_plan(reg):
    planned = {cid for cid, c in reg.cells.items() if c.report.status == "planned"}
    missing = sorted(planned - _named("docs/ANALYSIS_PLAN.md"))
    assert not missing, f"planned cells the analysis plan does not name: {missing}"


@pytest.mark.parametrize("doc", PLANS)
def test_every_cell_a_plan_names_has_a_record(reg, doc):
    unknown = sorted(_named(doc) - set(reg.cells))
    assert not unknown, f"{doc} names cells with no record in configs/cells/: {unknown}"
