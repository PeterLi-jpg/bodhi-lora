"""Generate the rebuttal-cell values from raw outputs and diff them against the posted ones (PROV-1).

data/rebuttal_cells/posted_values.json is a hand transcription of the numbers posted on
OpenReview. This script computes the same numbers from the per-seed eval outputs,
writes them in the same shape to data/rebuttal_cells/generated_values.json, and lists
each posted value that does not match at the posted precision.

Layout (rebuttal/launch/_common.sh and run_cell.sh):

    <cells-root>/<bench>__<tag>/seed_<N>/eval/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json
    <cells-root>/<bench>__<tag>/seed_<N>/eval/epistemic_scores.json

<tag> is the HF repo with "/", ":" and "." replaced by "_". The submitted cell is read
from results_modal/seed_<N>/, which has no eval/ folder.

The metrics are the ones scripts/paper_stats.py computes for the paper: a per-seed mean
over scored responses (epistemic metrics) or over all completions (aggregate), then the
mean and the sample SD over seeds. The posts do not say which SD they used.

The per-seed eval JSON of the added cells (full responses, about 370 MB) is copied
from the shared GPU host into results_rebuttal/ and gitignored; the output file is
tracked. Without those copies only the submitted cell has data. The script then names
each missing cell, checks the submitted cell, and writes no file: a file with only the
submitted cell adds nothing to make numbers, which reads that cell from results_modal/,
and it would leave an untracked file in the tree.

Usage:
    python3 rebuttal/analyses/cell_values.py [--cells-root results_rebuttal]
        [--cell-dir <cell id>=<path> ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts import paper_stats as ps  # noqa: E402

# Cell id in posted_values.json -> (benchmark folder prefix, HF repo of the base model).
# The repos are the ones rebuttal/launch/run_grid.sh passes for each cell.
CELLS = {
    "healthbench__medgemma-27b": ("healthbench", "google/medgemma-27b-text-it"),
    "healthbench__mistral-small-24b": ("healthbench", "mistralai/Mistral-Small-24B-Instruct-2501"),
    "healthbench__med42-8b": ("healthbench", "m42-health/Llama3-Med42-8B"),
    "healthbench__biomistral-7b": ("healthbench", "BioMistral/BioMistral-7B"),
    "chatdoctor__mistral-small-24b": ("chatdoctor", "mistralai/Mistral-Small-24B-Instruct-2501"),
    "chatdoctor__med42-8b": ("chatdoctor", "m42-health/Llama3-Med42-8B"),
    "medquad__mistral-small-24b": ("medquad", "mistralai/Mistral-Small-24B-Instruct-2501"),
    "medqa__mistral-small-24b": ("medqa", "mistralai/Mistral-Small-24B-Instruct-2501"),
}


def cell_dir_name(bench: str, repo: str) -> str:
    """The folder name run_cell.sh writes: TAG=$(echo $MODEL | tr '/:.' '___')."""
    return f"{bench}__{re.sub(r'[/:.]', '_', repo)}"


def locate(cell_id: str, cells_root: Path, results_modal: Path, overrides: dict) -> Path | None:
    if cell_id in overrides:
        return overrides[cell_id]
    if cell_id == ps.MAIN_CELL_ID:
        return results_modal
    if cell_id not in CELLS:
        return None
    return cells_root / cell_dir_name(*CELLS[cell_id])


def askers_gap(df) -> dict:
    """Aggregate score of the LoRA responses that ask minus those that do not, pooled over seeds.

    A response asks when its epistemic grade sets active inquiry; responses without a
    parsed grade are left out. The posted replies (HomM_3_followup_part2.md:16) used this
    gap to test whether the rubric penalizes asking. The two groups are self-selected,
    so the gap is descriptive.
    """
    lora = df[(df["config"] == "lora_no_wrapper") & (df["ep_status"] == "scored")]
    ask = lora.loc[lora["active_inquiry"] == 1.0, "healthbench"]
    rest = lora.loc[lora["active_inquiry"] == 0.0, "healthbench"]
    gap = float(ask.mean() - rest.mean()) if len(ask) and len(rest) else None
    return {"gap": gap, "n_ask": int(len(ask)), "n_no_ask": int(len(rest)),
            "seeds_with_grades": sorted(int(s) for s in lora["seed"].unique())}


def cell_values(cell_dir: Path) -> dict:
    """Seed mean and sample SD of every posted metric, per posted condition.

    A seed whose epistemic grading failed has no value for the seven dimensions; the
    mean and SD of those metrics are then null, and per_seed shows which seed is missing.
    """
    seeds = ps.discover_seeds(cell_dir)
    # A rebuttal cell keeps its eval files in seed_<N>/eval/; results_modal does not.
    subdir = "eval" if (cell_dir / f"seed_{seeds[0]}" / "eval").is_dir() else ""
    df = ps.load_responses(cell_dir, seeds, subdir)
    conditions = {}
    for pcond, cfg in ps.POSTED_COND.items():
        conditions[pcond] = {}
        for pmetric, (key, mult, _dp) in ps.POSTED_METRIC.items():
            sm = ps.seed_means(ps.metric_frame(df, key), seeds)[cfg].to_numpy(dtype=float) * mult
            conditions[pcond][pmetric] = {"mean": float(np.mean(sm)), "sd": float(np.std(sm, ddof=1)),
                                          "per_seed": {str(s): float(v) for s, v in zip(seeds, sm)}}
    # Paired t over seeds, LoRA minus Base aggregate score, for Table J.2 and the drop tests.
    # It runs here on the unrounded seed means: the output keeps 6 significant digits per
    # seed, and a test on those moves p in its fourth digit.
    lora, base = (conditions[c]["aggregate"]["per_seed"] for c in ("lora", "base"))
    common = sorted(set(lora) & set(base), key=int)
    test = ps.paired_seed_test(np.array([lora[s] for s in common]), np.array([base[s] for s in common]))
    return {"seeds": seeds, "eval_subdir": subdir, "conditions": conditions,
            "aggregate_lora_minus_base": test,
            "lora_askers_minus_non_askers": askers_gap(df)}


def diff_cell(posted_cell: dict, generated: dict) -> list[dict]:
    """Each posted mean next to the generated one; agrees = equal at the posted decimals."""
    rows = []
    for pcond, metrics in sorted(posted_cell["conditions"].items()):
        for pmetric, v in sorted(metrics.items()):
            if pmetric not in ps.POSTED_METRIC or v.get("mean") is None:
                continue
            dp = ps.POSTED_METRIC[pmetric][2]
            g = generated["conditions"][pcond][pmetric]["mean"]
            agrees = abs(g - v["mean"]) <= 0.5 * 10 ** -dp + 1e-9
            rows.append({"condition": pcond, "metric": pmetric, "posted": v["mean"],
                         "generated": g, "agrees": agrees})
    return rows


def run(posted_path: Path, cells_root: Path, results_modal: Path, overrides: dict, out: Path) -> dict:
    posted = ps.load_posted_values(posted_path)
    res = {"_provenance": "generated by rebuttal/analyses/cell_values.py; metric definitions "
                          "from scripts/paper_stats.py; sd is the sample SD over seeds",
           "cells": [], "missing": [], "mismatches": []}
    for c in posted["cells"]:
        d = locate(c["id"], cells_root, results_modal, overrides)
        if d is None or not d.is_dir():
            res["missing"].append({"id": c["id"], "expected_dir": None if d is None else ps._rel(d)})
            print(f"  missing {c['id']}: {ps._rel(d) if d is not None else 'no folder mapping'}")
            continue
        gen = cell_values(d)
        rows = diff_cell(c, gen)
        res["cells"].append({"id": c["id"], "dir": ps._rel(d), **gen, "diff_vs_posted": rows})
        bad = [r for r in rows if not r["agrees"]]
        res["mismatches"] += [{"id": c["id"], **r} for r in bad]
        print(f"  {c['id']}: {len(gen['seeds'])} seeds, {len(rows)} posted values, "
              f"{len(bad)} mismatch(es)")
        for r in bad:
            print(f"    {r['condition']}.{r['metric']}: posted {r['posted']}, generated {r['generated']:.4f}")
    summary = (f"{len(res['cells'])} cell(s), {len(res['missing'])} missing, "
               f"{len(res['mismatches'])} mismatch(es)")
    if not any(c["id"] != ps.MAIN_CELL_ID for c in res["cells"]):
        print(f"no added cell found, so {ps._rel(out)} is not written: {summary}")
        return res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ps.round6(res), indent=2) + "\n")
    print(f"wrote {ps._rel(out)}: {summary}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--posted", type=Path, default=REPO / "data" / "rebuttal_cells" / "posted_values.json")
    ap.add_argument("--cells-root", type=Path, default=REPO / "results_rebuttal")
    ap.add_argument("--results-modal", type=Path, default=REPO / "results_modal")
    ap.add_argument("--cell-dir", action="append", default=[], metavar="ID=PATH",
                    help="folder of one cell, when its name on the host differs from the default")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "rebuttal_cells" / "generated_values.json")
    a = ap.parse_args()
    overrides = {}
    for item in a.cell_dir:
        cid, sep, path = item.partition("=")
        if not sep:
            ap.error(f"--cell-dir needs ID=PATH, got {item!r}")
        overrides[cid] = Path(path)
    run(a.posted, a.cells_root, a.results_modal, overrides, a.out)


if __name__ == "__main__":
    main()
