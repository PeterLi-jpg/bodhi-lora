"""Rebuttal analysis: is the Base -> LoRA drop in aggregate rubric score real? (HomM, PROV-2)

The replies (posted/HomM_3_followup_part2.md:14, drafts/meta_review_part2.md:7) say
the aggregate score falls in all six rebuttal cells, with paired t-tests over five
seeds putting all six below p = 0.02 and all six surviving Holm correction. The
numbers sit in rebuttal/responses/drafts/FINDINGS_quality_drop.md, but the code was run
interactively on the remote host and never committed. This is that test as a script.

For each run of the GPU host (host_runs.py), keyed by the run name the host gave it:

    results/<cell>/seed_<N>/eval/{base,lora}.json

it takes the per-seed aggregate score (the ``mean`` field the eval stage writes),
pairs Base and LoRA within each seed, and reports the mean difference, its t interval
(n_seeds - 1 df) and the two-sided paired t p. Holm runs over the cells (statsmodels).

The per-seed eval JSON of the added cells is copied from the shared H100 host and
gitignored (PROV-1). Without that copy the script has nothing to read and exits with a
message.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rebuttal.analyses.host_runs import host_runs  # noqa: E402

# The condition files. The output keeps the names the rows inside them use.
BASE, LORA = "base_no_wrapper", "lora_no_wrapper"
FILES = {BASE: "base", LORA: "lora"}


def seed_scores(cell_dir):
    """seed -> {config: aggregate score} for the seeds that have both configs."""
    out = {}
    for seed_dir in sorted(Path(cell_dir).glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        scores = {}
        for cfg in (BASE, LORA):
            p = seed_dir / "eval" / f"{FILES[cfg]}.json"
            if p.is_file():
                with open(p) as f:
                    scores[cfg] = float(json.load(f)["mean"])
        if len(scores) == 2:
            out[seed] = scores
    return dict(sorted(out.items()))


def paired_test(base, lora):
    """LoRA - Base over seeds: mean difference, 95% t interval, paired t."""
    base, lora = np.asarray(base, dtype=float), np.asarray(lora, dtype=float)
    d = lora - base
    n = len(d)
    half = float(stats.t.ppf(0.975, n - 1)) * d.std(ddof=1) / np.sqrt(n)
    t = stats.ttest_rel(lora, base)
    return {"n_seeds": n, "base_mean": float(base.mean()), "lora_mean": float(lora.mean()),
            "diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "t": float(t.statistic), "df": n - 1, "p": float(t.pvalue),
            "per_seed_diff": d.tolist()}


def analyse(results_root):
    cells = {}
    for name, cell_dir in host_runs(results_root).items():
        per_seed = seed_scores(cell_dir)
        if len(per_seed) < 2:
            print(f"  skip {name}: {len(per_seed)} seed(s) with both configs")
            continue
        r = paired_test([v[BASE] for v in per_seed.values()], [v[LORA] for v in per_seed.values()])
        r["seeds"] = list(per_seed)
        cells[name] = r
    if cells:
        names = list(cells)
        adj = multipletests([cells[c]["p"] for c in names], method="holm")[1]
        for c, p in zip(names, adj):
            cells[c]["p_holm"] = float(p)
    return {"test": "paired t over seeds, LoRA minus Base aggregate score; Holm over cells",
            "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "results" / "cross_cell" / "quality_drop.json"))
    args = ap.parse_args()

    res = analyse(Path(args.results_root))
    if not res["cells"]:
        sys.exit(f"quality_drop: no host run under {args.results_root} has eval files. Sync them "
                 "from the shared H100 box first (PROV-1); the posted t values cannot be checked without them.")
    print(f"{'cell':40s} {'base':>7s} {'lora':>7s} {'diff':>8s} {'t':>7s} {'p':>8s} {'p_holm':>8s}")
    for name, r in res["cells"].items():
        print(f"{name:40s} {r['base_mean']:7.4f} {r['lora_mean']:7.4f} {r['diff']:+8.4f} "
              f"{r['t']:7.2f} {r['p']:8.4f} {r['p_holm']:8.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
