"""Filter threshold, trace counts and LoRA recipe of each added cell, from the host logs (PROV-3).

The paper's Appendix J and Table F.2 need, per cell, the tau the filter used, how many
traces it kept, the train/val split and the recipe of the evaluated adapters. These sit
in two places:

  logs/host/filter_<bench>_<tag>.log      the last filter run of the cell, copied from the
                                          GPU host (gitignored, 28 MB). Its lines
                                          "Kept K/G (score_field=..., threshold=T)" and
                                          "raw=R graded=G kept=K train=N val=M" give tau
                                          and the counts.
  logs/host/*.log                         any "re-filtering at 22nd-pct tau" line. The
                                          launcher that made the posted cells cut again at
                                          the 22nd percentile when a non-HealthBench cell
                                          kept under 40% (1262448:rebuttal/launch/run_cell.sh:160-200).
  results_rebuttal/<cell>/seed_<N>/ckpt/best/{adapter_config,trainer_state}.json
                                          tracked; r, alpha, dropout and target modules,
                                          and the steps, epochs and best checkpoint.

The script checks the two against each other: the train count from the log, the
per-device batch from trainer_state.json and the optimizer steps per epoch must admit
one gradient-accumulation value (HF Trainer: steps per epoch =
ceil(ceil(train / per-device batch) / accumulation)). It writes
rebuttal/results/cell_recipe.json, which scripts/paper_stats.py reads.

Every folder under results_rebuttal/ is listed, including runs that did not finish, with
a status: "complete" (5 seeds trained), "partial" (fewer seeds) or "not run" (no
adapter; the last error line of the generation log is kept). n_eval_ids is the size of
the cell's eval_ids.json.

Usage:
    python3 rebuttal/analyses/cell_recipe.py [--cells-root results_rebuttal] [--logs logs/host]
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

KEPT = re.compile(r"^Kept (\d+)/(\d+) \(score_field=(\S+), threshold=([0-9.eE+-]+)\)", re.M)
SUMMARY = re.compile(r"^raw=(\d+) graded=(\d+) kept=(\d+) train=(\d+) val=(\d+)", re.M)
PARSE = re.compile(r"parse failures: (\d+)/(\d+)")
REFILTER = re.compile(r"re-filtering at")
N_SEEDS = 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _text(path: Path) -> str:
    # Progress bars redraw with carriage returns; split them into lines.
    return path.read_text(errors="replace").replace("\r", "\n")


def filter_log(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = _text(path)
    kept = KEPT.findall(text)
    summ = SUMMARY.findall(text)
    if not kept or not summ:
        return {"path": _rel(path), "sha256": _sha256(path), "parsed": False}
    k, g, field, tau = kept[-1]
    raw, graded, k2, train, val = map(int, summ[-1])
    if (int(k), int(g)) != (k2, graded):
        raise SystemExit(f"{path}: Kept line {k}/{g} disagrees with the summary {k2}/{graded}")
    pf = PARSE.findall(text)
    return {"path": _rel(path), "sha256": _sha256(path), "parsed": True,
            "tau": float(tau), "score_field": field, "n_filter_passes": len(kept),
            "raw": raw, "graded": graded, "kept": k2, "train": train, "val": val,
            "kept_pct": 100.0 * k2 / graded,
            "rubric_parse_failures": [int(pf[-1][0]), int(pf[-1][1])] if pf else None}


def last_error(path: Path) -> str | None:
    if not path.is_file():
        return None
    errs = [ln.strip() for ln in _text(path).splitlines() if "Error" in ln]
    return errs[-1] if errs else None


def accumulation(train: int, per_device: int, steps_per_epoch: int) -> list[int]:
    """Gradient-accumulation values that give this many optimizer steps per epoch."""
    batches = math.ceil(train / per_device)
    return [a for a in range(1, 65) if math.ceil(batches / a) == steps_per_epoch]


def seed_recipe(best: Path) -> dict:
    a = json.loads((best / "adapter_config.json").read_text())
    t = json.loads((best / "trainer_state.json").read_text())
    return {"base_model": a["base_model_name_or_path"], "r": a["r"], "lora_alpha": a["lora_alpha"],
            "lora_dropout": a["lora_dropout"], "target_modules": sorted(a["target_modules"]),
            "peft_version": a.get("peft_version"), "max_steps": t["max_steps"],
            "num_train_epochs": t["num_train_epochs"], "train_batch_size": t["train_batch_size"],
            "best_global_step": t.get("best_global_step")}


def cell(cell_dir: Path, logs: Path) -> dict:
    bench, _, tag = cell_dir.name.partition("__")
    seeds = sorted((int(p.parents[2].name.removeprefix("seed_")), p.parent)
                   for p in cell_dir.glob("seed_*/ckpt/best/adapter_config.json"))
    ids = cell_dir / "eval_ids.json"
    rec = {"dir": _rel(cell_dir), "benchmark": bench, "model_tag": tag,
           "seeds": [s for s, _ in seeds],
           "n_eval_ids": len(json.loads(ids.read_text())["prompt_ids"]) if ids.is_file() else None,
           "filter": filter_log(logs / f"filter_{bench}_{tag}.log")}
    if not seeds:
        rec["status"] = "not run"
        # Why it never trained. A finished cell's log also holds retried errors, so
        # only a cell without an adapter keeps this line.
        rec["generation_error"] = last_error(logs / f"gen_{bench}_{tag}.log")
        return rec
    rec["status"] = "complete" if len(seeds) == N_SEEDS else "partial"
    per_seed = {str(s): seed_recipe(best) for s, best in seeds}
    shared = ("base_model", "r", "lora_alpha", "lora_dropout", "target_modules", "peft_version",
              "max_steps", "num_train_epochs", "train_batch_size")
    for key in shared:
        values = {json.dumps(r[key]) for r in per_seed.values()}
        if len(values) != 1:
            raise SystemExit(f"{cell_dir.name}: seeds disagree on {key}: {sorted(values)}")
    first = next(iter(per_seed.values()))
    rec["adapter"] = {k: first[k] for k in shared}
    spe = first["max_steps"] / first["num_train_epochs"]
    rec["adapter"]["steps_per_epoch"] = int(spe) if spe == int(spe) else spe
    # The evaluated adapter is the epoch checkpoint with the lowest validation loss
    # (scripts/train_lora.py: load_best_model_at_end on eval_loss, one eval per epoch).
    rec["best_epoch"] = {s: (r["best_global_step"] / spe if r["best_global_step"] else None)
                         for s, r in per_seed.items()}
    f = rec["filter"]
    if f and f.get("parsed"):
        acc = accumulation(f["train"], first["train_batch_size"], rec["adapter"]["steps_per_epoch"])
        rec["grad_accumulation_consistent"] = acc
        rec["effective_batch_consistent"] = [a * first["train_batch_size"] for a in acc]
    return rec


def run(cells_root: Path, logs: Path, out: Path) -> dict:
    if not cells_root.is_dir() or not logs.is_dir():
        raise SystemExit(f"cell_recipe: need {_rel(cells_root)} and {_rel(logs)} "
                         "(copies from the GPU host; see REPRODUCIBILITY.md section 7)")
    refilter = sorted(f"{p.name}: {ln.strip()}" for p in logs.glob("*.log")
                      for ln in _text(p).splitlines() if REFILTER.search(ln))
    res = {"_provenance": "generated by rebuttal/analyses/cell_recipe.py from logs/host/ "
                          "(copied from the GPU host) and results_rebuttal/*/seed_*/ckpt/best/",
           "refilter_lines": refilter,
           "cells": {p.name: cell(p, logs) for p in sorted(cells_root.iterdir()) if p.is_dir()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    for name, c in res["cells"].items():
        f = c["filter"] or {}
        print(f"  {name}: {c['status']}, {len(c['seeds'])} seed(s)"
              + (f", tau={f['tau']}, kept {f['kept']}/{f['graded']}, train {f['train']}, val {f['val']}"
                 if f.get("parsed") else "")
              + (f", accumulation {c['grad_accumulation_consistent']}"
                 if "grad_accumulation_consistent" in c and c["status"] == "complete" else ""))
    print(f"  re-filter lines in the logs: {len(refilter)}")
    print(f"wrote {_rel(out)}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells-root", type=Path, default=REPO / "results_rebuttal")
    ap.add_argument("--logs", type=Path, default=REPO / "logs" / "host")
    ap.add_argument("--out", type=Path, default=REPO / "rebuttal" / "results" / "cell_recipe.json")
    a = ap.parse_args()
    run(a.cells_root, a.logs, a.out)


if __name__ == "__main__":
    main()
