"""Filter threshold, trace counts and LoRA recipe of each added cell, from the host logs (PROV-3).

The paper's Appendix J and Table F.2 need, per cell, the tau the filter used, how many
traces it kept, the train/val split and the recipe of the evaluated adapters. These sit
in two places in each cell folder <C> of the results root:

  <C>/logs/filter_<bench>_<tag>.log       the last filter run of the cell, copied from the
                                          GPU host (gitignored). Its lines
                                          "Kept K/G (score_field=..., threshold=T)" and
                                          "raw=R graded=G kept=K train=N val=M" give tau
                                          and the counts.
  <C>/logs/*.log                          any "re-filtering at 22nd-pct tau" line. The
                                          launcher that made the posted cells cut again at
                                          the 22nd percentile when train.jsonl of a
                                          non-HealthBench cell held under 40% of the raw
                                          traces (1262448:rebuttal/launch/run_cell.sh:160-200).
                                          The launcher overwrites the filter log on each
                                          pass, so the log is the cell's last pass.
  <C>/seed_<N>/train/{adapter_config,trainer_state}.json
                                          tracked; r, alpha, dropout and target modules,
                                          and the steps, epochs and best checkpoint.

The cells are the host's runs (host_runs.py), and the output keys each one by the run
name the host gave it.

The script checks the two against each other: the train count from the log, the
per-device batch from trainer_state.json and the optimizer steps per epoch must admit
one gradient-accumulation value (HF Trainer: steps per epoch =
ceil(ceil(train / per-device batch) / accumulation)). It writes
results/cross_cell/cell_recipe.json, which scripts/paper_stats.py reads.

Every run that wrote a folder on the host is listed, including runs that did not finish,
with a status: "complete" (5 seeds trained), "partial" (fewer seeds) or "not run" (no
adapter; the last error line of the generation log is kept). n_eval_ids is the size of
the cell's eval_ids.json. refilter_rule_met says whether the filter log's counts meet the
launcher's re-filter condition above.

A run can also stop before it writes a folder on the host. runs_without_folder lists
each such run, with the last line of its generation log and any "Killed" line in the
launcher's section for the run (the lines after its "=== cell: <bench> x <model>"
header in any log).

Usage:
    python3 rebuttal/analyses/cell_recipe.py [--results-root results] [--out <file>]
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rebuttal.analyses.host_runs import host_folders, host_runs  # noqa: E402

KEPT = re.compile(r"^Kept (\d+)/(\d+) \(score_field=(\S+), threshold=([0-9.eE+-]+)\)", re.M)
SUMMARY = re.compile(r"^raw=(\d+) graded=(\d+) kept=(\d+) train=(\d+) val=(\d+)", re.M)
PARSE = re.compile(r"parse failures: (\d+)/(\d+)")
REFILTER = re.compile(r"re-filtering at")
HEADER = re.compile(r"^=== cell: (\S+) x (\S+)", re.M)
N_SEEDS = 5
# The launcher re-filters a non-HealthBench cell when train.jsonl holds fewer than this
# percent of the raw traces, in shell integer arithmetic (1262448:rebuttal/launch/run_cell.sh:162-164).
REFILTER_RULE_PCT = 40


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
            "kept_pct": 100.0 * k2 / graded, "train_pct_of_raw": 100.0 * train / raw if raw else None,
            "rubric_parse_failures": [int(pf[-1][0]), int(pf[-1][1])] if pf else None}


def refilter_rule_met(bench: str, f: dict | None) -> bool | None:
    """Whether the launcher's re-filter condition holds for these filter-log counts."""
    if not f or not f.get("parsed"):
        return None
    return bench != "healthbench" and f["raw"] > 0 and f["train"] * 100 // f["raw"] < REFILTER_RULE_PCT


def _tag(model: str) -> str:
    # The launcher's TAG: tr '/:.' '___' (1262448:rebuttal/launch/run_cell.sh:39).
    return model.translate(str.maketrans("/:.", "___"))


def run_sections(text: str):
    """(benchmark, tag, text) for each "=== cell:" section of a launcher log."""
    heads = list(HEADER.finditer(text))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        yield m.group(1), _tag(m.group(2)), text[m.start():end]


def runs_without_folder(runs: dict, folders: set, texts: dict) -> dict:
    """Runs that started trace generation but left no folder on the host."""
    res = {}
    for name, cell_dir in runs.items():
        if name in folders:
            continue
        bench, _, tag = name.partition("__")
        p = cell_dir / "logs" / f"gen_{bench}_{tag}.log"
        lines = [ln.strip() for ln in texts[p.name].splitlines() if ln.strip()]
        killed = [f"{log}: {ln.strip()}" for log, text in texts.items()
                  for b, t, sec in run_sections(text) if (b, t) == (bench, tag)
                  for ln in sec.splitlines() if "Killed" in ln]
        res[name] = {"benchmark": bench, "model_tag": tag, "status": "no folder",
                     "gen_log": _rel(p), "gen_log_last_line": lines[-1] if lines else None,
                     "killed_lines": killed}
    return res


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


def cell(name: str, cell_dir: Path) -> dict:
    bench, _, tag = name.partition("__")
    logs = cell_dir / "logs"
    seeds = sorted((int(p.parents[1].name.removeprefix("seed_")), p.parent)
                   for p in cell_dir.glob("seed_*/train/adapter_config.json"))
    ids = cell_dir / "shared" / "inputs" / "eval_ids.json"
    rec = {"dir": _rel(cell_dir), "benchmark": bench, "model_tag": tag,
           "seeds": [s for s, _ in seeds],
           "n_eval_ids": len(json.loads(ids.read_text())["prompt_ids"]) if ids.is_file() else None,
           "filter": filter_log(logs / f"filter_{bench}_{tag}.log")}
    rec["refilter_rule_met"] = refilter_rule_met(bench, rec["filter"])
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
            raise SystemExit(f"{name}: seeds disagree on {key}: {sorted(values)}")
    first = next(iter(per_seed.values()))
    rec["adapter"] = {k: first[k] for k in shared}
    spe = first["max_steps"] / first["num_train_epochs"]
    rec["adapter"]["steps_per_epoch"] = int(spe) if spe == int(spe) else spe
    # The evaluated adapter is the epoch checkpoint with the lowest validation loss
    # (bodhi_lora/pipeline/train.py: load_best_model_at_end on eval_loss, one eval per epoch).
    rec["best_epoch"] = {s: (r["best_global_step"] / spe if r["best_global_step"] else None)
                         for s, r in per_seed.items()}
    f = rec["filter"]
    if f and f.get("parsed"):
        acc = accumulation(f["train"], first["train_batch_size"], rec["adapter"]["steps_per_epoch"])
        rec["grad_accumulation_consistent"] = acc
        rec["effective_batch_consistent"] = [a * first["train_batch_size"] for a in acc]
    return rec


def run(results_root: Path, out: Path) -> dict:
    runs = host_runs(results_root)
    if not runs:
        raise SystemExit(f"cell_recipe: no host logs under {_rel(results_root)}/*/logs/ "
                         "(copies from the GPU host; see REPRODUCIBILITY.md section 7)")
    folders = host_folders(results_root)
    # Log names are unique across cells: each host log went to the one cell it belongs to.
    texts = {p.name: _text(p) for d in runs.values() for p in sorted((d / "logs").glob("*.log"))}
    refilter = sorted(f"{name}: {ln.strip()}" for name, text in texts.items()
                      for ln in text.splitlines() if REFILTER.search(ln))
    res = {"_provenance": "generated by rebuttal/analyses/cell_recipe.py from the host logs in "
                          "results/*/logs/ (copied from the GPU host) and results/*/seed_*/train/",
           "refilter_lines": refilter,
           "cells": {name: cell(name, d) for name, d in runs.items() if name in folders},
           "runs_without_folder": runs_without_folder(runs, folders, texts)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    for name, c in res["cells"].items():
        f = c["filter"] or {}
        print(f"  {name}: {c['status']}, {len(c['seeds'])} seed(s)"
              + (f", tau={f['tau']}, kept {f['kept']}/{f['graded']}, train {f['train']}, val {f['val']}"
                 if f.get("parsed") else "")
              + (f", accumulation {c['grad_accumulation_consistent']}"
                 if "grad_accumulation_consistent" in c and c["status"] == "complete" else ""))
    for name, r in res["runs_without_folder"].items():
        print(f"  {name}: {r['status']}, gen log ends {r['gen_log_last_line']!r}, "
              f"{len(r['killed_lines'])} Killed line(s)")
    met = [n for n, c in res["cells"].items() if c["refilter_rule_met"]]
    print(f"  re-filter lines in the logs: {len(refilter)}; re-filter condition met by: {met or 'none'}")
    print(f"wrote {_rel(out)}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "cross_cell" / "cell_recipe.json")
    a = ap.parse_args()
    run(a.results_root, a.out)


if __name__ == "__main__":
    main()
