"""Physician validation: Cohen's kappa between physician grades and the LLM graders.

Each returned rater file is a Grading-*.xlsx built by scripts/make_irr_excel.py: one
row per (response, rubric criterion), with the rater's pass / fail / unsure in
``physician_grade``. The rater files carry personal names in their file names, so
they stay outside the repo. Pass their paths on the command line; the script reads
them and never copies them. The output names raters only as rater_A, rater_B, ...
in the order given, and records no file names.

Approach
--------
1. Physician consensus per criterion (grading_id):
   - one rater: that rater's grade (the preliminary single-physician analysis);
   - two or more raters: the majority grade among those who graded the row
     (needs at least 2 votes); a tie goes to the adjudicator file if one is given.
   ``unsure`` consensus rows are excluded from kappa.
2. LLM verdict per criterion, from the stored grader output:
   - eval_llama rows: results_modal/seed_<N>/<config>.json, results[].criteria_results;
   - filter_qwen rows: data/sft_qwen_graded/seed_7_train.jsonl, grade.criteria_results.
   The physician's ``pass`` means "the response did the right thing". For a positive
   criterion that is criteria_met = True. For a negative (avoid) criterion it is
   criteria_met = False: the response avoided the harm. The first version of this
   script mapped criteria_met = True to pass for both kinds, which flips every
   negative row and gives kappa 0.19 instead of the reported 0.35.
3. Binary Cohen's kappa (pass vs fail) for all rows, positive rows, negative rows and
   each eval condition, with a 95% CI from a bootstrap that resamples prompt ids
   (rows from one prompt share a response and a rater sitting, so they are not
   independent; the eval rows come from 10 prompts).
4. With two or more raters, pairwise inter-physician kappa on rows both graded.

Output: results_modal/irr/kappa_results.json (de-identified).

Usage:
    python3 scripts/compute_irr_kappa.py --rater-files /path/to/Grading-<rater>.xlsx [...]
"""
from __future__ import annotations

import argparse
import json
import string
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IRR_DIR = ROOT / "results_modal" / "irr"
GRADES = ("pass", "fail", "unsure")
CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0


def load_rater(path: Path, label: str) -> pd.DataFrame:
    """grading_id, prompt_id, grade (lower-case; '' when blank)."""
    df = pd.read_excel(path, sheet_name="Grading")
    df["grade"] = df["physician_grade"].astype(str).str.strip().str.lower().replace({"nan": ""})
    if not df["grade"].isin(GRADES).any():
        raise SystemExit(f"compute_irr_kappa: {label} has no pass/fail/unsure grades; "
                         "this looks like a blank template, not a returned file")
    bad = sorted(set(df["grade"]) - set(GRADES) - {""})
    if bad:
        raise SystemExit(f"compute_irr_kappa: {label} has unknown grades {bad}")
    df["grading_id"] = df["grading_id"].astype(str)
    return df[["grading_id", "prompt_id", "grade"]]


def load_adjudicator(path: Path) -> dict[str, str]:
    df = pd.read_excel(path, sheet_name="Tiebreaker")
    out = {}
    for r in df.itertuples():
        v = str(getattr(r, "tiebreaker_grade", "")).strip().lower()
        if v in ("pass", "fail"):
            out[str(r.grading_id)] = v
    return out


def derive_consensus(rater_grades: list[dict[str, str]], adj: dict[str, str]) -> dict[str, str]:
    """grading_id -> consensus grade (pass / fail / unsure)."""
    min_votes = 1 if len(rater_grades) == 1 else 2
    consensus = {}
    for gid in sorted(set().union(*rater_grades)):
        votes = [g[gid] for g in rater_grades if g.get(gid) in GRADES]
        if len(votes) < min_votes:
            continue
        counts = pd.Series(votes).value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            consensus[gid] = top[0]
        elif adj.get(gid) in ("pass", "fail"):
            consensus[gid] = adj[gid]
        # else: disputed with no adjudicator vote; left out
    return consensus


def cohen_kappa_from_counts(cm: np.ndarray) -> float:
    """Kappa from a 2x2 table [[pass/pass, pass/fail], [fail/pass, fail/fail]].

    Works on the last two axes so a stack of bootstrap tables is one call.
    """
    n = cm.sum(axis=(-2, -1))
    po = (cm[..., 0, 0] + cm[..., 1, 1]) / n
    rows, cols = cm.sum(axis=-1), cm.sum(axis=-2)
    pe = (rows[..., 0] * cols[..., 0] + rows[..., 1] * cols[..., 1]) / n ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def _table(a: list[str], b: list[str]) -> np.ndarray:
    cm = np.zeros((2, 2))
    for x, y in zip(a, b):
        cm[0 if x == "pass" else 1, 0 if y == "pass" else 1] += 1
    return cm


def kappa_block(sub: pd.DataFrame, rng_seed: int = BOOTSTRAP_SEED) -> dict:
    """Kappa, agreement, pass rates and a prompt-clustered bootstrap CI.

    ``sub`` has columns prompt_id, consensus, llm (both pass/fail).
    """
    if sub.empty:
        return {"kappa": None, "ci95": [None, None], "n": 0, "n_prompts": 0,
                "agree": None, "physician_pass": None, "llm_pass": None}
    cm = _table(sub["consensus"].tolist(), sub["llm"].tolist())
    prompts = sorted(sub["prompt_id"].unique())
    # One 2x2 table per prompt; a resample is a weighted sum of these tables.
    per_prompt = np.stack([_table(g["consensus"].tolist(), g["llm"].tolist())
                           for _, g in sub.groupby("prompt_id", sort=True)])
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(prompts), size=(BOOTSTRAP_RESAMPLES, len(prompts)))
    weights = np.zeros((BOOTSTRAP_RESAMPLES, len(prompts)))
    np.add.at(weights, (np.repeat(np.arange(BOOTSTRAP_RESAMPLES), len(prompts)), draws.ravel()), 1.0)
    boot = cohen_kappa_from_counts(np.einsum("bp,pij->bij", weights, per_prompt))
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    n = int(cm.sum())
    return {
        "kappa": float(cohen_kappa_from_counts(cm)),
        "ci95": [float(lo), float(hi)],
        "n": n,
        "n_prompts": len(prompts),
        "agree": float((cm[0, 0] + cm[1, 1]) / n),
        "physician_pass": float(cm[0].sum() / n),
        "llm_pass": float(cm[:, 0].sum() / n),
    }


def llm_verdicts(rows: pd.DataFrame, results_dir: Path, filter_traces: Path | None) -> pd.Series:
    """grading_id -> LLM pass/fail; NaN where the stored grade is not available."""
    eval_cache: dict[tuple, dict] = {}
    filter_cache: dict[str, dict] | None = None
    if filter_traces is not None and filter_traces.is_file():
        filter_cache = {}
        with open(filter_traces) as f:
            for line in f:
                d = json.loads(line)
                filter_cache[d["prompt_id"]] = d
    out = {}
    for r in rows.itertuples():
        if r.source == "eval_llama":
            key = (int(r.seed), r.config)
            if key not in eval_cache:
                with open(results_dir / f"seed_{key[0]}" / f"{key[1]}.json") as f:
                    eval_cache[key] = {x["prompt_id"]: x for x in json.load(f)["results"]}
            criteria = eval_cache[key][r.prompt_id]["criteria_results"]
        elif filter_cache is not None:
            criteria = filter_cache[r.prompt_id]["grade"]["criteria_results"]
        else:
            continue
        c = criteria[int(r.criterion_number) - 1]
        negative = c["points"] < 0
        if negative != (str(r.is_avoid_item).strip().upper() == "YES"):
            raise SystemExit(f"compute_irr_kappa: {r.grading_id} avoid flag disagrees with rubric sign")
        if c["criterion"].strip() != str(r.criterion_text).strip():
            raise SystemExit(f"compute_irr_kappa: {r.grading_id} criterion text does not match the grader output")
        # pass = the response did the right thing (see module docstring).
        out[r.grading_id] = "pass" if (c["criteria_met"] != negative) else "fail"
    return pd.Series(out, dtype=object)


def run(rater_files: list[Path], answer_key: Path, template: Path, results_dir: Path,
        filter_traces: Path | None, adjudicator: Path | None) -> dict:
    labels = [f"rater_{string.ascii_uppercase[i]}" for i in range(len(rater_files))]
    raters = [load_rater(p, lab) for p, lab in zip(rater_files, labels)]
    # The template is the sheet the grader output was matched against. The copy a
    # rater received may reword criterion text (19 rows say "mark pass" where the
    # rubric says "criteria_met = true"), so text checks use the template.
    sheet = pd.read_excel(template, sheet_name="Grading")[["grading_id", "prompt_id", "criterion_text"]]
    sheet["grading_id"] = sheet["grading_id"].astype(str)
    ref = sheet.set_index("grading_id")["prompt_id"]
    for lab, r in zip(labels, raters):
        if not r.set_index("grading_id")["prompt_id"].equals(ref):
            raise SystemExit(f"compute_irr_kappa: {lab} rows do not match the template")
    grades = [dict(zip(r["grading_id"], r["grade"])) for r in raters]
    adj = load_adjudicator(adjudicator) if adjudicator else {}
    consensus = derive_consensus(grades, adj)

    ak = pd.read_excel(answer_key)
    ak["grading_id"] = ak["grading_id"].astype(str)
    rows = ak.merge(sheet, on="grading_id", how="left", validate="one_to_one")
    rows["consensus"] = rows["grading_id"].map(consensus)
    rows["llm"] = rows["grading_id"].map(llm_verdicts(rows, results_dir, filter_traces))
    rows["negative"] = rows["is_avoid_item"].astype(str).str.strip().str.upper().eq("YES")
    usable = rows[rows["consensus"].isin(["pass", "fail"]) & rows["llm"].isin(["pass", "fail"])]

    out = {
        "n_raters": len(raters),
        "raters": labels,
        "consensus_rule": ("single rater: that rater's grade" if len(raters) == 1 else
                           "majority of >=2 votes; ties to the adjudicator"),
        "llm_pass_rule": "positive criterion: criteria_met; negative (avoid) criterion: not criteria_met",
        "kappa": "binary Cohen's kappa, pass vs fail; unsure excluded",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "cluster": "prompt_id", "interval": "percentile"},
        "rows": {"total": int(len(rows)),
                 **{src: int((rows["source"] == src).sum()) for src in ("eval_llama", "filter_qwen")}},
        "grades": {lab: {g: int(sum(v == g for v in d.values())) for g in GRADES}
                   for lab, d in zip(labels, grades)},
        "unsure_consensus": {src: int(((rows["source"] == src) & (rows["consensus"] == "unsure")).sum())
                             for src in ("eval_llama", "filter_qwen")},
        "llm_vs_physician": {},
    }
    for src in ("eval_llama", "filter_qwen"):
        sub = usable[usable["source"] == src]
        if sub.empty:
            print(f"SKIP {src}: no rows with both a physician and an LLM grade")
            continue
        block = {"all": kappa_block(sub), "positive": kappa_block(sub[~sub["negative"]]),
                 "negative": kappa_block(sub[sub["negative"]])}
        if src == "eval_llama":
            block["by_config"] = {c: kappa_block(sub[sub["config"] == c]) for c in CONFIGS}
        out["llm_vs_physician"][src] = block

    pairs = []
    for (la, ga), (lb, gb) in combinations(zip(labels, grades), 2):
        common = sorted(g for g in set(ga) & set(gb) if ga[g] in ("pass", "fail") and gb[g] in ("pass", "fail"))
        if len(common) >= 2:
            cm = _table([ga[g] for g in common], [gb[g] for g in common])
            pairs.append({"pair": f"{la}-{lb}", "kappa": float(cohen_kappa_from_counts(cm)), "n": len(common)})
    out["inter_physician"] = {"pairs": pairs} if pairs else {"pairs": [], "note": "needs at least 2 raters"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rater-files", type=Path, nargs="+", required=True,
                    help="returned Grading-*.xlsx files, read in place; labelled rater_A, rater_B, ... in this order")
    ap.add_argument("--adjudicator", type=Path, default=None, help="filled tiebreaker xlsx, if any")
    ap.add_argument("--answer-key", type=Path, default=IRR_DIR / "answer_key.xlsx")
    ap.add_argument("--template", type=Path, default=IRR_DIR / "Grading-template.xlsx")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_modal")
    ap.add_argument("--filter-traces", type=Path,
                    default=ROOT / "data" / "sft_qwen_graded" / "seed_7_train.jsonl")
    ap.add_argument("--out", type=Path, default=IRR_DIR / "kappa_results.json")
    a = ap.parse_args()

    if not a.filter_traces.is_file():
        print(f"SKIP filter_qwen rows: {a.filter_traces} not found")
    out = run(a.rater_files, a.answer_key, a.template, a.results_dir, a.filter_traces, a.adjudicator)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    ev = out["llm_vs_physician"].get("eval_llama", {}).get("all")
    if ev:
        print(f"Llama vs physician: kappa {ev['kappa']:.3f} [{ev['ci95'][0]:.2f}, {ev['ci95'][1]:.2f}] "
              f"n={ev['n']} rows from {ev['n_prompts']} prompts")
    fq = out["llm_vs_physician"].get("filter_qwen", {}).get("all")
    if fq:
        print(f"Qwen vs physician:  kappa {fq['kappa']:.3f} [{fq['ci95'][0]:.2f}, {fq['ci95'][1]:.2f}] "
              f"n={fq['n']} rows from {fq['n_prompts']} prompts")
    print(f"wrote {a.out.name}")


if __name__ == "__main__":
    main()
