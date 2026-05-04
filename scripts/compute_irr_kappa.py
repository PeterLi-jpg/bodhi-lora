"""Stage 3 IRR analysis: Cohen's κ on criterion-level pass/fail grades.

Run AFTER:
  1. The 3 reviewers return Grading-<name>.xlsx in results_modal/irr/
  2. scripts/make_tiebreaker_excel.py → Grading-tiebreaker.xlsx
  3. The adjudicator returns Grading-tiebreaker-<name>.xlsx (or whatever name —
     anything matching Grading-tiebreaker*.xlsx that has a tiebreaker_grade column)

Approach
--------
For each criterion (grading_id) in the 3 reviewer files:

  1. Compute physician consensus per criterion:
     - If >=2 of 3 reviewers agree (majority): consensus = the majority grade.
     - If 3-way split (one of each pass/fail/unsure): consensus = adjudicator's grade.
     - If reviewer answers contain only `unsure` votes among the agreers
       (e.g. all three said unsure): consensus = unsure → excluded from κ.

  2. Reduce to a binary {pass, fail} for κ math (drop unsure).

  3. The LLM grader's per-criterion verdict comes from results_modal data:
     - For eval rows (Llama grader): the per-criterion `criteria_met` flag
       lives in seed_<N>/<config>.json under results[i].criteria_results[j].
     - For filter rows (Qwen grader): per-criterion `criteria_met` in
       data/sft_qwen_graded/seed_7_train.jsonl under grade.criteria_results.
     - LLM "criteria_met" is a boolean; map True → pass, False → fail.

  4. Compute κ between LLM and physician consensus separately for:
     - eval_llama source (overall + per-config)
     - filter_qwen source (overall)

  5. Inter-physician κ: average over 3 pairwise comparisons on the 1-vs-1
     subset of criteria where both reviewers gave a non-unsure grade.

Output: results_modal/irr/kappa_results.{json,md}.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IRR_DIR = ROOT / "results_modal" / "irr"
RESULTS_DIR = ROOT / "results_modal"
FILTER_PATH = ROOT / "data" / "sft_qwen_graded" / "seed_7_train.jsonl"

GRADE_VALUES = {"pass", "fail", "unsure"}
SEEDS = (7, 13, 42, 99, 101)
CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")


def find_reviewer_files() -> list[Path]:
    out: list[Path] = []
    for p in sorted(IRR_DIR.glob("Grading-*.xlsx")):
        if p.stem.lower() in ("grading-template",) or p.stem.lower().startswith("grading-tiebreaker"):
            continue
        out.append(p)
    return out


def find_tiebreaker_file() -> Path | None:
    for p in IRR_DIR.glob("Grading-tiebreaker*.xlsx"):
        # Open and check if the tiebreaker_grade col has any non-empty values —
        # this distinguishes the unfilled file we generated from the filled
        # adjudicator return.
        try:
            df = pd.read_excel(p, sheet_name="Tiebreaker")
            tb = df.get("tiebreaker_grade")
            if tb is None:
                continue
            tb_clean = tb.astype(str).str.strip().str.lower().replace({"nan": ""})
            if (tb_clean.isin({"pass", "fail"})).any():
                return p
        except Exception:
            continue
    return None


def load_reviewer(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Grading")
    df["physician_grade"] = (
        df["physician_grade"].astype(str).str.strip().str.lower().replace({"nan": ""})
    )
    return df[["grading_id", "physician_grade"]]


def derive_consensus(reviewer_grades: list[dict[str, str]],
                     adj_grades: dict[str, str]) -> dict[str, str]:
    """grading_id → consensus (one of pass/fail/unsure or absent)."""
    all_ids = set().union(*[d.keys() for d in reviewer_grades])
    consensus: dict[str, str] = {}
    for gid in all_ids:
        votes = [d.get(gid, "") for d in reviewer_grades]
        votes = [v for v in votes if v in GRADE_VALUES]
        if len(votes) < 2:
            continue
        # Majority?
        counts: dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        majority = [k for k, c in counts.items() if c == top]
        if len(majority) == 1:
            consensus[gid] = majority[0]
        else:
            # 3-way split (or 2-2 if only 2 reviewers — shouldn't happen w/ 3).
            adj = adj_grades.get(gid, "").lower().strip()
            if adj in {"pass", "fail"}:
                consensus[gid] = adj
            # else: skip — disputed but no adjudicator vote
    return consensus


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> tuple[float, int]:
    """Cohen's κ for binary {pass, fail} ratings. Drops any other label."""
    pairs = [(a, b) for a, b in zip(rater_a, rater_b)
             if a in {"pass", "fail"} and b in {"pass", "fail"}]
    n = len(pairs)
    if n < 2:
        return float("nan"), n
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    cats = ["pass", "fail"]
    cm = np.zeros((2, 2))
    for x, y in pairs:
        cm[cats.index(x), cats.index(y)] += 1
    n_total = cm.sum()
    if n_total == 0:
        return float("nan"), n
    po = (cm[0, 0] + cm[1, 1]) / n_total
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    pe = (row_marg[0] * col_marg[0] + row_marg[1] * col_marg[1]) / (n_total ** 2)
    if 1 - pe == 0:
        return float("nan"), n
    return float((po - pe) / (1 - pe)), n


def load_llm_criterion_verdicts() -> dict[str, str]:
    """grading_id → LLM grade {pass, fail}.

    'pass' if the rubric criterion's `criteria_met` is True; 'fail' otherwise.
    Built by re-deriving grading_id from (response_id, criterion_number)
    using answer_key.xlsx + the source data for each row.
    """
    ak_path = IRR_DIR / "answer_key.xlsx"
    if not ak_path.is_file():
        raise SystemExit(f"missing {ak_path}")
    ak = pd.read_excel(ak_path)
    # Map response_id → (source, seed, config, prompt_id) (one row per response).
    response_meta = (
        ak[["response_id", "source", "seed", "config", "prompt_id"]]
        .drop_duplicates("response_id")
        .set_index("response_id")
        .to_dict(orient="index")
    )

    # Cache per (seed, config) → results list.
    eval_cache: dict[tuple, list[dict]] = {}
    def _eval_results(seed: int, config: str) -> list[dict]:
        key = (seed, config)
        if key not in eval_cache:
            with open(RESULTS_DIR / f"seed_{seed}" / f"{config}.json") as f:
                eval_cache[key] = json.load(f)["results"]
        return eval_cache[key]

    # Filter cache: load entire jsonl once.
    filter_cache: dict[str, dict] = {}
    if FILTER_PATH.is_file():
        with open(FILTER_PATH) as f:
            for line in f:
                d = json.loads(line)
                filter_cache[d.get("prompt_id", "")] = d

    out: dict[str, str] = {}
    for _, row in ak.iterrows():
        gid = str(row["grading_id"])
        rid = row["response_id"]
        meta = response_meta[rid]
        cnum = int(row["criterion_number"])  # 1-indexed
        idx = cnum - 1
        criteria = []
        if meta["source"] == "eval_llama":
            seed = int(meta["seed"])
            cfg = meta["config"]
            for r in _eval_results(seed, cfg):
                if r["prompt_id"] == meta["prompt_id"]:
                    criteria = r.get("criteria_results", [])
                    break
        else:  # filter_qwen
            tr = filter_cache.get(meta["prompt_id"], {})
            criteria = tr.get("grade", {}).get("criteria_results", [])
        if 0 <= idx < len(criteria):
            met = bool(criteria[idx].get("criteria_met", False))
            out[gid] = "pass" if met else "fail"
        # else: missing rubric on the LLM side — leave unmapped, will skip in κ
    return out


def main() -> None:
    files = find_reviewer_files()
    if len(files) < 2:
        raise SystemExit(f"need ≥2 reviewer files, found {len(files)}")

    print("loading reviewers:")
    reviewer_dicts: list[dict[str, str]] = []
    for p in files:
        df = load_reviewer(p)
        d = {str(r.grading_id): r.physician_grade for r in df.itertuples()
             if r.physician_grade in GRADE_VALUES}
        print(f"  {p.name}: {len(d)} graded criteria")
        reviewer_dicts.append(d)

    adj_path = find_tiebreaker_file()
    adj_grades: dict[str, str] = {}
    if adj_path is not None:
        df = pd.read_excel(adj_path, sheet_name="Tiebreaker")
        for r in df.itertuples():
            v = str(getattr(r, "tiebreaker_grade", "")).strip().lower()
            if v in {"pass", "fail"}:
                adj_grades[str(r.grading_id)] = v
        print(f"  adjudicator ({adj_path.name}): {len(adj_grades)} resolved")
    else:
        print("  no adjudicator file yet (3-way-split rows will be skipped from κ)")

    # 1. Inter-physician κ (pairwise, average) — binary pass/fail only.
    pairwise = []
    for (i, da), (j, db) in combinations(enumerate(reviewer_dicts), 2):
        common = set(da) & set(db)
        a = [da[g] for g in common]
        b = [db[g] for g in common]
        k, n = cohen_kappa(a, b)
        pairwise.append({"pair": f"{i+1}↔{j+1}", "k": k, "n": n})

    inter = float(np.nanmean([p["k"] for p in pairwise])) if pairwise else float("nan")

    # 2. Physician consensus.
    consensus = derive_consensus(reviewer_dicts, adj_grades)
    print(f"\nphysician consensus established for {len(consensus)} criteria")

    # 3. LLM verdicts per grading_id.
    llm = load_llm_criterion_verdicts()
    print(f"LLM verdict mapped for {len(llm)} criteria")

    # 4. κ vs LLM, per source.
    ak = pd.read_excel(IRR_DIR / "answer_key.xlsx")
    ak["grading_id"] = ak["grading_id"].astype(str)
    ak["consensus"] = ak["grading_id"].map(consensus)
    ak["llm_verdict"] = ak["grading_id"].map(llm)

    by_source: dict[str, dict] = {}
    for src in ("eval_llama", "filter_qwen"):
        sub = ak[(ak["source"] == src)
                 & ak["consensus"].isin(["pass", "fail"])
                 & ak["llm_verdict"].isin(["pass", "fail"])]
        k, n = cohen_kappa(sub["consensus"].tolist(), sub["llm_verdict"].tolist())
        by_source[src] = {"k": k, "n": n}

    by_eval_config: dict[str, dict] = {}
    for cfg in CONFIGS:
        sub = ak[(ak["source"] == "eval_llama") & (ak["config"] == cfg)
                 & ak["consensus"].isin(["pass", "fail"])
                 & ak["llm_verdict"].isin(["pass", "fail"])]
        k, n = cohen_kappa(sub["consensus"].tolist(), sub["llm_verdict"].tolist())
        by_eval_config[cfg] = {"k": k, "n": n}

    out = {
        "n_total_grading_rows": int(len(ak)),
        "n_with_physician_consensus": int(ak["consensus"].notna().sum()),
        "inter_physician_kappa": {
            "pairwise": pairwise,
            "mean_k": inter,
        },
        "by_source": by_source,
        "by_eval_config": by_eval_config,
    }
    json_path = IRR_DIR / "kappa_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {json_path}")

    # Markdown summary.
    lines = [
        "# Cohen's κ — physician validation of LLM graders\n",
        f"- Total grading rows:       **{out['n_total_grading_rows']}**",
        f"- Rows with physician consensus: **{out['n_with_physician_consensus']}**",
        "- Binary pass/fail κ; `unsure` votes excluded; 3-way splits resolved by adjudicator.",
        "",
        "## Inter-physician κ (upper bound)",
        "",
        "| Pair | n | κ |",
        "|---|---|---|",
    ]
    for p in pairwise:
        lines.append(f"| {p['pair']} | {p['n']} | {p['k']:.3f} |")
    lines.append(f"| **Mean** | — | **{inter:.3f}** |")
    lines.append("")
    lines.append("## LLM grader vs physician consensus")
    lines.append("")
    lines.append("| Source | Grader | n | κ |")
    lines.append("|---|---|---|---|")
    for src, label in [("eval_llama", "Llama-3.1-8B"), ("filter_qwen", "Qwen-14B")]:
        d = by_source[src]
        lines.append(f"| {src} | {label} | {d['n']} | **{d['k']:.3f}** |")
    lines.append("")
    lines.append("## Eval-grader (Llama) κ by Stage 4 config")
    lines.append("")
    lines.append("| Config | n | κ |")
    lines.append("|---|---|---|")
    for cfg, d in by_eval_config.items():
        lines.append(f"| {cfg} | {d['n']} | {d['k']:.3f} |")
    md_path = IRR_DIR / "kappa_results.md"
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
    print()
    print("=== headline numbers ===")
    print(f"  inter-physician mean κ:  {inter:.3f}")
    print(f"  Llama vs consensus κ:    {by_source['eval_llama']['k']:.3f}  (n={by_source['eval_llama']['n']})")
    print(f"  Qwen  vs consensus κ:    {by_source['filter_qwen']['k']:.3f}  (n={by_source['filter_qwen']['n']})")


if __name__ == "__main__":
    main()
