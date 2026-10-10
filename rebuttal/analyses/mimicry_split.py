"""Rebuttal analysis: calibration vs. template mimicry (meta-review W4, Reviewer LxMF W2).

SUPERSEDED on 2026-10-10. Kept only as the record of how the posted values were made;
no paper number and no current analysis reads its output. Two of its rules decide what
text means by keywords, which docs/CONVENTIONS.md rule 15 forbids: CLARIFY_RE labels a
prompt context-dependent when a rubric criterion contains a clarification keyword, and
CLINICIAN_RE counts prompts whose first user turn names a clinical role. The posted
discrimination index (+2.1, +11.0, +10.5 and -3.5 points), the odds ratios and the 4 of
667 clinician-posed prompts came from them; rebuttal/withdrawals.md W14 and W24 withdraw
them. The theme-only labelling here is structural, and the uniform analysis computes it
from the tag alone (label needs-info-theme, python -m bodhi_lora analyze).

LxMF's exact proposal: "isolate samples that genuinely have missing information and
cannot be answered, and compare with those that can be answered without additional
information, and observe if CoT or LoRA triggers more inquiry in the former group than
the latter."

We operationalize the split with an INDEPENDENT label (not derived from the grader's
own context_seeking score, which would be circular):

  context_dependent  = HealthBench theme == 'context_seeking'
                       OR the prompt's rubric rewards asking for clarification/context
                       (criterion text matches a clarification keyword).
  self_contained     = everything else.

Real calibration predicts the adapter raises active inquiry MORE on context_dependent
prompts and does NOT ask indiscriminately on self_contained prompts. We summarize this
with a discrimination index:

  discrimination(config) = P(ask | context_dependent) - P(ask | self_contained)

Template mimicry (asking everywhere) collapses discrimination toward Base. Internalized
calibration keeps LoRA's discrimination close to the wrapper's and well above Base.

A difference of proportions is compressed at low base rates, so each config also gets
the scale-free odds ratio of asking, context_dependent vs self_contained:

  odds_ratio(config) = [cd / (1 - cd)] / [sc / (1 - sc)]

with a CI from the same prompt-level bootstrap resamples. (The ORs quoted to LxMF,
1.16 / 1.54 / 1.56, were computed by hand from the rates; this is the scripted version.)

Runs against the submitted cell's grade files (--cell-dir). No inference, no H100.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.paper_stats import grade_path  # noqa: E402

# The submitted cell (results/PROVENANCE.md).
CELL = REPO / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05"
HARD = REPO / "data" / "raw" / "healthbench" / "hard.jsonl"
SEEDS = (7, 13, 42, 99, 101)

CONFIG_DISPLAY = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Wrapper (Base+CoT)",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+Wrapper",
}
CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]

# Keywords in a rubric CRITERION that indicate the ideal answer asks for more
# information / context before committing. Independent of the model output.
CLARIFY_RE = re.compile(
    r"\b(ask(s|ing)?|clarif\w*|follow[- ]?up|more (information|context|detail)"
    r"|additional (information|context|detail)|missing (information|context)"
    r"|request(s|ing)? (more|additional)|gather\w* (more|additional|context)"
    r"|what (information|else)|need(s|ed)? (more|additional|to know))\b",
    re.IGNORECASE,
)

# Heuristic persona detector (bonus: quantify clinician- vs patient-posed prompts,
# rebutting the premise that HealthBench is 'only patients googling').
CLINICIAN_RE = re.compile(
    r"\b(i'?m|i am|as)\s+(an?\s+)?(emergency\s+medicine\s+)?"
    r"(physician|doctor|clinician|resident|attending|nurse|np\b|physician\s+assistant"
    r"|\bpa\b|pharmacist|md\b|surgeon|cardiologist|paramedic|medical\s+student)",
    re.IGNORECASE,
)


def load_prompt_meta(path=HARD):
    """prompt_id -> {theme, clarify_rubric, clinician_posed}."""
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            pid = ex["prompt_id"]
            theme = None
            for t in ex.get("example_tags", []):
                if t.startswith("theme:"):
                    theme = t.split(":", 1)[1]
                    break
            clarify = any(
                CLARIFY_RE.search(r.get("criterion", "")) for r in ex.get("rubrics", [])
            )
            first_user = ""
            for m in ex.get("prompt", []):
                if m.get("role") == "user":
                    first_user = m.get("content", "")
                    break
            meta[pid] = {
                "theme": theme,
                "clarify_rubric": clarify,
                "clinician_posed": bool(CLINICIAN_RE.search(first_user)),
            }
    return meta


def group_of(m):
    """Primary label. context_dependent if HealthBench flags it as context-seeking
    OR the rubric rewards asking; else self_contained."""
    if m is None:
        return None
    if m["theme"] == "context_seeking" or m["clarify_rubric"]:
        return "context_dependent"
    return "self_contained"


def group_theme_only(m):
    """Robustness label: HealthBench's own context_seeking theme tag ONLY (does not
    depend on our clarification-keyword regex over rubric text)."""
    if m is None:
        return None
    return "context_dependent" if m["theme"] == "context_seeking" else "self_contained"


def load_examples(cell_dir=CELL):
    """Pool every graded response across the 5 seeds -> list of per-response rows."""
    rows = []
    for seed in SEEDS:
        # grade_path stops on a missing seed, so a moved file cannot shrink the pool.
        data = json.load(open(grade_path(Path(cell_dir), seed)))
        for cfg in data["configs"]:
            name = cfg["name"]
            for ex in cfg["examples"]:
                sc = ex.get("scores") or {}
                if ex.get("parse_failure"):
                    continue
                ai = sc.get("active_inquiry")
                if ai is None:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "config": name,
                        "prompt_id": ex["prompt_id"],
                        "active_inquiry": bool(ai),
                        "n_questions": sc.get("n_questions", 0) or 0,
                        "red_flag": sc.get("red_flag_identification"),
                    }
                )
    return rows


def _rate(bools):
    return (sum(1.0 for b in bools if b) / len(bools)) if bools else float("nan")


def odds_ratio(cd_rate, sc_rate):
    """Odds of asking on context_dependent prompts over odds on self_contained ones.
    NaN when a rate is 0 or 1 (the odds are then 0 or infinite)."""
    if not (0 < cd_rate < 1 and 0 < sc_rate < 1):
        return float("nan")
    return (cd_rate / (1 - cd_rate)) / (sc_rate / (1 - sc_rate))


def build_pid_index(rows, labelfn, meta):
    """config -> pid -> {"group":..., "ai":[bool,...]} pooling responses across seeds."""
    idx = {cfg: {} for cfg in CONFIG_ORDER}
    for r in rows:
        cfg = r["config"]
        if cfg not in idx:
            continue
        pid = r["prompt_id"]
        slot = idx[cfg].setdefault(pid, {"group": labelfn(meta.get(pid)), "ai": []})
        slot["ai"].append(r["active_inquiry"])
    return idx


def discrimination(pid_map, pids):
    """P(ask|context_dependent) - P(ask|self_contained) over the given pid list,
    pooling each pid's responses. Returns (disc, cd_rate, sc_rate)."""
    cd, sc = [], []
    for pid in pids:
        slot = pid_map.get(pid)
        if not slot:
            continue
        if slot["group"] == "context_dependent":
            cd.extend(slot["ai"])
        elif slot["group"] == "self_contained":
            sc.extend(slot["ai"])
    return _rate(cd) - _rate(sc), _rate(cd), _rate(sc)


def bootstrap_disc(idx, eval_pids, B=5000, seed=0):
    """Prompt-level bootstrap. Resample prompt_ids (paired across configs) B times;
    return per-config discrimination CI, the LoRA-Base discrimination-diff CI and the
    per-config odds-ratio CI. The ORs ride on the same resamples, so adding them did not
    change the discrimination CIs."""
    import numpy as np
    rng = np.random.default_rng(seed)
    pids = list(eval_pids)
    per_cfg = {cfg: [] for cfg in CONFIG_ORDER}
    or_cfg = {cfg: [] for cfg in CONFIG_ORDER}
    diff_lora_base = []
    for _ in range(B):
        samp = [pids[i] for i in rng.integers(0, len(pids), len(pids))]
        ds = {}
        for cfg in CONFIG_ORDER:
            d, cd, sc = discrimination(idx[cfg], samp)
            per_cfg[cfg].append(d)
            or_cfg[cfg].append(odds_ratio(cd, sc))
            ds[cfg] = d
        diff_lora_base.append(ds["lora_no_wrapper"] - ds["base_no_wrapper"])

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return ({cfg: ci(per_cfg[cfg]) for cfg in CONFIG_ORDER}, ci(diff_lora_base),
            {cfg: ci(or_cfg[cfg]) for cfg in CONFIG_ORDER})


def report(idx, eval_pids, meta, labelname, out):
    from collections import Counter
    # group sizes over unique pids present in the Base config's map
    gcount = Counter(idx["base_no_wrapper"].get(pid, {}).get("group") for pid in eval_pids)
    print("=" * 78)
    print(f"Active-inquiry rate (%) by config x group   [label: {labelname}]")
    print(f"  context_dependent={gcount.get('context_dependent',0)}  "
          f"self_contained={gcount.get('self_contained',0)}   (pooled 5 seeds)")
    print("=" * 78)
    header = (f"{'Config':22s} {'context_dep':>12s} {'self_cont':>12s} {'discrim':>10s} "
              f"{'95% CI (pp)':>18s} {'odds ratio [95% CI]':>22s}")
    print(header)
    print("-" * len(header))
    ci_cfg, diff_ci, or_ci = bootstrap_disc(idx, eval_pids)
    disc = {}
    label_out = {"groups": dict(gcount), "configs": {}}
    for cfg in CONFIG_ORDER:
        d, cd, sc = discrimination(idx[cfg], eval_pids)
        disc[cfg] = d
        lo, hi = ci_cfg[cfg]
        orr = odds_ratio(cd, sc)
        olo, ohi = or_ci[cfg]
        print(f"{CONFIG_DISPLAY[cfg]:22s} {100*cd:11.1f}% {100*sc:11.1f}% "
              f"{100*d:8.1f}pp   [{100*lo:+5.1f},{100*hi:+6.1f}]   "
              f"{orr:5.2f} [{olo:4.2f},{ohi:5.2f}]")
        label_out["configs"][cfg] = {
            "context_dependent_rate": cd, "self_contained_rate": sc,
            "discrimination": d, "discrimination_ci95": [lo, hi],
            "odds_ratio": orr, "odds_ratio_ci95": [olo, ohi],
        }
    dd = disc["lora_no_wrapper"] - disc["base_no_wrapper"]
    print("-" * len(header))
    print(f"  LoRA - Base discrimination difference: {100*dd:+.1f}pp  "
          f"95% CI [{100*diff_ci[0]:+.1f}, {100*diff_ci[1]:+.1f}]")
    verdict = "EXCLUDES zero" if (diff_ci[0] > 0 or diff_ci[1] < 0) else "contains zero"
    print(f"  -> {verdict}: ", end="")
    if diff_ci[0] > 0:
        print("adapter targets missing-info prompts MORE than base. Calibration, not mimicry.")
    else:
        print("targeting gain not distinguishable from base on this cut.")
    print()
    label_out["lora_minus_base_disc"] = dd
    label_out["lora_minus_base_disc_ci95"] = list(diff_ci)
    out[labelname] = label_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", default=str(CELL))
    ap.add_argument("--out", default=str(CELL / "analysis" / "mimicry_split.json"))
    # data/raw/*.jsonl is gitignored; pass the path when the file lives elsewhere.
    ap.add_argument("--healthbench", default=str(HARD))
    args = ap.parse_args()

    meta = load_prompt_meta(args.healthbench)
    rows = load_examples(args.cell_dir)
    eval_pids = sorted({r["prompt_id"] for r in rows})

    # ---- persona composition of the evaluated prompts (context for FzpA W2) ----
    clin = sum(1 for pid in eval_pids if meta.get(pid, {}).get("clinician_posed"))
    print("=" * 78)
    print("HealthBench-Hard evaluated prompts: persona (heuristic self-identification)")
    print("=" * 78)
    print(f"  unique evaluated prompt_ids: {len(eval_pids)}")
    print(f"  explicit clinician self-ID : {clin} ({100*clin/max(len(eval_pids),1):.1f}%)")
    print("  -> predominantly patient/consumer-framed as posed; a clinician-facing")
    print("     benchmark (e.g. MedQA open-ended) is the right generality complement.")
    print()

    out = {"n_eval_prompts": len(eval_pids), "clinician_posed_heuristic": clin}

    # Primary label (theme OR clarification-rubric) + theme-only robustness label.
    idx_primary = build_pid_index(rows, group_of, meta)
    idx_theme = build_pid_index(rows, group_theme_only, meta)
    report(idx_primary, eval_pids, meta, "primary_theme_or_rubric", out)
    report(idx_theme, eval_pids, meta, "robustness_theme_only", out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
