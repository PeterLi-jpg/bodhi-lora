"""Rebuttal analysis: what drives the LoRA+BODHI red-flag drop? (Reviewer LxMF W4)

SUPERSEDED on 2026-10-10. Kept only as the record of how the posted values were made;
no paper number reads its output. Its nine-marker rule counts two free words,
DIFFERENTIAL and SCOPE OF/BOUNDAR, that are not headers of the BODHI format, so it reads
meaning from the text (docs/CONVENTIONS.md rule 15). The posted 54.6% and the numbers
split by it are withdrawn (rebuttal/withdrawals.md W25). Its seven-header rule
(PASS1_RE) is the analysis-format leak rule of configs/protocols/bodhi.yaml, which the
uniform analysis applies to every cell (python -m bodhi_lora analyze).

LxMF: "The claim that CoT and LoRA compete for the same behavioral capacity is
speculative. The drop in red-flag identification could be caused by context window
truncation or attention dilution from long outputs, because this combination
generates long outputs."

We test LxMF's alternative directly on the committed eval outputs. The paper's
mechanism (Figure interference) is that under LoRA+BODHI the model emits the BODHI
Pass-1 INTERNAL ANALYSIS format ("RED FLAGS: None", "KEY UNCERTAINTIES", ...) instead
of a patient-facing answer. If that is the cause, then:

  (a) red-flag identification should be depressed specifically on the responses that
      LEAK the Pass-1 analysis format, and
  (b) LoRA+BODHI responses that do NOT leak (clean patient-facing) should retain a
      red-flag rate close to LoRA-alone, and
  (c) the drop should NOT be explained by length/truncation: if it were attention
      dilution from long outputs, red-flag would fall with length regardless of format.

Runs against the submitted cell's grade files (--cell-dir). No inference, no H100.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rebuttal.analyses.cell_files import grade_path  # noqa: E402

# The submitted cell (results/PROVENANCE.md).
CELL = REPO / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05"
SEEDS = (7, 13, 42, 99, 101)
CONFIG_DISPLAY = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Wrapper",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+Wrapper",
}
CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]

# Analysis-format markers. The first seven are the seven Pass-1 section headers of
# bodhi-llm 0.1.4 (bodhi/prompts.py, render_analysis_prompt). The last two are not
# Pass-1 headers: "differential" appears in the Pass-2 instructions, and both appear in
# the SOAP-note drafts that leaked responses often continue into. The posted 54.6%
# used all nine. PASS1_RE gives the same rule with the seven headers only (audit
# NUM-4), reported next to it.
LEAK_MARKERS = [
    r"TASK TYPE",
    r"KEY UNCERTAINT",
    r"SAFE RECOMMENDATION",
    r"RED FLAGS?\s*:",
    r"WHAT I THINK",
    r"QUESTIONS TO ASK",
    r"AUDIENCE\s*:",
    r"DIFFERENTIAL",
    r"SCOPE (OF|BOUNDAR)",
]
LEAK_RE = [re.compile(m, re.IGNORECASE) for m in LEAK_MARKERS]
PASS1_RE = LEAK_RE[:7]


def count_leak_markers(text, patterns=LEAK_RE):
    return sum(1 for r in patterns if r.search(text or ""))


def is_leaked(text, threshold=2, patterns=LEAK_RE):
    """A response 'leaks' the analysis format if >= threshold markers appear
    (a patient-facing answer has ~0)."""
    return count_leak_markers(text, patterns) >= threshold


def load(config_name, cell_dir=CELL):
    """Pool a config's per-response rows across the 5 seeds.

    Only responses with a parsed epistemic grade are kept, because the red-flag
    split needs the grade. So leaked_rate is over graded responses: 507/928 =
    54.6% for LoRA+CoT, the posted figure. Over every LoRA+CoT completion in the
    HealthBench result files the same rule gives 532/959 = 55.5%.
    """
    rows = []
    for seed in SEEDS:
        # grade_path stops on a missing seed, so a moved file cannot shrink the pool.
        data = json.load(open(grade_path(Path(cell_dir), seed)))
        for cfg in data["configs"]:
            if cfg["name"] != config_name:
                continue
            for ex in cfg["examples"]:
                if ex.get("parse_failure"):
                    continue
                sc = ex.get("scores") or {}
                rf = sc.get("red_flag_identification")
                if rf is None:
                    continue
                resp = ex.get("response") or ""
                rows.append({
                    "prompt_id": ex["prompt_id"],
                    "red_flag": rf,
                    "len": len(resp),
                    "leaked": is_leaked(resp),
                    "leaked_pass1": is_leaked(resp, patterns=PASS1_RE),
                    "n_markers": count_leak_markers(resp),
                })
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", default=str(CELL))
    ap.add_argument("--out", default=str(CELL / "analysis" / "interference_mechanism.json"))
    args = ap.parse_args()

    data = {c: load(c, args.cell_dir) for c in CONFIG_ORDER}
    out = {"configs": {}}

    print("=" * 82)
    print("Pass-1 analysis-format LEAKAGE and red-flag (0-2), by config (pooled 5 seeds)")
    print("=" * 82)
    hdr = f"{'Config':14s} {'n':>5s} {'leaked%':>8s} {'redflag_all':>12s} {'len_med':>8s}"
    print(hdr); print("-" * len(hdr))
    for c in CONFIG_ORDER:
        rows = data[c]
        n = len(rows)
        leak = mean([1.0 if r["leaked"] else 0.0 for r in rows])
        rf = mean([r["red_flag"] for r in rows])
        lens = sorted(r["len"] for r in rows)
        med = lens[len(lens)//2] if lens else 0
        print(f"{CONFIG_DISPLAY[c]:14s} {n:5d} {100*leak:7.1f}% {rf:12.3f} {med:8d}")
        out["configs"][c] = {"n": n, "leaked_rate": leak, "red_flag_mean": rf, "len_median": med,
                             "leaked_rate_pass1_headers": mean([1.0 if r["leaked_pass1"] else 0.0
                                                                for r in rows])}
    print()

    # --- The decisive cut: within LoRA+Wrapper, split leaked vs non-leaked ---
    lb = data["lora_bodhi"]
    leaked = [r for r in lb if r["leaked"]]
    clean = [r for r in lb if not r["leaked"]]
    lora = data["lora_no_wrapper"]
    print("=" * 82)
    print("Decisive test: LoRA+Wrapper split by whether the response leaked the analysis format")
    print("=" * 82)
    print(f"  LoRA (no wrapper) red-flag:                 {mean([r['red_flag'] for r in lora]):.3f}  (n={len(lora)})")
    print(f"  LoRA+Wrapper, NON-leaked (patient-facing):  {mean([r['red_flag'] for r in clean]):.3f}  (n={len(clean)})")
    print(f"  LoRA+Wrapper, LEAKED (Pass-1 analysis):     {mean([r['red_flag'] for r in leaked]):.3f}  (n={len(leaked)})")
    print()
    print("  Response length (median chars):")
    def med_len(rs):
        L = sorted(r["len"] for r in rs); return L[len(L)//2] if L else 0
    print(f"    non-leaked = {med_len(clean)}   leaked = {med_len(leaked)}")
    out["decisive"] = {
        "lora_red_flag": mean([r["red_flag"] for r in lora]),
        "lora_bodhi_clean_red_flag": mean([r["red_flag"] for r in clean]),
        "lora_bodhi_leaked_red_flag": mean([r["red_flag"] for r in leaked]),
        "n_clean": len(clean), "n_leaked": len(leaked),
        "len_median_clean": med_len(clean), "len_median_leaked": med_len(leaked),
    }
    # The same split with the seven Pass-1 headers only.
    leaked1 = [r for r in lb if r["leaked_pass1"]]
    clean1 = [r for r in lb if not r["leaked_pass1"]]
    out["decisive_pass1_headers"] = {
        "lora_bodhi_clean_red_flag": mean([r["red_flag"] for r in clean1]),
        "lora_bodhi_leaked_red_flag": mean([r["red_flag"] for r in leaked1]),
        "n_clean": len(clean1), "n_leaked": len(leaked1),
    }
    print(f"  Pass-1 headers only: leaked {out['decisive_pass1_headers']['lora_bodhi_leaked_red_flag']:.3f} "
          f"(n={len(leaked1)}), non-leaked {out['decisive_pass1_headers']['lora_bodhi_clean_red_flag']:.3f} "
          f"(n={len(clean1)})")
    print()

    # --- Length control: does red-flag fall with length WITHIN non-leaked? ---
    # If LxMF's attention-dilution story held, red-flag should drop with length even
    # among clean responses. Bin clean LoRA+Wrapper responses by length quartile.
    print("=" * 82)
    print("Length control: red-flag by length quartile, NON-leaked LoRA+Wrapper only")
    print("=" * 82)
    if clean:
        cl = sorted(clean, key=lambda r: r["len"])
        q = len(cl) // 4 or 1
        quarts = [cl[:q], cl[q:2*q], cl[2*q:3*q], cl[3*q:]]
        qout = []
        for i, grp in enumerate(quarts, 1):
            rf = mean([r["red_flag"] for r in grp])
            print(f"  Q{i} (len~{med_len(grp)}): red-flag {rf:.3f}  (n={len(grp)})")
            qout.append({"quartile": i, "len_median": med_len(grp), "red_flag": rf, "n": len(grp)})
        out["length_control_clean"] = qout
    print()

    print("=" * 82)
    print("Interpretation")
    print("=" * 82)
    c_rf = mean([r['red_flag'] for r in clean])
    l_rf = mean([r['red_flag'] for r in leaked])
    lora_rf = mean([r['red_flag'] for r in lora])
    if leaked and clean and l_rf < c_rf - 0.2 and abs(c_rf - lora_rf) < 0.25:
        print("  Red-flag collapses on the LEAKED (Pass-1-format) responses and stays ~LoRA-level")
        print("  on the NON-leaked ones. The drop tracks FORMAT LEAKAGE, not length/truncation.")
        print("  -> fits the format-leakage account (post hoc); attention dilution does not explain it.")
    else:
        print("  Pattern is weaker than expected on this cut; report the numbers as-is.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
