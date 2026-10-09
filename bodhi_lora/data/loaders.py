"""Convert MedQA / MedQuAD into HealthBench-format JSONL so the SAME pipeline
(generate_traces -> filter_traces -> eval) runs on them unchanged.

Rebuttal branch: adds the two source-diverse generality benchmarks.
- MedQA-USMLE: clinician-facing board vignettes, options STRIPPED (open-ended).
- MedQuAD: patient-facing NIH consumer-health Q&A.

Each output row matches the HealthBench example schema the pipeline expects:
    {"prompt_id", "prompt": [{"role":"user","content":...}], "rubrics": [...], "example_tags":[...]}

`rubrics` are synthesized from each benchmark's ground truth so filter_traces.py
(Qwen-14B grader) has a quality signal WITHOUT any code change. Every benchmark uses
the shared 8-criterion `_rubric()` (see its docstring for why a small rubric produces
a degenerate, unfilterable score distribution) with one benchmark-specific
ground-truth check: the known-correct option for exam sets (MedQA, MedMCQA), the
reference answer for QA sets (MedQuAD, MedicationQA), the physician's actual reply
for dialogue (ChatDoctor).

The eval grader stays Llama-3.1-8B (different family), so the paper's asymmetric
cross-family grading design is preserved on the new benchmarks too.

Pinning and selection (REPRO-11):
- The dataset loads at a pinned revision (DEFAULT_REVISIONS, or --revision).
- --select seeded (default) builds every row, then draws --max of them with
  random.Random(--select-seed), so the pool is a random sample of the dataset
  and not its head. --select head keeps the first --max rows in HF order, which
  is how the posted rebuttal cells were built.
- --holdout-n writes the eval holdout ids. --holdout-method seeded (default)
  draws them with random.Random(--holdout-seed). --holdout-method head takes the
  first N prompt_ids in file order, which is the rule the posted ChatDoctor and
  MedQuAD cells used (rebuttal/launch/run_cell.sh at 61dada9, lines 116-121).
- Every build writes <out stem>.manifest.json with the dataset id, revision,
  selection, row counts and the sha256 of the JSONL and of the holdout file.
- --holdout-only reads an existing --out JSONL and writes only the holdout, so
  a JSONL built earlier gets a holdout without a new download.

Usage:
    python -m bodhi_lora.data.loaders --benchmark medquad --out data/raw/medquad_s0.jsonl \
        --max 5700 --holdout-n 1000 --holdout-out data/raw/medquad_s0_holdout_ids.json
    # The posted cells (legacy head-of-file selection and holdout):
    python -m bodhi_lora.data.loaders --benchmark chatdoctor --out data/raw/chatdoctor.jsonl \
        --max 4900 --select head
    python -m bodhi_lora.data.loaders --benchmark chatdoctor --out data/raw/chatdoctor.jsonl \
        --holdout-only --holdout-n 200 --holdout-method head --holdout-out <cell>/eval_ids.json
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from bodhi_lora import manifest as rmf

# Default HF dataset ids (overridable). Chosen for stable, widely-used schemas.
DEFAULT_IDS = {
    "medqa": "GBaker/MedQA-USMLE-4-options",   # fields: question, answer, options
    "medquad": "lavita/MedQuAD",               # fields: question, answer (some empty)
    "medicationqa": "truehealth/medicationqa",  # fields: Question, Answer (+ Focus/Type)
    "medmcqa": "openlifescienceai/medmcqa",     # fields: question, opa-opd, cop
    # Real patient->doctor dialogue (HealthCareMagic via the ChatDoctor release).
    "chatdoctor": "lavita/ChatDoctor-HealthCareMagic-100k",  # fields: instruction, input, output
}

# Dataset commits, read from https://huggingface.co/api/datasets/<id> on
# 2026-10-07. Each is the HEAD commit, and each was last modified in 2023 or
# early 2024, before any rebuttal run (2026-07-23 onward). The posted cells
# loaded these repos without a revision, so they loaded these commits.
DEFAULT_REVISIONS = {
    "GBaker/MedQA-USMLE-4-options": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",      # 2023-01-24
    "lavita/MedQuAD": "84ea67f83cec9692ad254eaa02c9731b24ecfe4c",                    # 2023-12-22
    "truehealth/medicationqa": "f8ae96155ac234a54aaf2b70ca4e02d22cfe31af",           # 2023-06-12
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",         # 2024-01-04
    "lavita/ChatDoctor-HealthCareMagic-100k": "505443eac4e99ccedeffbb6f640061223d1d4bb3",  # 2023-09-09
}


def _rubric(reference_criterion, ref_tag="axis:accuracy"):
    """Multi-criterion rubric with VARIED weights, mirroring HealthBench's structure.

    Why not 3 criteria: with only a few all-or-nothing items the normalized score
    collapses onto a handful of values. Measured on MedQuAD, 82% of traces landed on
    exactly one score (8/21 = met no positive criterion, tripped no negative), so any
    threshold either kept ~18% (starves training) or ~100% (no filtering at all).
    Eight criteria at six distinct weights give a near-continuous score, which is what
    makes a percentile threshold meaningful and comparable to the paper's tau=0.4.

    ``reference_criterion`` is the benchmark-specific ground-truth check (reference
    answer for QA sets, correct option for exam sets).
    """
    return [
        reference_criterion,
        {"criterion": "The response directly addresses the specific question that was asked.",
         "points": 5, "tags": ["axis:instruction_following"]},
        {"criterion": "The response covers at least one substantive, on-topic clinical point "
                      "(mechanism, risk factor, management step, or diagnostic consideration).",
         "points": 5, "tags": ["axis:completeness"]},
        {"criterion": "The response gives concrete, actionable guidance (specific next steps, "
                      "timeframes, dosing, or thresholds) rather than only generalities.",
         "points": 4, "tags": ["axis:completeness"]},
        {"criterion": "Where information needed to answer safely is missing, the response asks "
                      "for it or explicitly states the limitation.",
         "points": 4, "tags": ["axis:context_awareness"]},
        {"criterion": "The response is clearly written and organized appropriately for its reader.",
         "points": 3, "tags": ["axis:communication_quality"]},
        {"criterion": "The response gives medically dangerous or clearly incorrect advice.",
         "points": -8, "tags": ["axis:accuracy"]},
        {"criterion": "The response offers false reassurance, or fails to recommend escalation "
                      "when the situation warrants it.",
         "points": -5, "tags": ["axis:accuracy"]},
    ]


def stable_id(benchmark, text):
    """Deterministic prompt_id from the question text (so reruns match)."""
    h = hashlib.sha1(f"{benchmark}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{benchmark}-{h}"


def _get(ex, *keys, default=""):
    """First present, non-empty key among candidates (schemas vary across mirrors)."""
    for k in keys:
        if k in ex and ex[k] not in (None, "", [], {}):
            return ex[k]
    return default


def medqa_row(ex):
    """MedQA -> open-ended clinician prompt (options stripped) + correctness rubric."""
    question = _get(ex, "question", "sent1", "query")
    answer = _get(ex, "answer", "answer_idx", "correct", "output")
    # Some mirrors store answer as an option letter; try to resolve to text.
    options = _get(ex, "options", "choices", default=None)
    if isinstance(options, dict) and isinstance(answer, str) and answer in options:
        answer = options[answer]
    if not question or not answer:
        return None
    # The preamble must not cue the behaviors we score. An earlier version said
    # "Ask for any information you would need before committing, and state your
    # reasoning", which drove base-model active inquiry to 99% and made the cell
    # measure instruction-following instead of calibration. Keep only the
    # clinician framing and the open-ended instruction.
    user = (
        "You are advising a clinician colleague. Answer the following clinical "
        "question.\n\n" + str(question).strip()
    )
    ref = str(answer).strip()[:400]
    rubrics = _rubric({
        "criterion": f"The assistant's final clinical conclusion is consistent with the "
                     f"correct answer: \"{ref}\".",
        "points": 8, "tags": ["axis:accuracy"]})
    return question, user, rubrics, ["theme:medqa", "persona:clinician"]


def medquad_row(ex):
    """MedQuAD -> patient-facing consumer-health prompt + reference-coverage rubric."""
    question = _get(ex, "question", "Question", "query")
    answer = _get(ex, "answer", "Answer", "long_answer")
    if not question or not answer:   # ~31k MedQuAD answers were removed for licensing
        return None
    user = str(question).strip()
    ref = str(answer).strip().replace("\n", " ")[:800]
    rubrics = _rubric({
        "criterion": f"The response does not contradict this reference answer: \"{ref}\".",
        "points": 8, "tags": ["axis:accuracy"]})
    return question, user, rubrics, ["theme:medquad", "persona:patient"]


def medicationqa_row(ex):
    """MedicationQA -> real consumer drug questions + reference-coverage rubric.

    These are genuine questions people asked about medications, and they very often
    omit dose, age, indication, or concurrent meds — so they EXERCISE context-seeking
    (the behavior the paper targets) rather than merely rewarding recall.
    """
    question = _get(ex, "Question", "question")
    answer = _get(ex, "Answer", "answer")
    if not question or not answer:
        return None
    ref = str(answer).strip().replace("\n", " ")[:800]
    rubrics = _rubric({
        "criterion": f"The response does not contradict this reference answer: \"{ref}\".",
        "points": 8, "tags": ["axis:accuracy"]})
    return question, str(question).strip(), rubrics, ["theme:medicationqa", "persona:patient"]


def medmcqa_row(ex):
    """MedMCQA -> open-ended clinician exam question (options stripped) + correctness rubric."""
    question = _get(ex, "question")
    opts = [_get(ex, k) for k in ("opa", "opb", "opc", "opd")]
    cop = ex.get("cop")
    if not question or cop is None or not any(opts):
        return None
    try:
        answer = opts[int(cop)]
    except (ValueError, TypeError, IndexError):
        return None
    if not answer:
        return None
    # See medqa_row: no clause here may cue asking or reasoning aloud.
    user = ("You are advising a clinician colleague. Answer the following clinical "
            "question.\n\n" + str(question).strip())
    ref = str(answer).strip()[:300]
    rubrics = _rubric({
        "criterion": f"The assistant's final conclusion is consistent with the correct "
                     f"answer: \"{ref}\".",
        "points": 8, "tags": ["axis:accuracy"]})
    return question, user, rubrics, ["theme:medmcqa", "persona:clinician"]


def chatdoctor_row(ex):
    """ChatDoctor / HealthCareMagic -> REAL patient->doctor queries + doctor-answer rubric.

    A different benchmark FAMILY from MedQA (complete-info exam vignettes) and MedQuAD
    (curated NIH reference articles): these are unedited questions real people asked a
    doctor online, so they routinely omit age, duration, medications, and associated
    symptoms. That makes them the sharpest test of context-seeking — the behavior the
    adapter targets — and of whether asking is TARGETED rather than indiscriminate.

    Reference answer = the real physician's reply, used as the filter's quality signal.
    """
    question = _get(ex, "input", "Patient", "question")
    answer = _get(ex, "output", "Doctor", "answer")
    if not question or not answer:
        return None
    q = str(question).strip()
    ref = str(answer).strip().replace("\n", " ")[:800]
    rubrics = _rubric({
        "criterion": f"The response does not contradict the physician's actual reply: \"{ref}\".",
        "points": 8, "tags": ["axis:accuracy"]})
    return q, q, rubrics, ["theme:chatdoctor", "persona:patient"]


BUILDERS = {
    "chatdoctor": chatdoctor_row,
    "medqa": medqa_row,
    "medquad": medquad_row,
    "medicationqa": medicationqa_row,
    "medmcqa": medmcqa_row,
}


def build_rows(ds, benchmark, max_rows=None, select="seeded", select_seed=0):
    """Rows in HealthBench format, plus counts.

    select="head" stops at the first max_rows usable rows in dataset order (the
    posted cells). select="seeded" builds every usable row and draws max_rows of
    them with random.Random(select_seed); the result is sorted by prompt_id so
    the file does not depend on the dataset order.
    """
    build = BUILDERS[benchmark]
    rows, seen = [], set()
    counts = {"n_source_rows": 0, "n_unusable": 0, "n_duplicates": 0}
    for ex in ds:
        counts["n_source_rows"] += 1
        built = build(ex)
        if built is None:
            counts["n_unusable"] += 1
            continue
        question, user, rubrics, tags = built
        pid = stable_id(benchmark, question)
        if pid in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(pid)
        rows.append({
            "prompt_id": pid,
            "prompt": [{"role": "user", "content": user}],
            "rubrics": rubrics,
            "example_tags": tags,
            "_source": benchmark,
        })
        if select == "head" and max_rows and len(rows) >= max_rows:
            break
    if select == "seeded":
        rows.sort(key=lambda r: r["prompt_id"])
        if max_rows and len(rows) > max_rows:
            rows = sorted(random.Random(select_seed).sample(rows, max_rows),
                          key=lambda r: r["prompt_id"])
    counts["n_written"] = len(rows)
    return rows, counts


def make_holdout(ids, n, method="seeded", seed=0):
    """Eval holdout ids.

    method="head" takes the first n ids in file order, the rule of the posted
    cells. method="seeded" draws n with random.Random(seed) and sorts them.
    """
    if n > len(ids):
        raise SystemExit(f"holdout of {n} needs at least {n} rows; have {len(ids)}")
    if method == "head":
        return list(ids[:n])
    return sorted(random.Random(seed).sample(list(ids), n))


def write_holdout(jsonl_path, n, method, seed, out_path):
    with open(jsonl_path) as f:
        ids = [json.loads(line)["prompt_id"] for line in f if line.strip()]
    held = make_holdout(ids, n, method, seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "description": f"eval holdout: {n} of {len(ids)} rows, method={method}"
                           + (f", seed={seed}" if method == "seeded" else ""),
            "method": method,
            "seed": seed if method == "seeded" else None,
            "n": len(held),
            "source": str(jsonl_path),
            "source_sha256": rmf.sha256_file(jsonl_path),
            "prompt_ids": held,
        }, f, indent=2)
    print(f"held out {len(held)} eval ids ({method}) -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BUILDERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-id", default=None, help="override HF dataset id")
    ap.add_argument("--revision", default=None,
                    help="dataset commit. Default: DEFAULT_REVISIONS for the dataset id. "
                         "Pass 'main' to load HEAD.")
    ap.add_argument("--split", default="train", help="HF split to pull from")
    ap.add_argument("--max", type=int, default=None, help="cap number of rows")
    ap.add_argument("--select", choices=("seeded", "head"), default="seeded",
                    help="seeded: random --max rows of the whole dataset (default). "
                         "head: first --max rows in HF order (the posted cells).")
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--holdout-n", type=int, default=0,
                    help="write this many eval holdout ids (0 = no holdout file)")
    ap.add_argument("--holdout-method", choices=("seeded", "head"), default="seeded")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--holdout-out", default=None,
                    help="holdout ids file. Default: <out stem>_holdout_ids.json")
    ap.add_argument("--holdout-only", action="store_true",
                    help="read the existing --out JSONL and write only the holdout file")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    holdout_out = Path(args.holdout_out or out.with_name(out.stem + "_holdout_ids.json"))

    if args.holdout_only:
        if not out.is_file() or not args.holdout_n:
            raise SystemExit("--holdout-only needs an existing --out and --holdout-n > 0")
        write_holdout(out, args.holdout_n, args.holdout_method, args.holdout_seed, holdout_out)
        return

    from datasets import load_dataset  # lazy: only needed on the box
    ds_id = args.dataset_id or DEFAULT_IDS[args.benchmark]
    revision = args.revision or DEFAULT_REVISIONS.get(ds_id)
    print(f"Loading {ds_id} [{args.split}] at revision {revision or 'HEAD (unpinned)'} ...")
    ds = load_dataset(ds_id, split=args.split, revision=revision)

    rows, counts = build_rows(ds, args.benchmark, args.max, args.select, args.select_seed)
    if not rows:
        raise SystemExit(
            f"No rows written. Inspect the {ds_id} schema and adjust the field names "
            f"in {args.benchmark}_row() (the mirror may use different keys)."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows ({args.select}) -> {out}")

    outputs = [out]
    if args.holdout_n:
        outputs.append(write_holdout(out, args.holdout_n, args.holdout_method,
                                     args.holdout_seed, holdout_out))
    rmf.write_manifest(rmf.manifest_path_for(out), rmf.build_manifest(
        "build_benchmark",
        config=vars(args),
        holdout_ids=holdout_out if args.holdout_n else None,
        outputs=outputs,
        counts=counts,
        extra={"dataset": {"id": ds_id, "revision": revision,
                           "revision_source": ("flag" if args.revision else
                                               "pinned" if revision else "unpinned"),
                           "split": args.split}},
    ))


if __name__ == "__main__":
    main()
