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

Usage:
    python scripts/build_benchmark_jsonl.py --benchmark medqa   --out data/raw/medqa_open.jsonl --max 4200
    python scripts/build_benchmark_jsonl.py --benchmark medquad --out data/raw/medquad.jsonl    --max 4200
"""

import argparse
import hashlib
import json
from pathlib import Path

# Default HF dataset ids (overridable). Chosen for stable, widely-used schemas.
DEFAULT_IDS = {
    "medqa": "GBaker/MedQA-USMLE-4-options",   # fields: question, answer, options
    "medquad": "lavita/MedQuAD",               # fields: question, answer (some empty)
    "medicationqa": "truehealth/medicationqa",  # fields: Question, Answer (+ Focus/Type)
    "medmcqa": "openlifescienceai/medmcqa",     # fields: question, opa-opd, cop
    # Real patient->doctor dialogue (HealthCareMagic via the ChatDoctor release).
    "chatdoctor": "lavita/ChatDoctor-HealthCareMagic-100k",  # fields: instruction, input, output
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BUILDERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-id", default=None, help="override HF dataset id")
    ap.add_argument("--split", default="train", help="HF split to pull from")
    ap.add_argument("--max", type=int, default=None, help="cap number of rows")
    args = ap.parse_args()

    from datasets import load_dataset  # lazy: only needed on the box
    ds_id = args.dataset_id or DEFAULT_IDS[args.benchmark]
    print(f"Loading {ds_id} [{args.split}] ...")
    ds = load_dataset(ds_id, split=args.split)

    build = BUILDERS[args.benchmark]
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, written = set(), 0
    with open(out, "w") as f:
        for ex in ds:
            built = build(ex)
            if built is None:
                continue
            question, user, rubrics, tags = built
            pid = stable_id(args.benchmark, question)
            if pid in seen:
                continue
            seen.add(pid)
            f.write(json.dumps({
                "prompt_id": pid,
                "prompt": [{"role": "user", "content": user}],
                "rubrics": rubrics,
                "example_tags": tags,
                "_source": args.benchmark,
            }) + "\n")
            written += 1
            if args.max and written >= args.max:
                break
    print(f"wrote {written} rows -> {out}")
    if written == 0:
        raise SystemExit(
            f"No rows written. Inspect the {ds_id} schema and adjust the field names "
            f"in {args.benchmark}_row() (the mirror may use different keys)."
        )


if __name__ == "__main__":
    main()
