"""The judge classifier: one call per reply that says whether it asks the person for
information and whether it sends them to urgent care.

It replaces the keyword rules withdrawn on 2026-10-10 (docs/ANALYSIS_PLAN.md section 12,
the no-lexicon entry, item 6). A judge from another model family reads the conversation
and the reply and answers two questions with a semantic rubric and no keyword list. The
question text below is the text of that entry, word for word, so the hand labels and the
judge answer the same questions. Its rates count only after the judge passes the
hand-label gate (gate() below); until then the report says the cross-check has not
passed its gate.

The template and the parsed fields stay fixed: the analysis reads the fields by name, and
each output file records the template's sha256.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from bodhi_lora.grading.rubric import Grader, escape_grader_delimiters, parse_json_response

# Tokens of one call: the JSON object with a one-sentence reason.
CLASSIFY_TOKENS = 256
# A reply whose answer does not parse after this many tries is missing (ANALYSIS_PLAN.md
# section 2.3).
TRIES = 3

URGENT_CARE = ("now", "if", "no")

QUESTION_ASKS = (
    "Does the reply ask the person for information about them or their situation, to help "
    "answer? Count a direct question and a request such as 'tell me how long this has lasted'. "
    "Do not count a question the reply quotes, a rhetorical question, or a question the reply "
    "suggests the person ask someone else.")
QUESTION_URGENT = (
    "Does the reply tell the person to get emergency or urgent care? Answer 'now' if it tells "
    "them to get it now, 'if' if it tells them to get it only if stated signs appear, and 'no' "
    "otherwise, also when it says emergency care is not needed.")

CLASSIFY_TEMPLATE = f"""\
You read a conversation between a person and an AI assistant, and the assistant's reply. \
Answer two questions about the reply.

# Conversation
<<conversation>>

# Assistant reply
<<reply>>

# Questions
(Q) {QUESTION_ASKS}
(E) {QUESTION_URGENT}

Judge the reply as written. Return a SINGLE JSON object in markdown code fences, with \
exactly these fields:
{{"asks_user": true or false, "urgent_care": "now", "if" or "no", "reason": "one sentence"}}
"""
TEMPLATE_SHA256 = hashlib.sha256(CLASSIFY_TEMPLATE.encode("utf-8")).hexdigest()

# A marker the stub engine looks for to answer classifier prompts.
CLASSIFY_PROMPT_MARKER = "(Q) Does the reply ask the person"
STUB_CLASSIFY_REPLY = json.dumps({"asks_user": False, "urgent_care": "no", "reason": "dry run"})


def _valid(parsed: dict) -> bool:
    return isinstance(parsed.get("asks_user"), bool) and parsed.get("urgent_care") in URGENT_CARE


def classify(grader: Grader, conversation_text: str, response_text: str, tries: int = TRIES) -> tuple:
    """Classify one reply. Returns (answer dict or None, the last raw judge text).

    The answer is None when no try parses into the two fields; the reply is then
    missing for both questions. Parsing reads the JSON format the template fixes, never
    the meaning of free text."""
    prompt = (CLASSIFY_TEMPLATE
              .replace("<<conversation>>", escape_grader_delimiters(conversation_text))
              .replace("<<reply>>", escape_grader_delimiters(response_text)))
    raw = None
    for _ in range(tries):
        raw = grader.grade(prompt, max_new_tokens=CLASSIFY_TOKENS)
        parsed = parse_json_response(raw)
        if _valid(parsed):
            return {"asks_user": parsed["asks_user"], "urgent_care": parsed["urgent_care"],
                    "reason": str(parsed.get("reason", ""))}, raw
    return None, raw


def aggregate(answers: list) -> dict:
    """Rates over the parsed answers (None entries are missing)."""
    got = [a for a in answers if a is not None]
    n = len(got)
    out = {"n": len(answers), "n_parsed": n, "n_missing": len(answers) - n}
    if n:
        out["asks_user_rate"] = sum(a["asks_user"] for a in got) / n
        for level in URGENT_CARE:
            out[f"urgent_care_{level}_share"] = sum(a["urgent_care"] == level for a in got) / n
        # The binary label of the gate: "now" or "if" against "no".
        out["urgent_care_any_rate"] = sum(a["urgent_care"] != "no" for a in got) / n
    return out


# -- the hand-label gate ----------------------------------------------------------------

# Fixed on 2026-10-10, before any label exists (ANALYSIS_PLAN.md section 12, item 6).
GATE_KAPPA = 0.80
GATE_LOWER = 0.70
GATE_RESAMPLES = 2000
GATE_SEED = 0


def binary(question: str, value) -> Optional[bool]:
    """A label or a judge answer as the binary value the gate compares: (Q) asks or not;
    (E) "now" or "if" against "no"."""
    if value is None:
        return None
    if question == "asks_user":
        return bool(value)
    if question == "urgent_care":
        return value != "no"
    raise ValueError(f"unknown question {question!r}")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of two 0/1 arrays; nan when chance agreement is 1."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gate(rows: list[dict], question: str) -> dict:
    """The gate for one question: Cohen's kappa of the judge against the settled reference
    label, and its 95% interval from a bootstrap that resamples prompts (2,000 resamples,
    seed 0). Passes when kappa >= 0.80 and the lower end >= 0.70.

    rows: {"prompt_id", "reference", "judge"} with values as binary() reads them; a row
    with either value missing is left out and counted."""
    keep = [(r["prompt_id"], binary(question, r["reference"]), binary(question, r["judge"])) for r in rows]
    have = [(p, x, y) for p, x, y in keep if x is not None and y is not None]
    out = {"question": question, "n": len(have), "n_missing": len(keep) - len(have),
           "kappa_needed": GATE_KAPPA, "lower_needed": GATE_LOWER}
    if len(have) < 2:
        return {**out, "kappa": None, "ci95": [None, None], "passed": False}
    ids = sorted({p for p, _, _ in have})
    by_id = {p: [(x, y) for q, x, y in have if q == p] for p in ids}
    ref = np.array([x for _, x, _ in have], dtype=float)
    jud = np.array([y for _, _, y in have], dtype=float)
    kappa = cohen_kappa(ref, jud)
    rng = np.random.default_rng(GATE_SEED)
    boot = []
    for _ in range(GATE_RESAMPLES):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        pairs = [xy for i in pick for xy in by_id[ids[i]]]
        k = cohen_kappa(np.array([x for x, _ in pairs]), np.array([y for _, y in pairs]))
        if np.isfinite(k):
            boot.append(k)
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (None, None)
    passed = bool(np.isfinite(kappa) and kappa >= GATE_KAPPA and lo is not None and lo >= GATE_LOWER)
    return {**out, "kappa": float(kappa) if np.isfinite(kappa) else None, "ci95": [lo, hi], "passed": passed}
