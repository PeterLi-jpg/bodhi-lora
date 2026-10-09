"""The leakage gates: no row a cell may train on is held out, by prompt id or by text.

check_leakage runs on the graded teacher traces, which hold every trace any seed of the
cell can train on, before training starts (audit EVAL-12). It fails when a traced prompt
is in the eval pool, the probe pool or any seed's eval ids, or when a traced prompt's last
user turn equals a held-out prompt's after lowercasing and collapsing spaces (a near-copy
under another id). It also checks that each seed's eval ids lie in the eval population.

check_adapter_leakage runs for an eval-only cell, which evaluates another cell's adapters,
often on another benchmark: MediQ cases are MedQA vignettes, for example, so they may be
the training prompts of a MedQA cell under other ids and another template. It fails when
an eval prompt is a trained prompt by id or by text, or when at least NEAR_COPY_SHARE of
its 13-word sequences occur in the trained prompts. 13 words is the overlap unit of the
GPT-3 contamination analysis (Brown et al., 2020, appendix C); the share, and not any
single match, decides, because clinical vignettes share stock phrases.
"""
from __future__ import annotations

from typing import Iterable

from bodhi_lora.data.splits import Split


class LeakageError(RuntimeError):
    """A held-out prompt reached the training data. .report holds the details."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def normalize_text(text: str) -> str:
    """Lowercase, spaces collapsed: the text compare of the gate."""
    return " ".join(str(text).lower().split())


def check_leakage(trained: dict[str, str], split: Split, held_texts: dict[str, str],
                  eval_ids_by_seed: dict[int, Iterable[str]], population: Iterable[str]) -> dict:
    """Check the trained prompts against everything held out; return the report.

    trained: prompt id -> last user turn of each prompt a trace exists for.
    held_texts: prompt id -> last user turn of each held-out prompt.
    population: the ids eval ids may come from (eval pool, or the label groups of
        eval.ids.only).
    Raises LeakageError, with the report attached, on any problem.
    """
    eval_ids = {seed: sorted(set(ids)) for seed, ids in eval_ids_by_seed.items()}
    held = set(split.eval) | set(split.probe) | {pid for ids in eval_ids.values() for pid in ids}
    id_leak = sorted(held & set(trained))
    held_norm = {normalize_text(held_texts[pid]) for pid in held if held_texts.get(pid)}
    text_leak = sorted(pid for pid, text in trained.items()
                       if text and normalize_text(text) in held_norm and pid not in held)
    allowed = set(population)
    outside = {seed: sorted(set(ids) - allowed) for seed, ids in eval_ids.items()}
    report = {
        "n_trained_prompts": len(trained), "n_eval_pool": len(split.eval),
        "n_probe_pool": len(split.probe), "n_held_out": len(held),
        "id_leak": len(id_leak), "id_leak_first": id_leak[:5],
        "text_duplicates": len(text_leak), "text_duplicates_first": text_leak[:5],
        "eval_ids": {str(s): {"n": len(ids), "outside_population": len(outside[s])}
                     for s, ids in eval_ids.items()},
    }
    problems = []
    if id_leak:
        problems.append(f"{len(id_leak)} traced prompts are held out, e.g. {id_leak[0]}")
    if text_leak:
        problems.append(f"{len(text_leak)} traced prompts copy a held-out prompt's text, e.g. {text_leak[0]}")
    for seed, ids in outside.items():
        if ids:
            problems.append(f"seed {seed}: {len(ids)} eval ids are outside the eval population")
    report["passed"] = not problems
    report["problems"] = problems
    if problems:
        raise LeakageError("leakage gate failed: " + "; ".join(problems), report)
    return report


NGRAM = 13
# Measured on HealthBench, the largest share of a held-out prompt's 13-word sequences
# found in the 3,321 train-pool prompts of the default split is 0.49 (one prompt;
# two more are above 0.2). A copy of a vignette under another template is far above.
NEAR_COPY_SHARE = 0.8


def word_ngrams(text: str, n: int = NGRAM) -> set:
    words = normalize_text(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def check_adapter_leakage(trained: dict[str, str], eval_texts: dict[str, str]) -> dict:
    """Check an eval-only cell's eval prompts against the prompts its adapters trained on.

    trained: prompt id -> last user turn of every prompt the adapter cell kept.
    eval_texts: prompt id -> last user turn of every eval prompt of every seed.
    Raises LeakageError, with the report attached, on any problem.
    """
    index: set = set()
    for text in trained.values():
        index |= word_ngrams(text)
    trained_norm = {normalize_text(t) for t in trained.values() if t}
    id_leak = sorted(set(eval_texts) & set(trained))
    text_leak = sorted(pid for pid, t in eval_texts.items() if t and normalize_text(t) in trained_norm)
    shares = {}
    for pid, text in eval_texts.items():
        grams = word_ngrams(text)
        if grams:
            shares[pid] = len(grams & index) / len(grams)
    near = sorted(pid for pid, share in shares.items() if share >= NEAR_COPY_SHARE)
    report = {
        "n_trained_prompts": len(trained), "n_eval_prompts": len(eval_texts),
        "id_leak": len(id_leak), "id_leak_first": id_leak[:5],
        "text_duplicates": len(text_leak), "text_duplicates_first": text_leak[:5],
        "ngram": NGRAM, "near_copy_share": NEAR_COPY_SHARE,
        "near_copies": len(near), "near_copies_first": near[:5],
        "max_share": max(shares.values()) if shares else None,
        "n_share_at_least_half": sum(v >= 0.5 for v in shares.values()),
        "n_too_short_for_ngrams": len(eval_texts) - len(shares),
    }
    problems = []
    if id_leak:
        problems.append(f"{len(id_leak)} eval prompts are trained prompts, e.g. {id_leak[0]}")
    if text_leak:
        problems.append(f"{len(text_leak)} eval prompts copy a trained prompt's text, e.g. {text_leak[0]}")
    if near:
        problems.append(f"{len(near)} eval prompts share at least {NEAR_COPY_SHARE:.0%} of their "
                        f"{NGRAM}-word sequences with the trained prompts, e.g. {near[0]}")
    report["passed"] = not problems
    report["problems"] = problems
    if problems:
        raise LeakageError("adapter leakage gate failed: " + "; ".join(problems), report)
    return report
