"""The leakage gate: no row a cell may train on is held out, by prompt id or by text.

It runs on the graded teacher traces, which hold every trace any seed of the cell can
train on, before training starts (audit EVAL-12). It fails when a traced prompt is in
the eval pool, the probe pool or any seed's eval ids, or when a traced prompt's last user
turn equals a held-out prompt's after lowercasing and collapsing spaces (a near-copy
under another id). It also checks that each seed's eval ids lie in the eval population.
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
