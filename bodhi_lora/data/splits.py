"""The pools of a split, and the eval ids of each seed.

A split (SplitRule in a benchmark record) divides the rows into three pools:

  train  the rows a cell may generate teacher traces for, in generation order
  eval   the rows a cell evaluates on by default; no row of it is ever trained on
  probe  labelled rows held out of training (hold_out), evaluated only when a cell's
         eval ids name their label groups

Draw rules (Draw in the schema), applied to an ordered list of prompt ids:
  all     every id, in order
  head    the first `size` ids, in order
  seeded  random.Random(seed).sample of `size` ids from the list sorted by prompt id,
          then sorted. Sorting first makes the draw independent of the source order.
These reproduce the earlier builders exactly: the 2026-07 cells drew their pools and
holdouts by head, and the seeded pool and holdout were the default of the old launcher.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bodhi_lora.data.labels import in_groups
from bodhi_lora.data.loaders import SOURCE, Row
from bodhi_lora.registry.schema import BenchmarkConfig, CellConfig, Draw, LabelGroups


class SplitError(ValueError):
    """A split or an eval-id rule cannot be applied to these rows."""


@dataclass(frozen=True)
class Split:
    name: str
    train: list[str]
    eval: list[str]
    probe: list[str]


def draw(ids: list[str], rule: Draw, *, short_ok: bool) -> list[str]:
    """Apply a draw rule. short_ok: take every id when `size` exceeds the list."""
    if rule.select == "all":
        return list(ids)
    if rule.size > len(ids):
        if not short_ok:
            raise SplitError(f"cannot draw {rule.size} of {len(ids)} ids")
        return sorted(ids) if rule.select == "seeded" else list(ids)
    if rule.select == "head":
        return list(ids[:rule.size])
    return sorted(random.Random(rule.seed).sample(sorted(ids), rule.size))


def _in_file(rows: list[Row], key: str) -> list[str]:
    """Ids of the rows in one source file, in that file's order."""
    found = [(r.positions[key], r.prompt_id) for r in rows if key in r.positions]
    return [pid for _, pid in sorted(found)]


def _held_out(rows_by_id: dict[str, Row], ids: list[str], groups: list[LabelGroups]) -> set[str]:
    return {pid for pid in ids
            if any(in_groups(rows_by_id[pid].labels.get(g.label), g.groups) for g in groups)}


def split_rows(bench: BenchmarkConfig, split_name: str, rows: list[Row],
               fixed_eval_ids: Optional[list[str]] = None) -> Split:
    """The train, eval and probe pools of one split of a benchmark.

    fixed_eval_ids: the cell's eval ids, for an eval_ids_only split.
    """
    rule = bench.splits[split_name]
    by_id = {r.prompt_id: r for r in rows}
    if rule.method == "named_files":
        eval_pool = _in_file(rows, rule.eval_file)
        held = set(eval_pool)
        train = [pid for pid in _in_file(rows, rule.train_file) if pid not in held]
    elif rule.method == "eval_ids_only":
        if not fixed_eval_ids:
            raise SplitError(f"split {split_name} needs the cell's fixed eval ids")
        missing = set(fixed_eval_ids) - set(by_id)
        if missing:
            raise SplitError(f"{len(missing)} fixed eval ids are not rows of {bench.id}")
        eval_pool = list(fixed_eval_ids)
        held = set(eval_pool)
        files = {rule.train_file, rule.eval_file}
        train = [r.prompt_id for r in rows if files & set(r.positions) and r.prompt_id not in held]
    else:
        order = _in_file(rows, SOURCE) if any(SOURCE in r.positions for r in rows) else [r.prompt_id for r in rows]
        pool = draw(order, rule.pool, short_ok=True)
        eval_pool = draw(pool, rule.holdout, short_ok=False)
        held = set(eval_pool)
        train = [pid for pid in pool if pid not in held]
    probe = _held_out(by_id, train, rule.hold_out)
    return Split(name=split_name, train=[p for p in train if p not in probe], eval=eval_pool,
                 probe=[p for p in train if p in probe])


def eval_population(cell: CellConfig, split: Split, rows_by_id: dict[str, Row]) -> list[str]:
    """The ids a cell draws its eval ids from: the eval pool, or the label groups that
    eval.ids.only names, taken from the eval and probe pools."""
    only = cell.eval.ids.only
    if not only:
        return list(split.eval)
    return [pid for pid in split.eval + split.probe
            if any(in_groups(rows_by_id[pid].labels.get(g.label), g.groups) for g in only)]


def _whole_pairs(population: list[str], rows_by_id: dict[str, Row]) -> list[str]:
    """The sorted pair ids whose rows are all in the population."""
    members: dict[str, set] = {}
    for pid, row in rows_by_id.items():
        if row.pair_id is not None:
            members.setdefault(row.pair_id, set()).add(pid)
    if not members:
        raise SplitError("eval.ids.unit is pair, but the rows carry no pair ids")
    inside = set(population)
    return sorted(pair for pair, rows in members.items() if rows <= inside)


def read_id_list(path: Path) -> list[str]:
    """Prompt ids from a JSON list or a {"prompt_ids": [...]} file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["prompt_ids"] if isinstance(data, dict) else data)


def eval_ids(cell: CellConfig, split: Split, rows_by_id: dict[str, Row], seed: int,
             inputs_dir: Path) -> list[str]:
    """The eval ids of one seed.

    per_seed_draw: random.Random(seed).sample(n) of the population in pool order, sorted
        (the per-seed draw of the 2026-05 cell).
    fixed_pool: the whole population, which must hold n ids.
    fixed_file: the ids in <inputs_dir>/<file>, the same for every seed.
    With unit pair the same rules apply to the pair ids, sorted, and each drawn pair
    brings both of its rows; a pair with a row outside the population is left out.
    """
    ids = cell.eval.ids
    if ids.policy == "fixed_file":
        out = read_id_list(Path(inputs_dir) / ids.file)
        allowed = set(eval_population(cell, split, rows_by_id))
        units = out
    else:
        population = eval_population(cell, split, rows_by_id)
        allowed = set(population)
        units = _whole_pairs(population, rows_by_id) if ids.unit == "pair" else population
        if ids.policy == "fixed_pool":
            out = units
        else:
            if len(units) < ids.n:
                raise SplitError(f"cannot draw {ids.n} eval {ids.unit}s from {len(units)}")
            out = sorted(random.Random(seed).sample(units, ids.n))
    if len(out) != ids.n:
        raise SplitError(f"eval.ids.n is {ids.n}, but the {ids.policy} {ids.unit}s number {len(out)}")
    if ids.unit == "pair":
        members: dict[str, list[str]] = {}
        for pid in sorted(allowed):
            members.setdefault(rows_by_id[pid].pair_id, []).append(pid)
        out = [pid for pair in out for pid in members[pair]]
    outside = set(out) - allowed
    if outside:
        raise SplitError(f"{len(outside)} eval ids are outside the eval population, e.g. {sorted(outside)[0]}")
    return list(out)
