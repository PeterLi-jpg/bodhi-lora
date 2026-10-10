"""Raw stage outputs of a cell -> the tables the analysis reads.

    <cell>/seed_<n>/table.csv         one row per (condition, prompt id) with a completion
    <cell>/seed_<n>/table.meta.json   sha256 of every source file, completions drawn per
                                      condition, the grader's error counts, the tokenizer
    <cell>/analysis/prompts.csv       one row per prompt of the eval pool (or, when the
                                      benchmark rows are not local, per evaluated prompt):
                                      theme, positive rubric points, and the labels that
                                      configs/report.yaml lists
    <cell>/analysis/extract.json      how the pool was read, the draw check, rule hashes,
                                      and the population counts of every label

The raw eval and grade JSON are large and stay out of git; the tables are tracked, so
`analyze` without `--extract` needs nothing else and runs in CI.

Response table columns (docs/ARCHITECTURE.md section 6):

    key        cell, seed, condition, prompt_id, pair_id (the rows of one pair of a paired
               benchmark share it; empty for other benchmarks and in older files)
    response   chars; tokens (the model's own tokenizer on the reply, when its
               tokenizer.json is in the local Hugging Face cache); tokens_engine (the count
               the serving engine stored, single-call conditions only); calls, tokens_total,
               finish_length (stored by the package's eval stage; empty in older files)
    rubric     rubric_score (normalized; recomputed from the verdicts when the file stored
               another score, see STORED_SCORE), rubric_score_worst (failed negative
               criteria counted as met),
               rubric_items, rubric_parse_failures, neg_items_met, neg_points_lost,
               axis_<axis> for each rubric axis tag
    evaluator  epi_status (scored, parse_failure, missing, not_graded) and the dimension
               columns of EPISTEMIC, empty unless scored
    format     leak_<protocol>__<rule>, the marker count of each leak rule of each protocol
               the cell uses (the protocol's own section headers or field names, a parser
               of its fixed format; no column reads meaning from the reply text)
    judges     j_<judge>__rubric_score, j_<judge>__<dimension>, and the 2x2 counts of
               rubric criteria met by the evaluator and the judge (j_<judge>__crit_<ej>)

A value an older file does not have is empty, and the analysis reports it as not
measured for that cell.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bodhi_lora.analysis.config import ReportConfig
from bodhi_lora.paths import CellPaths, Project
from bodhi_lora.registry import CellConfig, Registry, condition_from_legacy, parse_condition

# The evaluator's grade fields -> table columns. The grader's template and field names
# are fixed (bodhi_lora/grading/epistemic.py); the rates follow its aggregate().
EPISTEMIC = (
    "uncertainty_ack", "active_inquiry", "context_seeking", "red_flag", "scope_bounding",
    "specificity", "hedging_quality", "questions", "red_flag_rate", "scope_bounded_rate",
    "specificity_rate", "blanket_disclaimer", "appropriate_hedging",
)
# The seven dimensions: the Holm family within a contrast (decision D7).
SEVEN = ("uncertainty_ack", "active_inquiry", "context_seeking", "red_flag", "scope_bounding",
         "hedging_quality", "specificity")

KEY = ["cell", "seed", "condition", "prompt_id", "pair_id"]
RESPONSE = ["chars", "tokens", "tokens_engine", "calls", "tokens_total", "finish_length"]
RUBRIC = ["rubric_score", "rubric_score_worst", "rubric_items", "rubric_parse_failures",
          "neg_items_met", "neg_points_lost"]


class ExtractError(RuntimeError):
    """A cell's raw outputs are missing or disagree with each other."""


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def epistemic_columns(scores: dict) -> dict:
    """One evaluator grade -> the dimension columns."""
    rf, sb = scores.get("red_flag_identification"), scores.get("scope_bounding")
    return {
        "uncertainty_ack": _num(scores.get("uncertainty_acknowledgment")),
        "active_inquiry": float(bool(scores.get("active_inquiry"))),
        "context_seeking": _num(scores.get("context_seeking")),
        "red_flag": _num(rf),
        "scope_bounding": _num(sb),
        "specificity": _num(scores.get("specificity")),
        "hedging_quality": _num(scores.get("hedging")),
        "questions": float(scores.get("n_questions") or 0),
        "red_flag_rate": float(rf is not None and rf >= 2),
        "scope_bounded_rate": float(sb is not None and sb >= 2),
        "specificity_rate": float(bool(scores.get("has_concrete_numbers"))),
        "blanket_disclaimer": float(bool(scores.get("is_blanket_disclaimer"))),
        "appropriate_hedging": float(bool(scores.get("appropriate_hedging"))),
    }


def raw_rubric_score(criteria: list[dict]) -> float:
    """The benchmark's own per-example score, before its aggregate clips it: earned points
    (negative criteria met count against) over positive points. Eval files without a
    score_metric field (written before the field existed) stored this one."""
    pos = sum(c["points"] for c in criteria if c["points"] > 0)
    earned = sum(c["points"] for c in criteria if c["criteria_met"])
    return earned / pos if pos > 0 else 0.0


def rubric_score(criteria: list[dict], flip_failed_negatives: bool = False) -> float:
    """Normalized rubric score: (earned - negative points) / (positive - negative points).

    With flip_failed_negatives a negative criterion whose grade failed to parse counts as
    met. The grader records a failed parse as not met, which is the conservative reading
    for a positive criterion but the generous one for a negative criterion, so this gives
    the worst case (decision D13).
    """
    pos = sum(c["points"] for c in criteria if c["points"] > 0)
    neg = sum(c["points"] for c in criteria if c["points"] < 0)
    earned = 0.0
    for c in criteria:
        met = c["criteria_met"]
        if flip_failed_negatives and c.get("parse_failed") and c["points"] < 0:
            met = True
        if met:
            earned += c["points"]
    rng = pos - neg
    return (earned - neg) / rng if rng > 0 else 0.0


# The score each eval file says it stored, by its score_metric field -> how to recompute
# it from the stored criterion verdicts. The table holds the normalized score for every
# file, so cells compare on one formula.
STORED_SCORE = {"normalized_score": rubric_score, None: raw_rubric_score}


# -- format leakage ---------------------------------------------------------------------

def leak_column(protocol_id: str, rule_id: str) -> str:
    return f"leak_{protocol_id}__{rule_id}"


def leak_rules(reg: Registry, cell: CellConfig) -> dict[str, tuple[list[re.Pattern], int]]:
    """Column -> (markers, threshold) for every leak rule of every protocol the cell uses.

    A marker is one of the protocol's own section headers or field names, counted only
    where a header starts (LeakRule.patterns), so the count parses the protocol's fixed
    output format; it reads no meaning from the reply."""
    protocols = {cell.protocol} | {p for c in cell.eval.conditions for p in [parse_condition(c)[1]] if p}
    out = {}
    for pid in sorted(protocols):
        for rid, rule in sorted(reg.protocols[pid].leak_rules.items()):
            out[leak_column(pid, rid)] = (rule.patterns(), rule.threshold)
    return out


def _definition_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- tokenizer --------------------------------------------------------------------------

def local_tokenizer(model) -> tuple[Optional[object], str]:
    """The model's tokenizer.json at its pinned revision, from the local Hugging Face cache.

    It never downloads (huggingface_hub.try_to_load_from_cache, which only reads the
    cache). Returns (tokenizer or None, what was used or why not).
    """
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model.hf_id, "tokenizer.json", revision=model.revision)
    if not isinstance(path, str):
        return None, (f"tokenizer.json of {model.hf_id} at {model.revision} is not in the local "
                      f"Hugging Face cache (fetch it with: huggingface-cli download {model.hf_id} "
                      f"tokenizer.json --revision {model.revision})")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path), f"{model.hf_id}@{model.revision}"


def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    """Tokens of each reply, without the special tokens a chat template adds."""
    return [len(e.ids) for e in tokenizer.encode_batch([t or "" for t in texts], add_special_tokens=False)]


# -- reading one seed -------------------------------------------------------------------

def _condition_of(data: dict) -> str:
    """The condition a file's rows belong to, in the package's names."""
    name = data.get("condition") or data.get("config")
    try:
        parse_condition(name)
        return name
    except ValueError:
        return condition_from_legacy(name)


def _grade_examples(path: Path) -> tuple[dict, dict]:
    """condition -> {prompt id -> example}, and condition -> the grader's counts."""
    if not path.is_file():
        return {}, {}
    data = json.loads(path.read_text())
    by_cond, counts = {}, {}
    for c in data["configs"]:
        cond = _condition_of({"condition": c["name"]})
        by_cond[cond] = {e["prompt_id"]: e for e in c["examples"]}
        counts[cond] = {"examples": len(c["examples"]), "n_grade_errors": c.get("n_grade_errors", 0),
                        "n_failed_lookups": c.get("n_failed_lookups", 0)}
    return by_cond, counts


def _judge_examples(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {_condition_of({"condition": c["name"]}): {e["prompt_id"]: e for e in c["examples"]}
            for c in data["configs"]}


def _judge_columns(judge_id: str, ex: Optional[dict], criteria: list[dict]) -> dict:
    """A judge's rubric score and dimensions, and its criterion verdicts against the evaluator's."""
    j = f"j_{judge_id}__"
    out = {f"{j}rubric_score": float("nan"), **{f"{j}{k}": float("nan") for k in EPISTEMIC}}
    out.update({f"{j}crit_{a}{b}": float("nan") for a in "10" for b in "10"})
    if ex is None:
        return out
    out[f"{j}rubric_score"] = _num(ex.get("judge_score"))
    if not ex.get("epistemic_parse_failure") and ex.get("scores") is not None:
        out.update({f"{j}{k}": v for k, v in epistemic_columns(ex["scores"]).items()})
    met = ex.get("judge_criteria_met")
    if met is not None and len(met) == len(criteria):
        pairs = [(bool(c["criteria_met"]), bool(m)) for c, m in zip(criteria, met)]
        for a in (True, False):
            for b in (True, False):
                out[f"{j}crit_{int(a)}{int(b)}"] = float(sum(p == (a, b) for p in pairs))
    return out


def seed_table(reg: Registry, cell: CellConfig, paths: CellPaths, seed: int, cfg: ReportConfig,
               tokenizer, project: Project) -> tuple[pd.DataFrame, dict]:
    """The response table of one seed and its meta record."""
    leaks = leak_rules(reg, cell)
    grade_path = paths.grade_file(seed, cell.grading.evaluator)
    grades, grade_counts = _grade_examples(grade_path)
    judges = {}
    for jid in cell.grading.judges:
        jp = paths.judge_file(seed, jid)
        if jp.is_file():
            judges[jid] = _judge_examples(jp)
    meta = {"cell": cell.id, "seed": seed, "sources": {}, "n_examples": {}, "completions": {},
            "grade": {"file": project.rel(grade_path), "present": grade_path.is_file(),
                      "counts": grade_counts}}
    rows = []
    for condition in cell.eval.conditions:
        path = paths.eval_file(seed, condition)
        data = json.loads(path.read_text())
        meta["sources"][project.rel(path)] = sha256_file(path)
        if _condition_of(data) != condition:
            raise ExtractError(f"{path}: rows are condition {_condition_of(data)!r}, not {condition!r}")
        results = data["results"]
        ids = [r["prompt_id"] for r in results]
        stored = data.get("score_metric")
        if stored not in STORED_SCORE:
            raise ExtractError(f"{path}: unknown score_metric {stored!r}")
        meta.setdefault("score_metric", {})[condition] = stored
        if len(ids) != len(set(ids)):
            raise ExtractError(f"{path}: a prompt id repeats")
        meta["n_examples"][condition] = int(data.get("n_examples", len(results)))
        meta["completions"][condition] = len(results)
        cond_grades = grades.get(condition, {})
        extra = set(cond_grades) - set(ids)
        if extra:
            raise ExtractError(f"{grade_path}: {len(extra)} {condition} grades have no completion")
        not_graded = not cond_grades
        tokens = count_tokens(tokenizer, [r["response"] for r in results]) if tokenizer else None
        for i, r in enumerate(results):
            criteria = r["criteria_results"]
            # The recompute of the stored score must equal it, or we do not read the
            # verdicts the way the run did, and the worst case is not comparable.
            again = STORED_SCORE[stored](criteria)
            if not math.isclose(again, r["score"], abs_tol=1e-9):
                raise ExtractError(f"{path}: {r['prompt_id']}: rubric score {r['score']} does not "
                                   f"recompute ({again})")
            score = float(r["score"]) if stored == "normalized_score" else rubric_score(criteria)
            neg_met = [c for c in criteria if c["points"] < 0 and c["criteria_met"]]
            row = {"cell": cell.id, "seed": seed, "condition": condition, "prompt_id": r["prompt_id"],
                   "pair_id": r.get("pair_id"), "chars": float(len(r["response"])),
                   "tokens": float(tokens[i]) if tokens is not None else float("nan"),
                   "tokens_engine": float(r["response_token_count"]) if r.get("response_token_count") else float("nan"),
                   "calls": _num(r.get("calls")),
                   "tokens_total": _num(r.get("completion_tokens_total")),
                   "finish_length": float(r["finish_length"]) if "finish_length" in r else float("nan"),
                   "rubric_score": score,
                   "rubric_score_worst": rubric_score(criteria, flip_failed_negatives=True),
                   "rubric_items": float(len(criteria)),
                   "rubric_parse_failures": float(sum(bool(c.get("parse_failed")) for c in criteria)),
                   "neg_items_met": float(len(neg_met)),
                   "neg_points_lost": float(-sum(c["points"] for c in neg_met))}
            for tag, v in (r.get("tag_scores") or {}).items():
                if tag.startswith("axis:"):
                    row["axis_" + tag.split(":", 1)[1]] = _num(v)
            e = cond_grades.get(r["prompt_id"])
            if not_graded:
                row["epi_status"] = "not_graded"
            elif e is None:
                row["epi_status"] = "missing"
            elif e.get("parse_failure") or e.get("scores") is None:
                row["epi_status"] = "parse_failure"
            else:
                row["epi_status"] = "scored"
                row.update(epistemic_columns(e["scores"]))
            for col, (markers, _threshold) in leaks.items():
                row[col] = float(sum(1 for m in markers if m.search(r["response"] or "")))
            for jid, jex in judges.items():
                row.update(_judge_columns(jid, jex.get(condition, {}).get(r["prompt_id"]), criteria))
            rows.append(row)
    for jid in judges:
        jp = paths.judge_file(seed, jid)
        meta["sources"][project.rel(jp)] = sha256_file(jp)
    if grade_path.is_file():
        meta["sources"][project.rel(grade_path)] = sha256_file(grade_path)
    df = pd.DataFrame(rows)
    for col in EPISTEMIC:
        if col not in df:
            df[col] = np.nan
    axes = sorted(c for c in df.columns if c.startswith("axis_"))
    judge_cols = sorted(c for c in df.columns if c.startswith("j_"))
    columns = KEY + RESPONSE + RUBRIC + axes + ["epi_status", *EPISTEMIC] + list(leaks) + judge_cols
    order = {c: i for i, c in enumerate(cell.eval.conditions)}
    df = df[columns].sort_values(["condition", "prompt_id"], key=lambda s: s.map(order) if s.name == "condition" else s)
    return df.reset_index(drop=True), meta


# -- the prompts of a cell --------------------------------------------------------------

def _theme(tags) -> Optional[str]:
    themes = [t.split(":", 1)[1] for t in tags if t.startswith("theme:")]
    return themes[0] if themes else None


def prompt_table(reg: Registry, cell: CellConfig, paths: CellPaths, project: Project, cfg: ReportConfig,
                 evaluated: dict[int, set[str]], criteria_points: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """One row per prompt of the eval population, with its theme, points and labels.

    The population comes from the benchmark rows when they are in the data root (no
    download). Otherwise only the evaluated prompts are known, and their positive points
    come from the criteria stored with the eval outputs.

    Only the labels that the report config lists get a column. The analysis plan keeps
    the other labels (the physician labels among them) unseen until its adoption, and a
    tracked column next to the outcome tables would make the join one merge away
    (docs/ANALYSIS_PLAN.md section 12, 2026-10-10). The record keeps every label's
    population counts, which say nothing about any output.
    """
    from bodhi_lora.data import DataError, SplitError, eval_ids, eval_population, read_id_list, \
        load_rows, split_rows

    bench = reg.benchmarks[cell.benchmark]
    every = sorted(set().union(*evaluated.values())) if evaluated else []
    info = {"benchmark": bench.id, "split": cell.split}
    try:
        rows = load_rows(bench, project.data_root, fetch=False)
    except DataError as e:
        # The error names the data file by its absolute path. The record is tracked, so it
        # names the path from the repository root: the same on every machine, and no home
        # folder in a public file.
        info["rows"] = "not available: " + str(e).replace(f"{project.repo_root.resolve()}/", "")
        df = pd.DataFrame({"prompt_id": every, "in_population": np.nan, "evaluated": 1.0,
                           "theme": None, "rubric_points_pos": [criteria_points.get(p, np.nan) for p in every]})
        return df, info
    by_id = {r.prompt_id: r for r in rows}
    fixed = None
    if bench.splits[cell.split].method == "eval_ids_only":
        fixed = read_id_list(paths.inputs() / cell.eval.ids.file)
    split = split_rows(bench, cell.split, rows, fixed_eval_ids=fixed)
    population = eval_population(cell, split, by_id)
    info["rows"] = "loaded"
    info["n_population"] = len(population)
    # The draw check: the ids each seed evaluated must be ids the record's rule draws.
    draws = {}
    for seed, ids in sorted(evaluated.items()):
        try:
            drawn = set(eval_ids(cell, split, by_id, seed, paths.inputs()))
        except (SplitError, FileNotFoundError) as e:
            draws[str(seed)] = {"error": str(e)}
            continue
        draws[str(seed)] = {"drawn": len(drawn), "evaluated": len(ids), "evaluated_not_drawn": len(ids - drawn)}
    info["draw_check"] = draws
    outside = sorted(set(every) - set(population))
    if outside:
        raise ExtractError(f"{cell.id}: {len(outside)} evaluated prompts are outside the eval population")
    ids = sorted(population)
    df = pd.DataFrame({"prompt_id": ids, "in_population": 1.0,
                       "evaluated": [float(p in set(every)) for p in ids],
                       "theme": [_theme(by_id[p].tags) for p in ids],
                       "rubric_points_pos": [float(sum(r["points"] for r in by_id[p].rubrics if r["points"] > 0))
                                             for p in ids]})
    listed = {lab for spec in cfg.discrimination for lab in spec.labels}
    info["labels"] = {}
    for name, rule in sorted(bench.labels.items()):
        values = [by_id[p].labels[name] for p in ids]
        info["labels"][name] = {"timing": rule.timing, "source": rule.source,
                                "definition_sha256": _definition_hash(rule.model_dump()),
                                "in_prompts_table": name in listed,
                                "population": {"positive": sum(v is True for v in values),
                                               "negative": sum(v is False for v in values),
                                               "unlabelled": sum(v is None for v in values)}}
        if name in listed:
            df[f"label_{name}"] = [None if v is None else float(v) for v in values]
    return df, info


# -- the cell ---------------------------------------------------------------------------

def output_seeds(cell: CellConfig, paths: CellPaths) -> list[int]:
    """The seeds of the cell whose eval outputs exist. A seed with some condition files
    but not all is an error, not a seed to skip."""
    seeds = []
    for seed in cell.seeds:
        have = [c for c in cell.eval.conditions if paths.eval_file(seed, c).is_file()]
        if not have:
            continue
        if len(have) != len(cell.eval.conditions):
            missing = sorted(set(cell.eval.conditions) - set(have))
            raise ExtractError(f"{cell.id} seed {seed}: eval outputs of {missing} are missing")
        seeds.append(seed)
    return seeds


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV with full float precision; read it back with read_table()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def read_table(path: Path) -> pd.DataFrame:
    # round_trip: the default float parser can be off in the last bit, and the
    # equivalence checks compare at full precision.
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True,
                       dtype={"prompt_id": str, "pair_id": str, "theme": str})


def extract(reg: Registry, cell_id: str, project: Project, cfg: ReportConfig) -> dict:
    """Write the tables of one cell from its raw outputs. Returns the extract record."""
    cell = reg.cells[cell_id]
    paths = project.cell(cell_id)
    seeds = output_seeds(cell, paths)
    if not seeds:
        raise ExtractError(f"{cell_id}: no eval outputs under {project.rel(paths.root)}")
    tokenizer, tok_source = local_tokenizer(reg.models[cell.model])
    evaluated, criteria_points = {}, {}
    for seed in seeds:
        df, meta = seed_table(reg, cell, paths, seed, cfg, tokenizer, project)
        meta["tokenizer"] = tok_source
        write_csv(df, paths.table(seed))
        paths.table(seed).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        evaluated[seed] = set(df["prompt_id"])
        # Positive points per prompt, for benchmarks whose rows are not local.
        for condition in cell.eval.conditions[:1]:
            data = json.loads(paths.eval_file(seed, condition).read_text())
            for r in data["results"]:
                criteria_points[r["prompt_id"]] = float(sum(c["points"] for c in r["criteria_results"]
                                                             if c["points"] > 0))
    prompts, info = prompt_table(reg, cell, paths, project, cfg, evaluated, criteria_points)
    write_csv(prompts, paths.analysis() / "prompts.csv")
    record = {"cell": cell_id, "seeds": seeds, "tokenizer": tok_source, "prompts": info,
              "leak_rules": {col: _definition_hash([[m.pattern for m in markers], threshold])
                             for col, (markers, threshold) in leak_rules(reg, cell).items()}}
    (paths.analysis() / "extract.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
