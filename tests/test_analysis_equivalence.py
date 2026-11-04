"""The uniform analysis reproduces the outputs it supersedes (docs/ARCHITECTURE.md 10.3).

It recomputes the 2026-05 cell from its tracked response tables and compares, at the
stored precision, with the files the old scripts wrote: the statistics of
scripts/paper_stats.py (removed on 2026-10-10; its last numbers.json sections are frozen in
tests/fixtures/legacy/paper_stats_numbers.json) and the cell's mimicry_split.json and
interference_mechanism.json (rebuttal/analyses/). It also checks every cell's stats.json
against the cross-cell records the removed scripts wrote: generated_values.json
(cell_values.py), quality_drop.json and cell_recipe.json. A mismatch is a finding to
explain, not a tolerance to raise. Old names map to new ones in the tables below; a
field the new output names differently is listed there, not skipped.

Some old values are not compared, for a stated reason:
  - mimicry_split groups: the old script counted labels over the base condition's scored
    prompts, so a prompt with no scored base reply showed as null; the rates match.
  - Values of the keyword rules withdrawn on 2026-10-10 (docs/CONVENTIONS.md rule 15),
    which the new code no longer computes: mimicry_split primary_theme_or_rubric (a
    rubric keyword labelling), second_person_questions.json (a keyword question rule),
    and interference_mechanism's nine-marker rule (leaked_rate, decisive and
    length_control_clean, whose clean set it defines). The theme-only labelling and the
    seven-header rule are compared.
"""
import json
import math

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.cell import analyze_tables, load_tables, round6  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

CELL = "healthbench__medgemma-27b-text__bodhi__2026-05"
CONDITION = {"base_no_wrapper": "base", "base_bodhi": "base+bodhi", "lora_no_wrapper": "lora",
             "lora_bodhi": "lora+bodhi"}
CONTRAST = {"LoRAvsBase": "lora - base", "BaseCoTvsBase": "base+bodhi - base",
            "LoRACoTvsBase": "lora+bodhi - base", "LoRAvsBaseCoT": "lora - base+bodhi",
            "LoRACoTvsLoRA": "lora+bodhi - lora"}
# Old field or metric name -> new name, wherever it appears.
FIELD = {"healthbench": "rubric_score", "p_holm_all_35": "p_holm_all_contrasts",
         "n_responses": "n", "n_unique_prompts_evaluated": "n_evaluated",
         "n_prompts_in_hard_set": "n_population", "hb_items": "rubric_items",
         "hb_item_parse_failures": "rubric_parse_failures",
         "hb_item_parse_failure_rate": "rubric_parse_failure_rate",
         "hb_item_parse_failure_rate_max_run": "rubric_parse_failure_rate_max_run",
         "prompt_evaluations_per_condition": "prompt_draws_per_condition"}


@pytest.fixture(scope="module")
def stats():
    reg = load_registry(CONFIGS_DIR)
    proj = project(CONFIGS_DIR)
    paths = proj.cell(CELL)
    if not paths.table(7).is_file():
        pytest.skip("the 2026-05 response tables are not in this checkout")
    table, prompts, meta = load_tables(reg.cells[CELL], paths)
    return round6(analyze_tables(reg, reg.cells[CELL], table, prompts, meta, paths, report_config()))


def _name(key):
    return CONDITION.get(key) or CONTRAST.get(key) or FIELD.get(key, key)


def compare(old, new, path, problems):
    """Every value of old has an equal value in new, after renaming keys."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            problems.append(f"{path}: not a mapping in the new output")
            return
        for k, v in old.items():
            if _name(k) not in new:
                problems.append(f"{path}/{k}: missing (as {_name(k)})")
                continue
            compare(v, new[_name(k)], f"{path}/{k}", problems)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            problems.append(f"{path}: list differs in length")
            return
        for i, (a, b) in enumerate(zip(old, new)):
            compare(a, b, f"{path}[{i}]", problems)
    elif isinstance(old, str):
        if _name(old) != new:
            problems.append(f"{path}: {old!r} against {new!r}")
    elif old is None or new is None or isinstance(old, bool):
        if old != new:
            problems.append(f"{path}: {old!r} against {new!r}")
    # The new output keeps 6 significant digits (round6), so the stored precision of a
    # comparison is half a unit in the sixth digit.
    elif not math.isclose(old, new, rel_tol=5e-6, abs_tol=1e-12):
        problems.append(f"{path}: {old!r} against {new!r}")


def test_numbers_of_the_paper(stats):
    numbers = json.loads((REPO_ROOT / "tests/fixtures/legacy/paper_stats_numbers.json").read_text())
    problems = []
    for section in ("conditions", "contrasts", "worst_case", "blanket_disclaimers", "length_quartiles",
                    "axis_scores", "themes", "counts"):
        compare(numbers[section], stats[section], section, problems)
    anchor = {c: {"n_score_2": v["n_score_2"], "n_score_2_no_question": v["n_score_2_no_question"],
                  "share_no_question": v["pct_no_question"] / 100}
              for c, v in numbers["context_seeking_anchor"].items()}
    compare(anchor, stats["context_seeking_anchor"], "context_seeking_anchor", problems)
    compare(numbers["shift_correlations"], stats["shift_correlations"]["lora - base"], "shift_correlations",
            problems)
    for c, v in numbers["response_size"].items():
        if c in CONDITION:
            new = stats["cost"]["conditions"][CONDITION[c]]
            compare({"chars_mean": v["chars_mean"], "chars_mean_scored": v["chars_mean_scored"],
                     "tokens_engine_mean": v["tokens_mean"], "tokens_engine_n": v["n_with_tokens"]},
                    new, f"response_size/{c}", problems)
    compare([{k: r[k] for k in ("tau", "share_of_train_split")} for r in numbers["tau_calibration"]["rows"]],
            stats["filter"]["scores"]["tau_grid"], "tau_calibration", problems)
    assert stats["filter"]["scores"]["n_train_split"] == numbers["tau_calibration"]["n_train_split"]
    assert not problems, "\n".join(problems)


def _analysis(name):
    """A withdrawn keyword-measure output of the cell, kept as a record (W23 to W25)."""
    return json.loads((REPO_ROOT / "results" / CELL / "analysis" / "withdrawn" / name).read_text())


def test_rebuttal_analyses_of_the_cell(stats):
    problems = []
    old = _analysis("mimicry_split.json")["robustness_theme_only"]
    disc = stats["discrimination"]["active_inquiry"]["needs-info-theme"]
    for c, v in old["configs"].items():
        compare({k: v[k] for k in ("discrimination", "discrimination_ci95", "odds_ratio", "odds_ratio_ci95")},
                disc["conditions"][CONDITION[c]], f"mimicry_split/{c}", problems)
        compare({"positive_rate": v["context_dependent_rate"], "negative_rate": v["self_contained_rate"]},
                disc["conditions"][CONDITION[c]], f"mimicry_split/{c}", problems)
    compare({"diff": old["lora_minus_base_disc"], "diff_ci95": old["lora_minus_base_disc_ci95"]},
            disc["contrasts"]["lora - base"], "mimicry_split/lora_minus_base", problems)

    # The analysis-format rule of the bodhi record is the old script's seven-header rule.
    old = _analysis("interference_mechanism.json")
    leak = stats["leakage"]["leak_bodhi__analysis-format"]["conditions"]
    assert set(stats["leakage"]) == {"leak_bodhi__analysis-format"}
    for c, v in old["configs"].items():
        new = CONDITION[c]
        compare({"n_scored": v["n"], "leaked_share": v["leaked_rate_pass1_headers"]}, leak[new],
                f"interference/{c}", problems)
        compare(v["red_flag_mean"], stats["conditions"]["red_flag"][new]["pooled_mean"], f"interference/{c}/rf",
                problems)
    lb, d = leak["lora+bodhi"], old["decisive_pass1_headers"]
    compare({"n_clean": d["n_clean"], "n_leaked": d["n_leaked"]}, lb, "interference/decisive", problems)
    compare([d["lora_bodhi_clean_red_flag"], d["lora_bodhi_leaked_red_flag"]],
            [lb["dimensions_clean"]["red_flag"], lb["dimensions_leaked"]["red_flag"]], "interference/decisive",
            problems)
    assert not problems, "\n".join(problems)


def test_the_committed_stats_are_current(stats):
    """results/<cell>/analysis/stats.json is what analyze writes from the tracked tables."""
    committed = json.loads((REPO_ROOT / "results" / CELL / "analysis" / "stats.json").read_text())
    assert json.loads(json.dumps(stats, sort_keys=True)) == committed


CROSS = REPO_ROOT / "results" / "cross_cell"


def _cell_stats(cid):
    return json.loads((REPO_ROOT / "results" / cid / "analysis" / "stats.json").read_text())


def _cell_of_host_dir():
    """The host folder of each added cell (results_rebuttal/<run> on the GPU host) -> its cell
    id, through the posted cell names that generated_values.json and the report config share."""
    cfg = report_config()
    gv = json.loads((CROSS / "generated_values.json").read_text())
    return {c["dir"]: cfg.posted.cells[c["id"]] for c in gv["cells"]}


def test_per_seed_means_of_generated_values():
    """generated_values.json (make cells, removed) held the per-seed means that Table 1 and
    the per-cell tables printed until 2026-10-10; scripts/paper_stats.py stopped on any
    difference from stats.json. Every per-seed mean agrees two decimals finer than the paper
    prints (the file keeps six significant digits)."""
    cfg = report_config()
    gv = json.loads((CROSS / "generated_values.json").read_text())
    problems, n = [], 0
    for g in gv["cells"]:
        stats = _cell_stats(cfg.posted.cells[g["id"]])["conditions"]
        for gc, cond in cfg.posted.conditions.items():
            for name, m in cfg.posted.metrics.items():
                ours = stats[m.metric][cond]["per_seed"]
                for seed, v in g["conditions"][gc][name]["per_seed"].items():
                    w = ours.get(str(seed))
                    n += 1
                    if (v is None) != (w is None) or (v is not None and abs(v - w * m.scale) > 10 ** -(m.decimals + 2)):
                        problems.append(f"{g['id']} {gc} {name} seed {seed}: {v} against {w}")
    assert n > 1000 and not problems, "\n".join(problems[:20])


def test_aggregate_tests_of_quality_drop():
    """quality_drop.json (removed script) tested LoRA - Base on the aggregate score per cell."""
    q = json.loads((CROSS / "quality_drop.json").read_text())
    cells = _cell_of_host_dir()
    problems = []
    for host, r in q["cells"].items():
        cid = cells[f"results_rebuttal/{host}"]
        new = _cell_stats(cid)["contrasts"]["lora - base"]["metrics"]["rubric_score"]
        compare({"diff": r["diff"], "ci95": r["ci95"], "p": r["p"], "df": r["df"]}, new, cid, problems)
    assert len(q["cells"]) >= 6 and not problems, "\n".join(problems)


def test_filter_and_recipe_of_cell_recipe():
    """cell_recipe.json (make cells) read tau and the trace counts from the host logs and the
    recipe from the adapter files; the filter manifests and stats.json carry the same."""
    rec = json.loads((CROSS / "cell_recipe.json").read_text())
    cells = _cell_of_host_dir()
    problems, n = [], 0
    for host, r in rec["cells"].items():
        cid = cells.get(r["dir"])
        if cid is None or r["status"] != "complete":
            continue
        s = _cell_stats(cid)
        n += 1
        compare({k: r["filter"][k] for k in ("tau", "graded", "kept", "train", "val")}, s["filter"]["recorded"],
                f"{cid}/filter", problems)
        compare({k: r["adapter"][k] for k in ("r", "lora_alpha", "lora_dropout", "num_train_epochs",
                                              "train_batch_size", "max_steps")}, s["recipe"], f"{cid}/recipe", problems)
        if sorted(r["adapter"]["target_modules"]) != s["recipe"]["target_modules"]:
            problems.append(f"{cid}: target modules")
        compare({"grad_accumulation_consistent": r["grad_accumulation_consistent"],
                 "effective_batch_consistent": r["effective_batch_consistent"]}, s["recipe"], f"{cid}/batch", problems)
    assert n >= 7 and not problems, "\n".join(problems)
