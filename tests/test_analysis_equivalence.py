"""The uniform analysis reproduces the outputs it supersedes (docs/ARCHITECTURE.md 10.3).

It recomputes the 2026-05 cell from its tracked response tables and compares, at the
stored precision, with the files the old scripts wrote: paper/generated/numbers.json
(scripts/paper_stats.py) and the cell's mimicry_split.json, interference_mechanism.json
and second_person_questions.json (rebuttal/analyses/). A mismatch is a finding to
explain, not a tolerance to raise. Old names map to new ones in the tables below; a
field the new output names differently is listed there, not skipped.

Two old values are not compared, for a stated reason:
  - interference_mechanism length_control_clean[3].len_median: the old script took the
    upper middle element of an even count (5395); the new code takes the median (5393.5).
  - mimicry_split groups: the old script counted labels over the base condition's scored
    prompts, so a prompt with no scored base reply showed as null; the rates match.
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
    numbers = json.loads((REPO_ROOT / "paper/generated/numbers.json").read_text())
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
    return json.loads((REPO_ROOT / "results" / CELL / "analysis" / name).read_text())


def test_rebuttal_analyses_of_the_cell(stats):
    problems = []
    old = _analysis("mimicry_split.json")["primary_theme_or_rubric"]
    disc = stats["discrimination"]["needs-info-theme-or-rubric"]
    for c, v in old["configs"].items():
        compare({k: v[k] for k in ("discrimination", "discrimination_ci95", "odds_ratio", "odds_ratio_ci95")},
                disc["conditions"][CONDITION[c]], f"mimicry_split/{c}", problems)
        compare({"positive_rate": v["context_dependent_rate"], "negative_rate": v["self_contained_rate"]},
                disc["conditions"][CONDITION[c]], f"mimicry_split/{c}", problems)
    compare({"diff": old["lora_minus_base_disc"], "diff_ci95": old["lora_minus_base_disc_ci95"]},
            disc["contrasts"]["lora - base"], "mimicry_split/lora_minus_base", problems)

    old = _analysis("interference_mechanism.json")
    leak = stats["leakage"]["leak_bodhi__analysis-format"]["conditions"]
    headers = stats["leakage"]["leak_bodhi__pass1-headers"]["conditions"]
    for c, v in old["configs"].items():
        new = CONDITION[c]
        compare({"n_scored": v["n"], "leaked_share": v["leaked_rate"]}, leak[new], f"interference/{c}", problems)
        compare(v["leaked_rate_pass1_headers"], headers[new]["leaked_share"], f"interference/{c}/headers",
                problems)
        compare(v["red_flag_mean"], stats["conditions"]["red_flag"][new]["pooled_mean"], f"interference/{c}/rf",
                problems)
    lb, d = leak["lora+bodhi"], old["decisive"]
    compare({"n_clean": d["n_clean"], "n_leaked": d["n_leaked"], "chars_median_clean": d["len_median_clean"],
             "chars_median_leaked": d["len_median_leaked"]}, lb, "interference/decisive", problems)
    compare([d["lora_bodhi_clean_red_flag"], d["lora_bodhi_leaked_red_flag"]],
            [lb["dimensions_clean"]["red_flag"], lb["dimensions_leaked"]["red_flag"]], "interference/decisive",
            problems)
    for i, q in enumerate(old["length_control_clean"]):
        new = lb["clean_red_flag_by_length_quartile"][i]
        compare({"n": q["n"], "red_flag": q["red_flag"]}, new, f"interference/quartile{i + 1}", problems)
        if i < 3:   # see the module docstring for the fourth median
            compare(q["len_median"], new["chars_median"], f"interference/quartile{i + 1}/median", problems)

    old = _analysis("second_person_questions.json")
    for c, v in old["conditions"].items():
        new = stats["conditions"]["second_person_q"][CONDITION[c]]
        compare({"rate": v["rate"], "n": v["n"], "per_seed_rate": v["per_seed_rate"]},
                {"rate": new["pooled_mean"], "n": new["n_pooled"], "per_seed_rate": new["per_seed"]},
                f"second_person/{c}", problems)
    lmb = stats["contrasts"]["lora - base"]["metrics"]["second_person_q"]
    compare({k: old["lora_minus_base"][k] for k in ("diff", "ci95", "t", "df", "p")}, lmb,
            "second_person/lora_minus_base", problems)
    assert not problems, "\n".join(problems)


def test_the_committed_stats_are_current(stats):
    """results/<cell>/analysis/stats.json is what analyze writes from the tracked tables."""
    committed = json.loads((REPO_ROOT / "results" / CELL / "analysis" / "stats.json").read_text())
    assert json.loads(json.dumps(stats, sort_keys=True)) == committed
