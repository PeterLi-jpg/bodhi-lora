"""Tests for rebuttal/analyses/cell_recipe.py (PROV-3).

The real inputs are the host logs (gitignored) and the tracked adapter files in the cell
folders of the results root. The test builds a small results root in the same layout,
with log lines in the format bodhi_lora/pipeline/filter.py prints.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "cell_recipe", ROOT / "rebuttal" / "analyses" / "cell_recipe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FILTER_LOG = (
    "Grading:   0%|  | 0/4200\rGrading: 100%|##| 4200/4200\n"
    "  parse failures: 1/33600 (0.0%)\n"
    "Kept 3294/4200 (score_field=overall_score, threshold=0.4)\n"
    "train: 2965 -> results_rebuttal/medquad__m/sft/train.jsonl\n"
    "--- summary ---\n"
    "raw=4200 graded=4200 kept=3294 train=2965 val=329\n"
)


def _seed(cell, seed, best_step, max_steps=558, batch=2):
    best = cell / f"seed_{seed}" / "train"
    best.mkdir(parents=True)
    (best / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "org/model", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj"], "peft_version": "0.19.1"}))
    (best / "trainer_state.json").write_text(json.dumps({
        "max_steps": max_steps, "num_train_epochs": 3, "train_batch_size": batch,
        "best_global_step": best_step}))


def _logs(cell, **files):
    logs = cell / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (logs / name).write_text(text)


def _provenance(root, *host_folders):
    """The move's record: one old file in each run folder the host wrote."""
    entries = [{"old": f"results_rebuttal/{f}/eval_ids.json"} for f in host_folders]
    entries.append({"old": "logs/host/grid.log"})
    (root / "provenance.json").write_text(json.dumps(entries))


def test_tau_counts_and_recipe_are_read_and_checked(tmp_path):
    cr = _load()
    root = tmp_path / "results"
    cell = root / "medquad__m__bodhi__2026-07"
    _logs(cell, **{"gen_medquad_m.log": "Generating\n", "filter_medquad_m.log": FILTER_LOG})
    for s, step in zip((7, 13, 42, 99, 101), (558, 558, 372, 558, 186)):
        _seed(cell, s, step)
    (cell / "shared" / "inputs").mkdir(parents=True)
    (cell / "shared" / "inputs" / "eval_ids.json").write_text(json.dumps({"prompt_ids": ["a", "b"]}))
    # A run that never trained: only eval ids and a failed generation log.
    _logs(root / "medqa__g__bodhi__2026-07",
          **{"gen_medqa_g.log": "RuntimeError: vllm serve did not become healthy\n"})
    # A run stopped before it wrote a folder on the host: only its generation log and the
    # launcher's log, which holds this run's section and then another run's.
    _logs(root / "healthbench__m-9b__bodhi__2026-07", **{
        "gen_healthbench_org_m-9b.log": "Generating 4000 traces\n  waiting (300s)\n\n",
        "launch.log": "=== cell: healthbench x org/m-9b  (seeds: 42 7) ===\n  waiting (300s)\n"
                      "run_cell.sh: line 103: 1 Killed   generate_traces.py\n"
                      "=== cell: medquad x m  (seeds: 42) ===\nrun_cell.sh: line 103: 2 Killed   other\n"})
    # A cell the package ran has no host logs and is not a host run.
    (root / "tiny__t__bodhi" / "seed_1" / "train").mkdir(parents=True)
    _provenance(root, "medquad__m", "medqa__g")

    res = cr.run(root, tmp_path / "out.json")
    c = res["cells"]["medquad__m"]
    assert sorted(res["cells"]) == ["medqa__g", "medquad__m"]
    assert c["status"] == "complete" and c["seeds"] == [7, 13, 42, 99, 101]
    assert c["dir"].endswith("results/medquad__m__bodhi__2026-07")
    assert c["filter"]["tau"] == 0.4 and c["filter"]["kept"] == 3294 and c["filter"]["train"] == 2965
    assert c["filter"]["rubric_parse_failures"] == [1, 33600]
    assert c["adapter"]["r"] == 16 and c["adapter"]["target_modules"] == ["k_proj", "q_proj", "v_proj"]
    # 2,965 traces at 2 per device give 1,483 batches; 186 steps per epoch fit only 8.
    assert c["grad_accumulation_consistent"] == [8] and c["effective_batch_consistent"] == [16]
    assert c["best_epoch"] == {"7": 3.0, "13": 3.0, "42": 2.0, "99": 3.0, "101": 1.0}
    assert c["n_eval_ids"] == 2
    g = res["cells"]["medqa__g"]
    assert g["status"] == "not run" and g["filter"] is None
    assert g["generation_error"].startswith("RuntimeError")
    assert res["refilter_lines"] == []
    # 2,965 of 4,200 raw traces in train is 70%, above the launcher's 40% re-filter mark.
    assert c["filter"]["train_pct_of_raw"] == pytest.approx(70.595, abs=1e-3)
    assert c["refilter_rule_met"] is False and g["refilter_rule_met"] is None
    assert list(res["runs_without_folder"]) == ["healthbench__org_m-9b"]
    w = res["runs_without_folder"]["healthbench__org_m-9b"]
    assert w["status"] == "no folder" and w["gen_log_last_line"] == "waiting (300s)"
    assert w["killed_lines"] == ["launch.log: run_cell.sh: line 103: 1 Killed   generate_traces.py"]
    assert json.loads((tmp_path / "out.json").read_text()) == res


def test_refilter_rule_follows_the_launcher(tmp_path):
    cr = _load()
    f = {"parsed": True, "raw": 4200, "train": 1679}
    # Shell integer arithmetic: 1,679 * 100 / 4,200 = 39 (rule met), 1,680 gives 40 (not met).
    assert cr.refilter_rule_met("chatdoctor", f) is True
    assert cr.refilter_rule_met("chatdoctor", {**f, "train": 1680}) is False
    # The launcher never re-filters HealthBench.
    assert cr.refilter_rule_met("healthbench", f) is False
    assert cr.refilter_rule_met("medqa", None) is None


def test_seeds_that_disagree_on_the_recipe_stop_the_run(tmp_path):
    cr = _load()
    root = tmp_path / "results"
    cell = root / "healthbench__m__bodhi__2026-07"
    _logs(cell, **{"gen_healthbench_m.log": "Generating\n"})
    _seed(cell, 7, 140)
    _seed(cell, 13, 140, batch=4)
    _provenance(root, "healthbench__m")
    with pytest.raises(SystemExit, match="train_batch_size"):
        cr.run(root, tmp_path / "out.json")


def test_a_root_without_host_logs_stops(tmp_path):
    cr = _load()
    (tmp_path / "results").mkdir()
    with pytest.raises(SystemExit, match="no host logs"):
        cr.run(tmp_path / "results", tmp_path / "out.json")
