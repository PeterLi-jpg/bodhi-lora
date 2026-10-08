"""Dry-run tests for rebuttal/launch/run_cell.sh and run_baselines.sh.

--dry-run needs no GPU, no HF token and no download: GPU commands are printed,
and trace generation and eval run with stub engines on the CPU. The fixture is
a small ChatDoctor-format benchmark file under a temporary DATA_RAW.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bodhi")
pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from build_benchmark_jsonl import chatdoctor_row, stable_id  # noqa: E402

MISTRAL = ("mistralai/Mistral-Small-24B-Instruct-2501",
           "rebuttal/configs/lora_mistral_small_24b_qlora.yaml")
MED42 = ("m42-health/Llama3-Med42-8B", "rebuttal/configs/lora_med42_8b.yaml")
TAG = {MISTRAL[0]: "mistralai_Mistral-Small-24B-Instruct-2501",
       MED42[0]: "m42-health_Llama3-Med42-8B"}


def _bench_rows(n=80):
    rows = []
    for i in range(n):
        q, user, rubrics, tags = chatdoctor_row(
            {"input": f"I have had a cough for {i} days. What should I do?", "output": f"reply {i}"})
        rows.append({"prompt_id": stable_id("chatdoctor", q), "prompt": [{"role": "user", "content": user}],
                     "rubrics": rubrics, "example_tags": tags, "_source": "chatdoctor"})
    return sorted(rows, key=lambda r: r["prompt_id"])


@pytest.fixture
def env(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    text = "".join(json.dumps(r) + "\n" for r in _bench_rows())
    # Default profile name: <bench>_s<SELECT_SEED>_n<GEN_MAX + HOLDOUT_N + 500>.
    (raw / "chatdoctor_s0_n530.jsonl").write_text(text)
    # Posted profile name (head-of-file selection).
    (raw / "chatdoctor.jsonl").write_text(text)
    py = sys.executable
    base = {**os.environ, "BENCH": "chatdoctor", "SEEDS": "42", "GEN_MAX": "20",
            "HOLDOUT_N": "10", "EVAL_N": "4", "RESULTS_ROOT": str(tmp_path / "results"),
            "DATA_RAW": str(raw), "LOG_DIR": str(tmp_path / "logs"),
            "INFER_PY": py, "TRAIN_PY": py}
    base.pop("HF_TOKEN", None)
    return tmp_path, base


def _run(script, env, model, *args, **extra):
    e = {**env, "MODEL": model[0], "CONFIG": model[1], **extra}
    res = subprocess.run(["bash", str(REPO / "rebuttal" / "launch" / script), *args],
                         capture_output=True, text=True, env=e, cwd=REPO, timeout=600)
    return res


def _work(tmp, model, suffix=""):
    return tmp / "results" / f"chatdoctor__{TAG[model[0]]}{suffix}"


def test_default_cell_dry_run_writes_manifests_and_submitted_settings(env):
    tmp, e = env
    res = _run("run_cell.sh", e, MISTRAL, "--dry-run")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    work = _work(tmp, MISTRAL)
    cell = json.loads((work / "manifest.json").read_text())
    s = cell["settings"]
    assert s["STATUS"] == "finished" and s["PROFILE"] == "submitted"
    assert (s["MAX_MODEL_LEN_EVAL"], s["MAX_NUM_SEQS_EVAL"]) == (4096, 32)
    assert s["GRADER_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert s["TAU_RULE"] == "fixed" and s["MIN_SCORE"] == 0.4
    assert "eval prompt draw" in s["SEED_VARIES"]
    assert cell["includes"]["family"]["passed"] is True
    assert cell["holdout_ids"]["n"] == 10
    assert cell["config"]["lora"]["r"] == 16
    # Per-seed eval ids come from the held-out pool.
    pool = set(json.loads((tmp / "raw" / "chatdoctor_s0_n530_holdout10_h0.json").read_text())["prompt_ids"])
    seed_ids = set(json.loads((work / "eval_ids" / "seed_42.json").read_text())["prompt_ids"])
    assert len(seed_ids) == 4 and seed_ids <= pool
    # Generation ran with the stub engine and excluded the pool.
    traces = (work / "sft" / "raw_traces.dryrun.jsonl").read_text().splitlines()
    assert traces and not {json.loads(t)["prompt_id"] for t in traces} & pool
    ev = work / "seed_42" / "eval"
    for name in ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"):
        out = json.loads((ev / f"{name}.dryrun.json").read_text())
        assert out["config"] == name
        assert out["run_metadata"]["max_model_len"] == 4096
        assert out["run_metadata"]["eval_ids_n"] == 4
    assert (ev / "epistemic_scores.dryrun.json").exists()
    # Training and filtering are GPU stages: printed, not run.
    train_log = (tmp / "logs" / f"train_chatdoctor_{TAG[MISTRAL[0]]}_s42.log").read_text()
    assert train_log.startswith("DRY-RUN [gpu 0]:") and "train_lora.py" in train_log
    assert "DRY-RUN [gpu 0]:" in res.stdout and "filter_traces.py" in res.stdout


def test_family_guard_refuses_med42_unless_waived(env):
    tmp, e = env
    res = _run("run_cell.sh", e, MED42, "--dry-run")
    assert res.returncode == 3
    assert "base and evaluator are both llama3" in res.stderr
    assert not (_work(tmp, MED42) / "manifest.json").exists()

    res = _run("run_cell.sh", e, MED42, "--dry-run", "--allow-same-family")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    cell = json.loads((_work(tmp, MED42) / "manifest.json").read_text())
    assert cell["settings"]["ALLOW_SAME_FAMILY"] == 1
    assert cell["includes"]["family"]["waived"] is True


def test_posted_profile_reproduces_the_posted_settings(env):
    tmp, e = env
    res = _run("run_cell.sh", e, MISTRAL, "--dry-run", PROFILE="rebuttal-posted", HOLDOUT_N="6")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    work = _work(tmp, MISTRAL)
    s = json.loads((work / "manifest.json").read_text())["settings"]
    assert (s["MAX_MODEL_LEN_EVAL"], s["MAX_NUM_SEQS_EVAL"]) == (8192, 64)
    assert s["GRADER_MODEL"] == "NousResearch/Meta-Llama-3.1-8B-Instruct"
    assert s["TAU_RULE"] == "legacy-auto" and s["EVAL_IDS_POLICY"] == "fixed"
    # The posted holdout: the first N prompt_ids of the benchmark file.
    rows = (tmp / "raw" / "chatdoctor.jsonl").read_text().splitlines()
    held = json.loads((work / "eval_ids.json").read_text())["prompt_ids"]
    assert held == [json.loads(r)["prompt_id"] for r in rows][:6]
    out = json.loads((work / "seed_42" / "eval" / "base_no_wrapper.dryrun.json").read_text())
    assert out["run_metadata"]["max_model_len"] == 8192
    assert out["run_metadata"]["eval_ids_n"] == 6


def test_second_protocol_cell_is_one_setting(env):
    tmp, e = env
    res = _run("run_cell.sh", e, MISTRAL, "--dry-run", PROTOCOL="ask_first")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    work = _work(tmp, MISTRAL, "__ask_first")
    traces = [json.loads(t) for t in (work / "sft" / "raw_traces.dryrun.jsonl").read_text().splitlines()]
    assert {t["protocol"] for t in traces} == {"ask_first"}
    ev = work / "seed_42" / "eval"
    assert json.loads((ev / "base_ask_first.dryrun.json").read_text())["wrapper"] == "ask_first"
    assert json.loads((ev / "lora_ask_first.dryrun.json").read_text())["config"] == "lora_ask_first"


def test_qlora_cell_can_be_evaluated_on_the_matching_4bit_base(env):
    tmp, e = env
    res = _run("run_cell.sh", e, MISTRAL, "--dry-run", EVAL_BASE_QUANT="match")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    ev = _work(tmp, MISTRAL) / "seed_42" / "eval"
    for name in ("base_no_wrapper", "lora_bodhi"):
        md = json.loads((ev / f"{name}.dryrun.json").read_text())["run_metadata"]
        assert md["base_quantization"] == "bitsandbytes"

