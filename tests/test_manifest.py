"""Run manifests (bodhi_lora/manifest.py). CPU only."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bodhi_lora import manifest as rm  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_sha256_json_ignores_key_order():
    assert rm.sha256_json({"a": 1, "b": [1, 2]}) == rm.sha256_json({"b": [1, 2], "a": 1})
    assert rm.sha256_json({"a": 1}) != rm.sha256_json({"a": 2})


def test_id_list_digest_is_format_and_order_independent(tmp_path):
    ids = ["p3", "p1", "p2"]
    (tmp_path / "list.json").write_text(json.dumps(ids))
    (tmp_path / "dict.json").write_text(json.dumps({"prompt_ids": sorted(ids)}))
    (tmp_path / "rows.jsonl").write_text(
        "".join(json.dumps({"prompt_id": i}) + "\n" for i in ids))
    digests = [rm.id_list_digest(tmp_path / n) for n in ("list.json", "dict.json", "rows.jsonl")]
    assert {d["ids_sha256"] for d in digests} == {digests[0]["ids_sha256"]}
    assert all(d["n"] == 3 for d in digests)
    # The file hash still tells the three files apart.
    assert len({d["file_sha256"] for d in digests}) == 3


def test_adapter_digest_reads_shape_and_hashes_weights(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps(
        {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights-v1")
    first = rm.adapter_digest(tmp_path)
    assert first["r"] == 8 and first["lora_alpha"] == 16
    assert set(first["files"]) == {"adapter_config.json", "adapter_model.safetensors"}
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights-v2")
    assert rm.adapter_digest(tmp_path)["sha256"] != first["sha256"]
    assert rm.adapter_digest(tmp_path / "missing") == {
        "path": str(tmp_path / "missing"), "exists": False}


def test_git_state_on_this_repo_and_outside_git(tmp_path):
    state = rm.git_state(REPO)
    assert SHA_RE.match(state["sha"])
    assert isinstance(state["dirty"], bool)
    assert rm.git_state(tmp_path)["sha"] is None


def test_git_state_keeps_the_commit_when_status_times_out(monkeypatch):
    real = rm._git

    def slow_status(args, repo_root):
        if args[0] == "status":
            raise rm.subprocess.TimeoutExpired(["git", *args], 60)
        return real(args, repo_root)

    monkeypatch.setattr(rm, "_git", slow_status)
    state = rm.git_state(REPO)
    assert SHA_RE.match(state["sha"])
    assert state["dirty"] is None and state["error"].startswith("TimeoutExpired")


def test_build_manifest_records_models_hardware_time_and_hashes(tmp_path):
    reg = load_registry(CONFIGS_DIR)
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["a", "b"]))
    timer = rm.Timer()
    m = rm.build_manifest(
        "eval", cell="c", seed=7, config={"max_model_len": 4096},
        models={"grader": reg.models["llama-3.1-8b"], "judge": reg.models["meditron3-8b"]},
        eval_ids=ids, counts={"n": 2}, timer=timer, extra={"status": "complete"})
    assert m["schema"] == 3 and (m["cell"], m["seed"], m["status"]) == ("c", 7, "complete")
    assert m["config_sha256"] == rm.sha256_json({"max_model_len": 4096})
    assert len(m["host_sha256"]) == 64 and "hostname" not in json.dumps(m)
    grader = m["models"]["grader"]
    assert (grader["id"], grader["family"], grader["revision_source"]) == ("llama-3.1-8b", "llama3", "record")
    assert m["models"]["judge"]["revision_source"] == "unpinned" and m["models"]["judge"]["unpinned_reason"]
    assert m["eval_ids"]["n"] == 2 and m["time"]["seconds"] >= 0
    assert set(m["hardware"]) == {"visible_devices", "gpus"}
    assert set(m["packages"]) >= {"python", "torch", "transformers", "vllm", "peft"}
    path = rm.write_manifest(tmp_path / "out" / "manifest.json", m)
    assert json.loads(path.read_text())["stage"] == "eval"


def test_another_interpreter_reports_the_same_versions_as_this_one():
    """The runner reads a stage env's versions through that env's interpreter."""
    import sys
    assert rm.interpreter_package_versions(sys.executable) == rm.package_versions()


def test_a_caller_can_record_the_env_that_did_the_work():
    m = rm.build_manifest("eval", packages={"python": "3.12.3", "vllm": "0.10.2"})
    assert m["packages"] == {"python": "3.12.3", "vllm": "0.10.2"}
