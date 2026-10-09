"""Tests for bodhi_lora/manifest.py and configs/families.yaml. CPU only."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bodhi_lora import manifest as rm  # noqa: E402

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


def test_manifest_path_for_files_and_dirs(tmp_path):
    assert rm.manifest_path_for(tmp_path / "eval" / "base_bodhi.json") == \
        tmp_path / "eval" / "base_bodhi.manifest.json"
    assert rm.manifest_path_for(tmp_path / "sft" / "raw_traces.jsonl") == \
        tmp_path / "sft" / "raw_traces.manifest.json"
    assert rm.manifest_path_for(tmp_path) == tmp_path / "manifest.json"
    assert rm.manifest_path_for(tmp_path / "ckpt") == tmp_path / "ckpt" / "manifest.json"


def test_git_state_on_this_repo_and_outside_git(tmp_path):
    state = rm.git_state(REPO)
    assert SHA_RE.match(state["sha"])
    assert isinstance(state["dirty"], bool)
    assert rm.git_state(tmp_path)["sha"] is None


def test_build_manifest_hashes_host_and_config(tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["a", "b"]))
    m = rm.build_manifest(
        "eval", config={"max_model_len": 4096},
        models={"grader": ("meta-llama/Llama-3.1-8B-Instruct", None)},
        eval_ids=ids, counts={"n": 2})
    assert m["config_sha256"] == rm.sha256_json({"max_model_len": 4096})
    assert len(m["host_sha256"]) == 64
    assert "hostname" not in json.dumps(m)
    grader = m["models"]["grader"]
    assert grader["family"] == "llama3"
    assert grader["revision_source"] == "registry" and grader["revision_is_commit"]
    assert m["eval_ids"]["n"] == 2
    assert set(m["packages"]) >= {"python", "torch", "transformers", "vllm", "peft"}
    path = rm.write_manifest(tmp_path / "out" / "manifest.json", m)
    assert json.loads(path.read_text())["stage"] == "eval"


def test_registry_entries_are_complete():
    reg = rm.load_registry()
    for repo_id, entry in reg["models"].items():
        assert entry["family"], repo_id
        rev = entry.get("revision")
        assert rev is None or SHA_RE.match(rev), repo_id
    for role, repo_id in reg["roles"].items():
        assert repo_id in reg["models"], role


def test_effective_revision_prefers_flag_and_skips_local_paths(tmp_path):
    pinned = rm.pinned_revision("Qwen/Qwen2.5-14B-Instruct")
    assert rm.effective_revision("Qwen/Qwen2.5-14B-Instruct") == pinned
    assert rm.effective_revision("Qwen/Qwen2.5-14B-Instruct", "main") == "main"
    assert rm.effective_revision(str(tmp_path)) is None
    assert rm.effective_revision("someone/unlisted-model") is None


def test_family_guard_refuses_same_family_and_records_waiver():
    filt, ev = "Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"
    ok = rm.family_check("mistralai/Mistral-Small-24B-Instruct-2501", filt, ev)
    assert ok["passed"] and not ok["waived"]
    with pytest.raises(rm.FamilyError, match="base and evaluator are both llama3"):
        rm.family_check("m42-health/Llama3-Med42-8B", filt, ev)
    waived = rm.family_check("m42-health/Llama3-Med42-8B", filt, ev, allow_same_family=True)
    assert waived["waived"] and not waived["passed"]
    with pytest.raises(rm.FamilyError, match="filter and evaluator"):
        rm.family_check("BioMistral/BioMistral-7B", filt, "Qwen/Qwen2.5-14B-Instruct")
    with pytest.raises(rm.FamilyError, match="no family"):
        rm.family_check("someone/unlisted-model", filt, ev)


def test_system_role_flags():
    assert rm.supports_system_role("BioMistral/BioMistral-7B") is False
    assert rm.supports_system_role("mistralai/Mistral-Small-24B-Instruct-2501") is True


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "bodhi_lora.manifest", *args], cwd=REPO,
                          capture_output=True, text=True)


def test_cli_check_family_exit_codes(tmp_path):
    base = ["check-family", "--filter", "Qwen/Qwen2.5-14B-Instruct",
            "--evaluator", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
    refused = _cli(*base, "--base", "m42-health/Llama3-Med42-8B")
    assert refused.returncode == 3 and "REFUSED" in refused.stderr
    out = tmp_path / "family.json"
    waived = _cli(*base, "--base", "m42-health/Llama3-Med42-8B",
                  "--allow-same-family", "--json-out", str(out))
    assert waived.returncode == 0 and "WAIVED" in waived.stdout
    assert json.loads(out.read_text())["waived"] is True


def test_cli_write_embeds_settings_and_includes(tmp_path):
    inc = tmp_path / "filter.json"
    inc.write_text(json.dumps({"counts": {"n_kept": 3}}))
    out = tmp_path / "cell" / "manifest.json"
    res = _cli("write", "--stage", "cell", "--out", str(out),
               "--config", str(REPO / "rebuttal/configs/lora_med42_8b.yaml"),
               "--model", "base=m42-health/Llama3-Med42-8B",
               "--model", "evaluator=NousResearch/Meta-Llama-3.1-8B-Instruct@abc",
               "--set", "TAU=0.4", "--set", "BENCH=healthbench",
               "--include", f"filter={inc}", "--include", "missing=/nonexistent.json")
    assert res.returncode == 0, res.stderr
    m = json.loads(out.read_text())
    assert m["settings"] == {"TAU": 0.4, "BENCH": "healthbench"}
    assert m["includes"]["filter"]["counts"]["n_kept"] == 3
    assert m["includes"]["missing"]["exists"] is False
    assert m["config"]["lora"]["r"] == 16
    assert m["models"]["evaluator"]["revision"] == "abc"
    assert m["models"]["evaluator"]["revision_source"] == "flag"
    assert m["models"]["base"]["revision_source"] == "registry"
