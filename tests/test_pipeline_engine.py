"""Serve-command tests for scripts/_vllm_engine.py. Nothing is started."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import _vllm_engine as ve  # noqa: E402


@pytest.fixture(autouse=True)
def _gpu_subprocess_env(monkeypatch):
    # Fix the mode detection so the test never probes docker or nvidia-smi.
    monkeypatch.setenv("BODHI_VLLM_MODE", "subprocess")
    monkeypatch.setenv("BODHI_VLLM_ACCEL", "gpu")
    monkeypatch.setenv("BODHI_VLLM_TP", "1")
    monkeypatch.delenv("BODHI_VLLM_PORT", raising=False)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_defaults_match_generation_settings():
    cmd = ve.VLLMEngine("google/medgemma-27b-text-it")._build_subprocess_cmd()
    assert _flag(cmd, "--max-model-len") == "8192"
    assert _flag(cmd, "--max-num-seqs") == "64"
    assert "--revision" not in cmd and "--quantization" not in cmd


def test_eval_settings_revision_and_quantization_reach_the_command():
    eng = ve.VLLMEngine("mistralai/Mistral-Small-24B-Instruct-2501",
                        max_model_len=4096, max_num_seqs=32, revision="abc123",
                        quantization="bitsandbytes", request_timeout=900)
    for cmd in (eng._build_subprocess_cmd(), eng._build_docker_cmd()):
        assert _flag(cmd, "--max-model-len") == "4096"
        assert _flag(cmd, "--max-num-seqs") == "32"
        assert _flag(cmd, "--revision") == "abc123"
        assert _flag(cmd, "--tokenizer-revision") == "abc123"
        assert _flag(cmd, "--quantization") == "bitsandbytes"
    s = eng.settings()
    assert s["max_model_len"] == 4096 and s["max_num_seqs"] == 32
    assert s["revision"] == "abc123" and s["request_timeout_s"] == 900
    assert s["quantization"] == "bitsandbytes" and s["run_mode"] == "subprocess"


def test_docker_image_is_pinned():
    cmd = ve.VLLMEngine("m", revision=None)._build_docker_cmd()
    assert "vllm/vllm-openai:v0.9.2" in cmd


def test_unknown_quantization_is_refused():
    with pytest.raises(ValueError):
        ve.VLLMEngine("m", quantization="awq")


def test_dry_run_engine_records_calls_and_supports_reply_functions():
    eng = ve.DryRunEngine("judge", reply=lambda msgs: msgs[-1]["content"].upper(),
                          max_model_len=8192)
    with eng as e:
        assert e.chat([{"role": "user", "content": "hi"}]) == "HI"
        text, logprobs = e.chat_with_logprobs([{"role": "user", "content": "x"}])
    assert text == "X" and logprobs == []
    assert len(eng.calls) == 2
    assert eng.settings() == {"model": "judge", "dry_run": True, "max_model_len": 8192}
