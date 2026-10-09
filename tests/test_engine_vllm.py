"""The vLLM engine's serve command and client, and the stub engine. Nothing is started."""
import json

import pytest

pytest.importorskip("pydantic")

from bodhi_lora.engine import StubEngine, make_engine, placement_from_env  # noqa: E402
from bodhi_lora.engine import vllm as ve  # noqa: E402
from bodhi_lora.registry.schema import ModelConfig, Serving  # noqa: E402

SERVING = Serving(max_model_len=4096, max_num_seqs=32, concurrency=32, request_timeout_s=900)
MODEL = ModelConfig(id="base-24b", hf_id="org/base-24b", revision="a" * 40, family="mistral",
                    domain="general", params_b=24, license="test")


@pytest.fixture(autouse=True)
def _no_placement(monkeypatch):
    monkeypatch.delenv("BODHI_VLLM_PORT", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_serving_settings_revision_and_quantization_reach_the_command(tmp_path):
    adapter = (tmp_path / "adapter").resolve()
    eng = make_engine(MODEL, serving=SERVING, quantize=True, adapter=adapter)
    cmd = eng.serve_args()
    assert cmd[1:3] == ["serve", "org/base-24b"]
    assert _flag(cmd, "--max-model-len") == "4096"
    assert _flag(cmd, "--max-num-seqs") == "32"
    assert _flag(cmd, "--revision") == "a" * 40
    assert _flag(cmd, "--tokenizer-revision") == "a" * 40
    assert _flag(cmd, "--quantization") == "bitsandbytes"
    assert _flag(cmd, "--dtype") == "bfloat16"
    assert _flag(cmd, "--lora-modules") == f"adapter={adapter}"
    assert "--enforce-eager" in cmd
    s = eng.settings()
    assert s["request_timeout_s"] == 900 and s["quantization"] == "bitsandbytes"
    assert s["lora_path"] == str(adapter)


def test_unpinned_full_precision_base_has_no_revision_or_quantization_flags():
    unpinned = MODEL.model_copy(update={"revision": None, "unpinned_reason": "test"})
    cmd = make_engine(unpinned, serving=SERVING, enforce_eager=False).serve_args()
    for flag in ("--revision", "--quantization", "--enable-lora", "--enforce-eager"):
        assert flag not in cmd


def test_unknown_quantization_is_refused():
    with pytest.raises(ValueError):
        ve.VLLMEngine("m", max_model_len=1, max_num_seqs=1, request_timeout=1, quantization="awq")


def test_runner_placement_sets_port_and_tensor_parallel(monkeypatch):
    assert placement_from_env() == (8000, 1)
    monkeypatch.setenv("BODHI_VLLM_PORT", "8003")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    assert placement_from_env() == (8003, 2)
    cmd = make_engine(MODEL, serving=SERVING).serve_args()
    assert _flag(cmd, "--port") == "8003" and _flag(cmd, "--tensor-parallel-size") == "2"


def test_complete_reads_tokens_and_finish_reason(monkeypatch):
    eng = make_engine(MODEL, serving=SERVING)
    sent = {}

    def fake_post(payload, timeout=None):
        sent.update(payload)
        return {"choices": [{"message": {"content": "hello"}, "finish_reason": "length",
                             "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.0}]}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2}}

    monkeypatch.setattr(eng, "_post", fake_post)
    done = eng.complete([{"role": "user", "content": "hi"}], max_new_tokens=2, seed=7, logprobs=True)
    assert (done.text, done.prompt_tokens, done.completion_tokens) == ("hello", 12, 2)
    assert done.finish_length and done.token_logprobs == [-0.5, -1.0]
    assert sent["max_tokens"] == 2 and sent["seed"] == 7 and sent["top_logprobs"] == 1
    assert sent["model"] == "org/base-24b"


def test_stub_engine_records_calls_and_supports_reply_functions():
    eng = make_engine(MODEL, serving=SERVING, stub=True,
                      stub_reply=lambda msgs: msgs[-1]["content"].upper())
    assert isinstance(eng, StubEngine)
    with eng as e:
        assert e.chat([{"role": "user", "content": "hi"}], max_new_tokens=8) == "HI"
        text, logprobs = e.chat_with_logprobs([{"role": "user", "content": "x y"}], max_new_tokens=8)
        long = e.complete([{"role": "user", "content": "a b c"}], max_new_tokens=2)
    assert (text, logprobs) == ("X Y", [])
    assert long.finish_length and long.completion_tokens == 3
    assert len(eng.calls) == 3
    s = eng.settings()
    assert s["stub"] is True and s["max_model_len"] == 4096
    json.dumps(s)
