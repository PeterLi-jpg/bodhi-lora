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


def test_cuda_graphs_run_unless_an_adapter_is_loaded_or_a_record_says_otherwise(tmp_path):
    """Eager mode only with LoRA (graph capture with LoRA took about 2 hours in vLLM 0.9);
    the record's serving.enforce_eager overrides that, and the bench's own setting wins."""
    adapter = tmp_path / "adapter"
    assert "--enforce-eager" not in make_engine(MODEL, serving=SERVING).serve_args()
    assert "--enforce-eager" in make_engine(MODEL, serving=SERVING, adapter=adapter).serve_args()
    eager = SERVING.model_copy(update={"enforce_eager": True})
    graphs = SERVING.model_copy(update={"enforce_eager": False})
    assert "--enforce-eager" in make_engine(MODEL, serving=eager).serve_args()
    assert "--enforce-eager" not in make_engine(MODEL, serving=graphs, adapter=adapter).serve_args()
    assert "--enforce-eager" in make_engine(MODEL, serving=graphs, enforce_eager=True).serve_args()
    # The stub records the same resolved setting for the manifest.
    assert make_engine(MODEL, serving=SERVING, stub=True).settings()["enforce_eager"] is False
    assert make_engine(MODEL, serving=SERVING, adapter=adapter, stub=True).settings()["enforce_eager"] is True
    # Built directly, the engine follows the same rule.
    assert ve.VLLMEngine("m", max_model_len=1, max_num_seqs=1, request_timeout=1).enforce_eager is False
    assert ve.VLLMEngine("m", max_model_len=1, max_num_seqs=1, request_timeout=1,
                         lora_path=str(adapter)).enforce_eager is True


def test_a_model_records_serving_env_reaches_its_server_only(monkeypatch, tmp_path):
    """Gemma 2 needs FlashAttention 2 on an H100 (run notes 2026-10-11, R1). The record's
    serving_env goes into that server's process and its manifest settings, not into this
    process and not into another model's server."""
    from bodhi_lora.paths import CONFIGS_DIR
    from bodhi_lora.registry import load_registry

    monkeypatch.delenv("VLLM_FLASH_ATTN_VERSION", raising=False)
    reg = load_registry(CONFIGS_DIR)
    for gemma2 in ("gemma-2-27b", "gemma-2-9b"):
        assert reg.models[gemma2].serving_env == {"VLLM_FLASH_ATTN_VERSION": "2"}
    judge = make_engine(reg.models["gemma-2-27b"], serving=SERVING, log_path=tmp_path / "judge.log")
    assert judge.server_env()["VLLM_FLASH_ATTN_VERSION"] == "2"
    assert judge.settings()["env"] == {"VLLM_FLASH_ATTN_VERSION": "2"}
    assert make_engine(reg.models["gemma-2-27b"], serving=SERVING, stub=True).settings()["env"] == {
        "VLLM_FLASH_ATTN_VERSION": "2"}
    assert "VLLM_FLASH_ATTN_VERSION" not in make_engine(MODEL, serving=SERVING).server_env()
    assert "VLLM_FLASH_ATTN_VERSION" not in make_engine(reg.models["medgemma-27b-text"], serving=SERVING).server_env()
    # The record wins over a launcher that set another value for every server.
    monkeypatch.setenv("VLLM_FLASH_ATTN_VERSION", "3")
    assert judge.server_env()["VLLM_FLASH_ATTN_VERSION"] == "2"
    monkeypatch.delenv("VLLM_FLASH_ATTN_VERSION")

    started = {}

    class FakePopen:
        pid = 12345

        def __init__(self, args, env=None, **kwargs):
            started.update(args=args, env=env)

    monkeypatch.setattr(ve.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(judge, "_wait_ready", lambda: None)
    judge.start()
    judge._proc = None
    assert started["env"]["VLLM_FLASH_ATTN_VERSION"] == "2" and started["args"] == judge.serve_args()
    assert "VLLM_FLASH_ATTN_VERSION" not in ve.os.environ


def test_serving_env_takes_only_vllm_variables():
    with pytest.raises(ValueError):
        MODEL.model_validate({**MODEL.model_dump(), "serving_env": {"CUDA_VISIBLE_DEVICES": "0"}})


def test_one_server_serves_the_base_and_named_adapters(tmp_path):
    """The shared eval server: every seed's adapter as a named LoRA module, as many LoRA
    slots as adapters, and the rank the adapters need (vLLM 0.10.2 flags)."""
    a7, a42 = (tmp_path / "s7").resolve(), (tmp_path / "s42").resolve()
    eng = make_engine(MODEL, serving=SERVING, adapters={"adapter-s7": a7, "adapter-s42": a42},
                      lora_rank=16, quantize=True)
    cmd = eng.serve_args()
    i = cmd.index("--lora-modules")
    assert cmd[i + 1:i + 3] == [f"adapter-s7={a7}", f"adapter-s42={a42}"]
    assert cmd[i + 3].startswith("--")
    assert "--enable-lora" in cmd and "--enforce-eager" in cmd
    assert _flag(cmd, "--max-loras") == "2" and _flag(cmd, "--max-lora-rank") == "16"
    assert _flag(cmd, "--quantization") == "bitsandbytes"
    assert "--max-cpu-loras" not in cmd
    s = eng.settings()
    assert s["lora_modules"] == {"adapter-s7": str(a7), "adapter-s42": str(a42)}
    assert s["max_loras"] == 2 and s["max_lora_rank"] == 16 and s["lora_path"] is None
    # One adapter needs no --max-loras (the default is 1); a rank vLLM does not list is
    # rounded up to one it does.
    one = make_engine(MODEL, serving=SERVING, adapter=a7, lora_rank=4).serve_args()
    assert "--max-loras" not in one and _flag(one, "--max-lora-rank") == "8"
    assert "--max-lora-rank" not in make_engine(MODEL, serving=SERVING, lora_rank=16).serve_args()
    with pytest.raises(ValueError, match="above"):
        make_engine(MODEL, serving=SERVING, adapter=a7, lora_rank=1024)
    with pytest.raises(ValueError, match="not both"):
        ve.VLLMEngine("m", max_model_len=1, max_num_seqs=1, request_timeout=1, lora_path=str(a7),
                      lora_modules={"x": str(a42)})
    with pytest.raises(ValueError, match="base model's name"):
        make_engine(MODEL, serving=SERVING, adapters={MODEL.hf_id: a7})


def test_a_route_sends_its_model_name_and_unknown_names_are_refused(monkeypatch, tmp_path):
    eng = make_engine(MODEL, serving=SERVING, adapters={"adapter-s7": tmp_path / "s7"})
    sent = []

    def fake_post(payload, timeout=None):
        sent.append(payload["model"])
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(eng, "_post", fake_post)
    msgs = [{"role": "user", "content": "hi"}]
    assert eng.route(None).chat(msgs, max_new_tokens=4) == "ok"
    eng.route("adapter-s7").complete(msgs, max_new_tokens=4, seed=1, logprobs=True)
    eng.complete(msgs, max_new_tokens=4)
    assert sent == ["org/base-24b", "adapter-s7", "org/base-24b"]
    with pytest.raises(ValueError, match="no adapter"):
        eng.route("adapter-s99")
    # With one adapter (the per-condition path) requests still go to it by default.
    single = make_engine(MODEL, serving=SERVING, adapter=tmp_path / "s7")
    monkeypatch.setattr(single, "_post", fake_post)
    single.complete(msgs, max_new_tokens=4)
    assert sent[-1] == ve.ADAPTER_NAME


def test_the_stub_serves_the_same_names_and_records_each_calls_route(tmp_path):
    eng = make_engine(MODEL, serving=SERVING, stub=True, adapters={"adapter-s7": tmp_path / "s7"})
    msgs = [{"role": "user", "content": "hi"}]
    eng.route("adapter-s7").chat(msgs, max_new_tokens=4)
    eng.route(None).complete(msgs, max_new_tokens=4)
    assert eng.routes == ["adapter-s7", "org/base-24b"] and len(eng.calls) == 2
    with pytest.raises(ValueError):
        eng.route("adapter-s42")
    with pytest.raises(ValueError, match="does not exist"):
        eng.complete(msgs, max_new_tokens=4, model="adapter-s42")
    single = make_engine(MODEL, serving=SERVING, stub=True, adapter=tmp_path / "s7")
    single.chat(msgs, max_new_tokens=4)
    assert single.routes == [ve.ADAPTER_NAME]


def test_an_adapter_that_does_not_answer_stops_the_start(monkeypatch, tmp_path):
    """After the base answers, each named adapter gets a one-token request, so a broken
    adapter fails at start and not hours into the eval."""
    eng = make_engine(MODEL, serving=SERVING, adapters={"adapter-s7": tmp_path / "s7",
                                                        "adapter-s42": tmp_path / "s42"})
    asked = []

    def fake_post(payload, timeout=None):
        asked.append(payload["model"])
        if payload["model"] == "adapter-s42":
            raise RuntimeError("HTTP 404")
        return {}

    monkeypatch.setattr(eng, "_post", fake_post)
    with pytest.raises(ve.AdapterError, match="adapter-s42"):
        eng._check_adapters()
    assert asked == ["adapter-s7", "adapter-s42"]


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


def test_raw_continuation_routes_send_token_ids_and_no_chat_template(monkeypatch):
    eng = make_engine(MODEL, serving=SERVING)
    sent = []

    def fake_post(payload, timeout=None, route="/v1/chat/completions"):
        sent.append((route, payload))
        if route == "/tokenize":
            return {"tokens": [5, 6, 7], "count": 3, "max_model_len": 4096}
        if route == "/detokenize":
            return {"prompt": "a b"}
        return {"choices": [{"text": " c d", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr(eng, "_post", fake_post)
    assert eng.tokenize("a b c") == [5, 6, 7]
    assert eng.detokenize([5, 6]) == "a b"
    done = eng.continue_ids([5, 6, 7], max_new_tokens=2)
    assert done.text == " c d" and done.finish_length and done.prompt_tokens == 3
    routes = [r for r, _ in sent]
    assert routes == ["/tokenize", "/detokenize", "/v1/completions"]
    assert sent[0][1]["add_special_tokens"] is False
    assert sent[2][1]["prompt"] == [5, 6, 7] and "messages" not in sent[2][1]


def test_stub_tokenizer_round_trips_and_continues_raw_text():
    eng = make_engine(MODEL, serving=SERVING, stub=True, stub_reply=lambda m: m[-1]["content"].upper())
    ids = eng.tokenize("the cat saw the dog")
    assert ids[0] == ids[3] and eng.detokenize(ids) == "the cat saw the dog"
    assert eng.continue_ids(ids[:2], max_new_tokens=5).text == "THE CAT"


def test_a_context_length_refusal_is_its_own_error(monkeypatch):
    """vLLM 0.9.2 answers HTTP 400 with "maximum context length" when prompt plus
    max_tokens exceed max_model_len; the stages skip such items instead of retrying."""
    import io
    import urllib.error

    eng = make_engine(MODEL, serving=SERVING)

    def refuse(body, code=400):
        def urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "Bad Request", {}, io.BytesIO(body.encode()))
        return urlopen

    msg = ('{"object": "error", "message": "This model\'s maximum context length is 4096 tokens. '
           'However, you requested 5000 tokens", "type": "BadRequestError", "code": 400}')
    monkeypatch.setattr(ve.urllib.request, "urlopen", refuse(msg))
    with pytest.raises(ve.ContextLengthError):
        eng.complete([{"role": "user", "content": "hi"}], max_new_tokens=8)
    monkeypatch.setattr(ve.urllib.request, "urlopen", refuse('{"message": "bad model name"}'))
    with pytest.raises(ve.urllib.error.HTTPError):
        eng.complete([{"role": "user", "content": "hi"}], max_new_tokens=8)
    monkeypatch.setattr(ve.urllib.request, "urlopen", refuse(msg, code=500))
    with pytest.raises(ve.urllib.error.HTTPError) as err:
        eng.complete([{"role": "user", "content": "hi"}], max_new_tokens=8)
    assert not isinstance(err.value, ve.ContextLengthError)


class _Proc:
    """A stand-in for the server process: returncode None while it runs."""

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _serving_engine(monkeypatch, answer, max_num_seqs=4):
    """An engine whose server is up (as after start), with urlopen replaced by answer."""
    eng = ve.VLLMEngine("org/m", max_model_len=4096, max_num_seqs=max_num_seqs, request_timeout=1)
    eng._proc, eng._serving = _Proc(), True
    monkeypatch.setattr(ve.urllib.request, "urlopen", answer)
    return eng


def _ok_body():
    import io
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())


def test_a_hung_server_gets_no_more_requests_after_every_slot_failed(monkeypatch):
    """A timed-out request is counted; once max_num_seqs fail in a row with no answer in
    between, every later request fails at once and none is sent."""
    sent = []

    def hung(req, timeout=None):
        sent.append(req.full_url)
        raise TimeoutError("timed out")

    eng = _serving_engine(monkeypatch, hung)
    msgs = [{"role": "user", "content": "hi"}]
    for _ in range(4):
        with pytest.raises(TimeoutError):
            eng.complete(msgs, max_new_tokens=4)
    with pytest.raises(ve.ServerDown, match="4 requests in a row failed"):
        eng.route(None).complete(msgs, max_new_tokens=4)
    with pytest.raises(ve.ServerDown):
        eng.tokenize("a b")
    assert len(sent) == 4


def test_an_answer_resets_the_count_and_a_bad_request_does_not_count(monkeypatch):
    import io
    import urllib.error

    replies = iter(["timeout", "timeout", "timeout", "ok", "timeout", "timeout", "timeout", "400", "ok"])

    def mixed(req, timeout=None):
        what = next(replies)
        if what == "timeout":
            raise TimeoutError("timed out")
        if what == "400":
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"message": "x"}'))
        return _ok_body()

    eng = _serving_engine(monkeypatch, mixed)
    msgs = [{"role": "user", "content": "hi"}]
    results = []
    for _ in range(9):
        try:
            results.append(eng.chat(msgs, max_new_tokens=4))
        except Exception as e:
            results.append(type(e).__name__)
    assert results[3] == "ok" and results[-1] == "ok" and "ServerDown" not in results


def test_a_server_that_exited_gets_no_more_requests(monkeypatch):
    import urllib.error

    def refused(req, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    eng = _serving_engine(monkeypatch, refused, max_num_seqs=64)
    eng._proc = _Proc(returncode=-9)
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(urllib.error.URLError):
        eng.complete(msgs, max_new_tokens=4)
    with pytest.raises(ve.ServerDown, match="exited with code -9"):
        eng.complete(msgs, max_new_tokens=4)


def test_failures_while_the_server_starts_are_not_counted(monkeypatch):
    """The start polls until the chat route answers; those failures are expected."""
    def refused(req, timeout=None):
        raise ConnectionRefusedError(111, "Connection refused")

    eng = _serving_engine(monkeypatch, refused, max_num_seqs=1)
    eng._serving = False
    for _ in range(3):
        with pytest.raises(ConnectionRefusedError):
            eng.complete([{"role": "user", "content": "hi"}], max_new_tokens=1)
    assert eng._down is None
