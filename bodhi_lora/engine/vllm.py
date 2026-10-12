"""vLLM's OpenAI-compatible server as a subprocess, and the client that talks to it.

    with VLLMEngine("org/model", revision=sha, max_model_len=8192, max_num_seqs=64,
                    request_timeout=180) as engine:
        text = engine.chat(messages, max_new_tokens=1024)
        done = engine.complete(messages, max_new_tokens=1024)  # text, tokens, finish reason

The engine starts `vllm serve` from the environment of the running interpreter (the
inference venv, requirements-infer.txt), waits until the chat route answers, and stops
the whole process group on exit, also after an exception. Every serving setting is a
constructor argument, so a run never patches this file, and settings() returns them for
the run manifest. bodhi_lora.engine.make_engine() builds an engine from a model record.

Where a stage runs is the runner's choice: it sets CUDA_VISIBLE_DEVICES and
BODHI_VLLM_PORT for each stage (placement_from_env), so stages on different GPUs never
share a port.

The response fields follow vLLM 0.9.2's OpenAI protocol
(vllm/entrypoints/openai/protocol.py at tag v0.9.2): choices[0].message.content,
choices[0].finish_reason ("length" when the token limit stopped the reply),
usage.prompt_tokens and usage.completion_tokens.

The contamination probe needs raw text, not a chat turn: a chat template would wrap the
prefix in a user message, and the model would answer it instead of continuing it. For
that the engine also has tokenize(), detokenize() and continue_ids(), on the server's
/tokenize, /detokenize and /v1/completions routes (vllm/entrypoints/openai/api_server.py
and protocol.py at tag v0.9.2: TokenizeCompletionRequest with add_special_tokens,
DetokenizeRequest, and CompletionRequest, whose prompt may be a list of token ids).
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# The server serves an adapter under this name; requests for the adapter use it.
ADAPTER_NAME = "adapter"
QUANTIZATIONS = (None, "bitsandbytes")


class ContextLengthError(ValueError):
    """The server refused a request that does not fit its context window.

    A retry fails the same way, so a stage records such an item as too long and skipped
    instead of asking for a rerun. vLLM 0.9.2 answers HTTP 400 with "This model's maximum
    context length is N tokens. However, you requested M tokens ..." when prompt tokens
    plus max_tokens exceed max_model_len (OpenAIServing._validate_input in
    vllm/entrypoints/openai/serving_engine.py at tag v0.9.2).
    """


@dataclass(frozen=True)
class Completion:
    """One chat completion: the text and what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    # "length" when the token limit stopped the reply, "stop" otherwise.
    finish_reason: Optional[str]
    token_logprobs: List[float] = field(default_factory=list)

    @property
    def finish_length(self) -> bool:
        return self.finish_reason == "length"


def placement_from_env() -> Tuple[int, int]:
    """(port, tensor-parallel size) of this stage, as the runner placed it.

    The port is BODHI_VLLM_PORT (8000 when unset). The tensor-parallel size is the
    number of GPUs in CUDA_VISIBLE_DEVICES (1 when unset), so a stage given two GPUs
    shards the model over both.
    """
    port = int(os.environ.get("BODHI_VLLM_PORT", "8000"))
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    return port, max(1, len(visible))


class VLLMEngine:
    """A `vllm serve` subprocess and a client for its chat route."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int,
        max_num_seqs: int,
        request_timeout: int,
        revision: Optional[str] = None,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        lora_path: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        port: Optional[int] = None,
        tensor_parallel: Optional[int] = None,
        log_path: Optional[str] = None,
    ):
        if quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization={quantization!r}: use None or 'bitsandbytes'")
        env_port, env_tp = placement_from_env()
        self.model = model
        self.max_model_len = max_model_len
        # Scheduler slots. KV-cache memory grows with max_num_seqs * max_model_len, so
        # the two trade against each other on one GPU.
        self.max_num_seqs = max_num_seqs
        self.request_timeout = request_timeout
        # HF commit of the served weights and tokenizer. None loads the repo HEAD.
        self.revision = revision
        self.dtype = dtype
        # "bitsandbytes" quantizes the base to 4-bit NF4 with double quantization while
        # loading (vLLM 0.9.2 bitsandbytes_loader.py calls quantize_4bit with
        # compress_statistics=True and quant_type="nf4"), the base QLoRA training used.
        self.quantization = quantization
        self.lora_path = os.path.realpath(lora_path) if lora_path else None
        # None: eager mode only when an adapter is loaded. With --enable-lora vLLM 0.9
        # captured about 67 graph shapes at about 2 minutes each before the first
        # request; eager mode is a little slower per token and much faster for a
        # 200-prompt eval. With no adapter the server captures CUDA graphs: --enforce-eager
        # is off by default in vLLM 0.10.2, and graphs are captured for batch sizes up to
        # min(2 * max_num_seqs, 512) (docs.vllm.ai/en/v0.10.2/cli/serve.html,
        # --enforce-eager and --cuda-graph-sizes).
        self.enforce_eager = bool(self.lora_path) if enforce_eager is None else enforce_eager
        self.port = port if port is not None else env_port
        self.tensor_parallel = tensor_parallel if tensor_parallel is not None else env_tp
        self.log_path = log_path or os.path.expanduser(f"~/vllm_serve_{self.port}.log")
        self._proc: Optional[subprocess.Popen] = None

    # -- the server ---------------------------------------------------------------

    def serve_args(self) -> List[str]:
        """The `vllm serve` command line."""
        # The vllm binary next to this interpreter, so the server runs in the same
        # environment even when that environment is not on PATH.
        vllm = os.path.join(os.path.dirname(sys.executable), "vllm")
        if not os.path.exists(vllm):
            vllm = "vllm"
        return [
            vllm, "serve", self.model,
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--port", str(self.port),
            "--max-num-seqs", str(self.max_num_seqs),
            # Pin the weights and the tokenizer, so an upstream push cannot change a rerun.
            *(["--revision", self.revision, "--tokenizer-revision", self.revision]
              if self.revision else []),
            *(["--quantization", self.quantization] if self.quantization else []),
            *(["--enforce-eager"] if self.enforce_eager else []),
            *(["--enable-lora", "--lora-modules", f"{ADAPTER_NAME}={self.lora_path}"]
              if self.lora_path else []),
        ]

    def settings(self) -> dict:
        """Serving settings for the run manifest."""
        return {
            "model": self.model,
            "revision": self.revision,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": self.tensor_parallel,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "enforce_eager": self.enforce_eager,
            "request_timeout_s": self.request_timeout,
            "lora_path": self.lora_path,
        }

    def start(self) -> "VLLMEngine":
        print(f"Starting vllm serve: {self.model} (TP={self.tensor_parallel}, port={self.port}"
              f"{', adapter' if self.lora_path else ''})", flush=True)
        # vLLM is chatty; its log goes to a file, and _wait_ready reports progress.
        log_fh = open(self.log_path, "w")
        env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"}
        # A new process group, so stop() also ends the engine's worker processes.
        self._proc = subprocess.Popen(self.serve_args(), stdout=log_fh, stderr=subprocess.STDOUT,
                                      env=env, preexec_fn=os.setsid)
        print(f"  pid={self._proc.pid}, log={self.log_path}", flush=True)
        self._wait_ready()
        return self

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), 15)
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self._proc.pid), 9)
            except ProcessLookupError:
                pass
        self._proc = None
        print("  vllm serve stopped", flush=True)

    def _wait_ready(self, timeout_s: int = 2700) -> None:
        # /health answers 200 as soon as the HTTP server is up, before the chat route
        # is registered: early requests then fail fast with 404. So poll /health, then
        # send a one-token chat request until it succeeds.
        health_url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout_s
        last_log = time.time()
        health_ok = False
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {self._proc.returncode}; "
                                   f"see {self.log_path}")
            try:
                if not health_ok:
                    with urllib.request.urlopen(health_url, timeout=5) as r:
                        health_ok = r.status == 200
                if health_ok:
                    self._post({"model": self._served_name(),
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 1, "temperature": 0.0}, timeout=30)
                    print(f"  vllm serve ready ({int(time.time() - deadline + timeout_s)}s)", flush=True)
                    return
            except Exception:
                pass
            if time.time() - last_log > 30:
                state = "chat route" if health_ok else "starting"
                print(f"  waiting for vllm serve [{state}]... "
                      f"({int(time.time() - deadline + timeout_s)}s)", flush=True)
                last_log = time.time()
            time.sleep(5)
        raise RuntimeError(f"vllm serve did not become ready within {timeout_s}s; see {self.log_path}")

    def __enter__(self) -> "VLLMEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- the client ---------------------------------------------------------------

    def _served_name(self) -> str:
        return ADAPTER_NAME if self.lora_path else self.model

    def _post(self, payload: dict, timeout: Optional[int] = None,
              route: str = "/v1/chat/completions") -> dict:
        req = urllib.request.Request(
            f"http://localhost:{self.port}{route}",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "maximum context length" in body:
                raise ContextLengthError(body) from e
            raise

    def tokenize(self, text: str) -> List[int]:
        """Token ids of raw text with the served tokenizer, without special tokens."""
        resp = self._post({"model": self._served_name(), "prompt": text, "add_special_tokens": False},
                          route="/tokenize")
        return list(resp["tokens"])

    def detokenize(self, ids: List[int]) -> str:
        return self._post({"model": self._served_name(), "tokens": list(ids)}, route="/detokenize")["prompt"]

    def continue_ids(self, prompt_ids: List[int], max_new_tokens: int,
                     temperature: float = 0.0) -> Completion:
        """A raw continuation of token ids, with no chat template."""
        resp = self._post({"model": self._served_name(), "prompt": list(prompt_ids),
                           "max_tokens": max_new_tokens, "temperature": temperature},
                          route="/v1/completions")
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        return Completion(text=choice["text"], prompt_tokens=int(usage.get("prompt_tokens") or 0),
                          completion_tokens=int(usage.get("completion_tokens") or 0),
                          finish_reason=choice.get("finish_reason"))

    def complete(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 logprobs: bool = False) -> Completion:
        """One chat completion, with its token counts and finish reason."""
        payload = {"model": self._served_name(), "messages": messages,
                   "max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}
        if seed is not None:
            payload["seed"] = seed
        if logprobs:
            payload.update(logprobs=True, top_logprobs=1)
        resp = self._post(payload)
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        lp = [e["logprob"] for e in ((choice.get("logprobs") or {}).get("content") or [])
              if e.get("logprob") is not None]
        return Completion(text=choice["message"]["content"],
                          prompt_tokens=int(usage.get("prompt_tokens") or 0),
                          completion_tokens=int(usage.get("completion_tokens") or 0),
                          finish_reason=choice.get("finish_reason"), token_logprobs=lp)

    def chat(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0) -> str:
        return self.complete(messages, max_new_tokens, temperature, top_p).text

    def chat_with_logprobs(self, messages: list, max_new_tokens: int,
                           temperature: float = 0.0) -> Tuple[str, List[float]]:
        """The reply and the log-probability of each generated token, from one call."""
        done = self.complete(messages, max_new_tokens, temperature, logprobs=True)
        return done.text, done.token_logprobs
