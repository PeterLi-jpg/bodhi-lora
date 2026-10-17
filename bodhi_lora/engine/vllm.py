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

A server that stops answering (its process exited, or every request it held failed)
gets no more requests: each later one fails at once with ServerDown (_failed). The
stage then ends with those items failed and its manifest incomplete, and a rerun starts
a new server.

Where a stage runs is the runner's choice: it sets CUDA_VISIBLE_DEVICES and
BODHI_VLLM_PORT for each stage (placement_from_env), so stages on different GPUs never
share a port.

The response fields follow vLLM 0.10.2's OpenAI protocol
(vllm/entrypoints/openai/protocol.py at tag v0.10.2): choices[0].message.content,
choices[0].finish_reason ("length" when the token limit stopped the reply),
usage.prompt_tokens and usage.completion_tokens.

The contamination probe needs raw text, not a chat turn: a chat template would wrap the
prefix in a user message, and the model would answer it instead of continuing it. For
that the engine also has tokenize(), detokenize() and continue_ids(), on the server's
/tokenize, /detokenize and /v1/completions routes (vllm/entrypoints/openai/api_server.py
and protocol.py at tag v0.10.2: TokenizeCompletionRequest with add_special_tokens,
DetokenizeRequest, and CompletionRequest, whose prompt may be a list of token ids).

One server can serve the base and several adapters at once, each adapter a named LoRA
module (lora_modules). A request picks its weights by the "model" field: the base
model's name or a module's name (docs.vllm.ai/en/v0.10.2/features/lora.html, "Serving
LoRA Adapters"). route(name) returns a client whose requests carry that name:

    with VLLMEngine("org/model", ..., lora_modules={"adapter-s7": path}) as engine:
        base_reply = engine.route(None).chat(messages, max_new_tokens=1024)
        lora_reply = engine.route("adapter-s7").chat(messages, max_new_tokens=1024)
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# A server with one adapter (lora_path) serves it under this name, and requests go to it.
ADAPTER_NAME = "adapter"
QUANTIZATIONS = (None, "bitsandbytes")
# How long start() waits for its port to be free: the server of the stage's previous
# engine on this port may still be shutting down (stop() waits 15 s, then kills).
PORT_FREE_WAIT_S = 60
# The values vLLM 0.10.2 accepts for --max-lora-rank (possible_max_ranks in
# vllm/config/lora.py at tag v0.10.2); the server refuses any other.
LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def lora_rank_flag(rank: int) -> int:
    """The smallest --max-lora-rank vLLM accepts that holds adapters of this rank."""
    for allowed in LORA_RANKS:
        if allowed >= rank:
            return allowed
    raise ValueError(f"LoRA rank {rank} is above vLLM's largest --max-lora-rank {LORA_RANKS[-1]}")


class ContextLengthError(ValueError):
    """The server refused a request that does not fit its context window.

    A retry fails the same way, so a stage records such an item as too long and skipped
    instead of asking for a rerun. vLLM 0.10.2 answers HTTP 400 with a message that holds
    "This model's maximum context length is N tokens" when the prompt alone, or prompt
    tokens plus max_tokens, exceed max_model_len (OpenAIServing._validate_input in
    vllm/entrypoints/openai/serving_engine.py at tag v0.10.2; 0.9.2 used the same words).
    """


class AdapterError(RuntimeError):
    """A named adapter of a running server does not answer."""


class ServerDown(RuntimeError):
    """The server stopped answering, so the engine sends it no more requests.

    vLLM 0.10.2 answers HTTP 500 once its engine has died and then shuts the HTTP server
    down (terminate_if_errored in vllm/entrypoints/launcher.py at tag v0.10.2), so the
    requests after it fail with a refused connection. A hung engine answers nothing, and
    each request waits out its timeout.
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
        lora_modules: Optional[dict] = None,
        lora_rank: Optional[int] = None,
        enforce_eager: Optional[bool] = None,
        port: Optional[int] = None,
        tensor_parallel: Optional[int] = None,
        log_path: Optional[str] = None,
        env: Optional[dict] = None,
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
        # loading (vllm/model_executor/model_loader/bitsandbytes_loader.py at tag v0.10.2
        # calls quantize_4bit with compress_statistics=True and quant_type="nf4"), the base
        # QLoRA training used.
        self.quantization = quantization
        if lora_path and lora_modules:
            raise ValueError("give lora_path (one adapter) or lora_modules (named adapters), not both")
        self.lora_path = os.path.realpath(lora_path) if lora_path else None
        # Adapter name -> folder. One adapter (lora_path) is served as ADAPTER_NAME and
        # requests go to it; with lora_modules requests go to the base unless routed.
        modules = {ADAPTER_NAME: self.lora_path} if self.lora_path else dict(lora_modules or {})
        self.lora_modules = {name: os.path.realpath(path) for name, path in modules.items()}
        if self.model in self.lora_modules:
            raise ValueError(f"an adapter may not take the base model's name {self.model!r}")
        # The LoRA rank the server must hold, rounded up to a value vLLM accepts. The
        # LoRA docs page above: max_lora_rank "should match the highest rank among your
        # adapters"; vLLM's default is 16.
        self.max_lora_rank = lora_rank_flag(lora_rank) if self.lora_modules and lora_rank else None
        # None: eager mode only when an adapter is loaded. With --enable-lora vLLM 0.9
        # captured about 67 graph shapes at about 2 minutes each before the first
        # request; eager mode is a little slower per token and much faster for a
        # 200-prompt eval. With no adapter the server captures CUDA graphs: --enforce-eager
        # is off by default in vLLM 0.10.2, and graphs are captured for batch sizes up to
        # min(2 * max_num_seqs, 512) (docs.vllm.ai/en/v0.10.2/cli/serve.html,
        # --enforce-eager and --cuda-graph-sizes).
        self.enforce_eager = bool(self.lora_modules) if enforce_eager is None else enforce_eager
        self.port = port if port is not None else env_port
        self.tensor_parallel = tensor_parallel if tensor_parallel is not None else env_tp
        self.log_path = log_path or os.path.expanduser(f"~/vllm_serve_{self.port}.log")
        # Environment variables this model's server needs (the model record's
        # serving_env), set in the server process only, never in this one.
        self.env = dict(env or {})
        self._proc: Optional[subprocess.Popen] = None
        # Failed requests in a row, and why the engine stopped sending (see _failed).
        # Counted only once the server is up, so the start's own polling never counts.
        self._lock = threading.Lock()
        self._serving = False
        self._failed_in_row = 0
        self._down: Optional[str] = None

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
            *(self._lora_args() if self.lora_modules else []),
        ]

    def _lora_args(self) -> List[str]:
        """--enable-lora and the adapters, per docs.vllm.ai/en/v0.10.2/cli/serve.html.

        --lora-modules takes one or more name=path values (nargs "+" for a list field in
        vllm/engine/arg_utils.py at tag v0.10.2). --max-loras is the number of adapters
        that may run in one batch, default 1; with several adapters at 1, requests for
        different adapters wait for each other, so it is the number of adapters. Base
        requests take no adapter slot. --max-cpu-loras is left out: it defaults to
        --max-loras (vllm/config/lora.py), and every adapter then stays on the GPU.
        --max-lora-rank: see max_lora_rank above.
        """
        args = ["--enable-lora", "--lora-modules", *(f"{name}={path}" for name, path in self.lora_modules.items())]
        if len(self.lora_modules) > 1:
            args += ["--max-loras", str(len(self.lora_modules))]
        if self.max_lora_rank:
            args += ["--max-lora-rank", str(self.max_lora_rank)]
        return args

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
            "lora_modules": self.lora_modules,
            "max_loras": len(self.lora_modules) or None,
            "max_lora_rank": self.max_lora_rank,
            "env": self.env,
        }

    def server_env(self) -> dict:
        """The environment of the server process: this process's, a quieter vLLM log, and
        last the model record's serving_env, so the record wins for this server only.

        Example: VLLM_FLASH_ATTN_VERSION=2 for Gemma 2. On an H100 vLLM 0.10.2 picks
        FlashAttention 3, which has no tanh softcapping; the variable forces version 2
        (get_flash_attn_version in vllm/attention/utils/fa_utils.py at tag v0.10.2).
        """
        return {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING", **self.env}

    def start(self) -> "VLLMEngine":
        print(f"Starting vllm serve: {self.model} (TP={self.tensor_parallel}, port={self.port}"
              f"{', adapters ' + ', '.join(self.lora_modules) if self.lora_modules else ''})", flush=True)
        self._wait_port_free()
        # vLLM is chatty; its log goes to a file, and _wait_ready reports progress.
        log_fh = open(self.log_path, "w")
        self._serving, self._failed_in_row, self._down = False, 0, None
        # A new process group, so stop() also ends the engine's worker processes.
        self._proc = subprocess.Popen(self.serve_args(), stdout=log_fh, stderr=subprocess.STDOUT,
                                      env=self.server_env(), preexec_fn=os.setsid)
        print(f"  pid={self._proc.pid}, log={self.log_path}", flush=True)
        self._wait_ready()
        self._serving = True
        return self

    def stop(self) -> None:
        self._serving = False
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

    def _port_answers(self) -> bool:
        try:
            with socket.create_connection(("localhost", self.port), timeout=2):
                return True
        except OSError:
            return False

    def _wait_port_free(self, timeout_s: Optional[int] = None) -> None:
        """Refuse to start while something answers on the port.

        vllm serve binds its port before it loads the model, and exits when the port is
        taken (setup_server in vllm/entrypoints/openai/api_server.py at tag v0.10.2). Until
        it exits, _wait_ready would take a server left on the port (one that outlived the
        stage that started it) for this one, and the stage would send its requests to
        other weights or settings than its manifest records.
        """
        deadline = time.time() + (PORT_FREE_WAIT_S if timeout_s is None else timeout_s)
        while self._port_answers():
            if time.time() >= deadline:
                raise RuntimeError(f"port {self.port} is in use: a server there answers, perhaps a vllm "
                                   f"serve that outlived its stage (see nvidia-smi). Stop it, then run "
                                   f"the stage again.")
            time.sleep(2)

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
                    self._check_adapters()
                    return
            except AdapterError:
                raise
            except Exception:
                pass
            if time.time() - last_log > 30:
                state = "chat route" if health_ok else "starting"
                print(f"  waiting for vllm serve [{state}]... "
                      f"({int(time.time() - deadline + timeout_s)}s)", flush=True)
                last_log = time.time()
            time.sleep(5)
        raise RuntimeError(f"vllm serve did not become ready within {timeout_s}s; see {self.log_path}")

    def _check_adapters(self) -> None:
        """One one-token request to every named adapter once the server is up, so an
        adapter the server cannot load stops the stage now, not hours into the run."""
        for name in self.lora_modules:
            if name == self._served_name():
                continue
            try:
                self._post({"model": name, "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1, "temperature": 0.0}, timeout=300)
            except Exception as e:
                raise AdapterError(f"adapter {name} ({self.lora_modules[name]}) did not answer: {e}; "
                                   f"see {self.log_path}") from e

    def __enter__(self) -> "VLLMEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- the client ---------------------------------------------------------------

    def _served_name(self) -> str:
        return ADAPTER_NAME if self.lora_path else self.model

    def route(self, name: Optional[str]) -> "Route":
        """A client whose requests go to one served model: name None is the base, any
        other name a LoRA module of this server."""
        return Route(self, check_route(self.model, self.lora_modules, name))

    def _post(self, payload: dict, timeout: Optional[int] = None,
              route: str = "/v1/chat/completions") -> dict:
        if self._down is not None:
            raise ServerDown(self._down)
        req = urllib.request.Request(
            f"http://localhost:{self.port}{route}",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "maximum context length" in body:
                raise ContextLengthError(body) from e
            # A 4xx answer is about this request; a 5xx is about the server.
            if e.code >= 500:
                self._failed(e)
            raise
        except (OSError, http.client.HTTPException) as e:
            # No answer: the connection was refused, reset or timed out.
            self._failed(e)
            raise
        with self._lock:
            self._failed_in_row = 0
        return out

    def _failed(self, err: Exception) -> None:
        """Count a request the server did not answer, once the server is up.

        The engine stops sending when the server process has exited, or when max_num_seqs
        requests in a row failed: as many as the server runs at once, so each request it
        held failed and none came back. Every later request then fails at once with
        ServerDown, and the stage ends with those items failed, instead of waiting out a
        timeout for each one (900 s per eval request: days for a hung eval server).
        """
        if not self._serving:
            return
        with self._lock:
            self._failed_in_row += 1
            if self._down is not None:
                return
            exited = self._proc is not None and self._proc.poll() is not None
            if not exited and self._failed_in_row < self.max_num_seqs:
                return
            why = (f"vllm serve exited with code {self._proc.returncode}" if exited
                   else f"{self._failed_in_row} requests in a row failed")
            self._down = f"{why}, the last with {err!r}; see {self.log_path}"
        print(f"  vllm serve stopped answering: {self._down}. No more requests go to it.", flush=True)

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
                 logprobs: bool = False, model: Optional[str] = None) -> Completion:
        """One chat completion, with its token counts and finish reason. model: the served
        name to send it to (route() checks it); None sends it to the default one."""
        payload = {"model": model or self._served_name(), "messages": messages,
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


def check_route(base: str, lora_modules: dict, name: Optional[str]) -> str:
    """The served name of a route: the base for None, else a module the server loaded.
    vLLM answers 404 for a model name it does not serve (OpenAIServing._check_model in
    vllm/entrypoints/openai/serving_engine.py at tag v0.10.2); this fails before any
    request is sent."""
    if name is None:
        return base
    if name not in lora_modules:
        raise ValueError(f"no adapter {name!r} on this server; it serves {base} and {sorted(lora_modules)}")
    return name


class Route:
    """The client of one model a running server serves, by its served name.

    It has what the protocols (CallLog) and the rubric grader call: complete() and
    chat(). Each request carries the route's name in its "model" field.
    """

    def __init__(self, engine, name: str):
        self.engine = engine
        self.name = name

    def complete(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None, logprobs: bool = False) -> Completion:
        return self.engine.complete(messages, max_new_tokens, temperature, top_p, seed, logprobs,
                                    model=self.name)

    def chat(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0) -> str:
        return self.complete(messages, max_new_tokens, temperature, top_p).text
