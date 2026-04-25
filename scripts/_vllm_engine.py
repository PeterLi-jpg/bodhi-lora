"""vLLM inference engine backed by Docker image vllm/vllm-tpu:latest.

Usage (context manager — preferred):
    with VLLMEngine("google/medgemma-27b-text-it", tp_size=8) as engine:
        text = engine.chat(messages)
        text, logprobs = engine.chat_with_logprobs(messages)

    # With LoRA adapter for eval:
    with VLLMEngine("google/gemma-3-4b-it", tp_size=1,
                    lora_path="checkpoints/seed_42/best") as engine:
        text = engine.chat(messages)

The engine starts a detached docker container running `vllm serve`, waits
for the /health endpoint, then serves requests via the OpenAI-compatible
HTTP API on localhost:8000 (or the configured port).  __exit__ kills the
container so no cleanup is needed even on exception.

TP-size auto-selection: pass tp_size explicitly, or let _auto_tp() guess
from the model name (≤8B → 1 chip, everything else → 8 chips).
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple


DOCKER_IMAGE = "vllm/vllm-tpu:latest"
DEFAULT_PORT = 8000


def _auto_tp(model_name: str) -> int:
    """Heuristic: small models (≤8B) run fine on one chip; use all 8 for bigger ones."""
    name = model_name.lower()
    for tag in ("1b", "2b", "3b", "4b", "7b", "8b"):
        if tag in name:
            return 1
    return 8


class VLLMEngine:
    """Manages a `vllm serve` process running inside vllm/vllm-tpu:latest Docker.

    HF model cache ($HOME/.cache/huggingface) and $HOME are both mounted into
    the container so model weights downloaded on the host are reused and LoRA
    adapter paths under $HOME/... resolve correctly.
    """

    def __init__(
        self,
        model: str,
        tp_size: Optional[int] = None,
        max_model_len: int = 4096,
        lora_path: Optional[str] = None,
        port: int = DEFAULT_PORT,
        hf_token: Optional[str] = None,
    ):
        self.model = model
        self.tp_size = tp_size if tp_size is not None else _auto_tp(model)
        self.max_model_len = max_model_len
        self.port = port
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._hf_cache_host = os.path.expanduser("~/.cache/huggingface")
        self._home_host = os.path.expanduser("~")
        # Resolve lora_path to absolute so the container mount is correct.
        self._lora_path_host = os.path.realpath(lora_path) if lora_path else None
        self._lora_name = "adapter" if lora_path else None
        self._container_id: Optional[str] = None

    # ── path translation ────────────────────────────────────────────────────

    def _to_container(self, host_path: str) -> str:
        """Translate an absolute host path to its in-container counterpart.

        Mounts:
          $HOME/.cache/huggingface  →  /hf_cache
          $HOME                     →  /host_home
        """
        hf = self._hf_cache_host
        home = self._home_host
        if host_path.startswith(hf):
            return "/hf_cache" + host_path[len(hf):]
        if host_path.startswith(home):
            return "/host_home" + host_path[len(home):]
        return host_path

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _build_docker_cmd(self) -> List[str]:
        lora_args: List[str] = []
        if self._lora_path_host:
            lora_container = self._to_container(self._lora_path_host)
            lora_args = [
                "--enable-lora",
                "--lora-modules", f"{self._lora_name}={lora_container}",
            ]

        serve_cmd = [
            "vllm", "serve", self.model,
            "--tensor-parallel-size", str(self.tp_size),
            "--max-model-len", str(self.max_model_len),
            "--dtype", "bfloat16",
            "--port", str(self.port),
            "--disable-log-requests",
            *lora_args,
        ]

        return [
            "sudo", "docker", "run", "-d",
            "--privileged", "--net=host",
            "-v", "/dev/shm:/dev/shm", "--shm-size", "10gb",
            "-v", f"{self._hf_cache_host}:/hf_cache",
            "-v", f"{self._home_host}:/host_home",
            "-e", f"HF_HOME=/hf_cache",
            "-e", f"HF_TOKEN={self.hf_token}",
            "-e", "VLLM_LOGGING_LEVEL=WARNING",
            DOCKER_IMAGE,
            *serve_cmd,
        ]

    def start(self) -> "VLLMEngine":
        lora_tag = f", lora={self._lora_name}" if self._lora_name else ""
        print(
            f"Starting vllm serve: {self.model} "
            f"(TP={self.tp_size}, port={self.port}{lora_tag})",
            flush=True,
        )
        cmd = self._build_docker_cmd()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed:\n{result.stderr.strip()}"
            )
        self._container_id = result.stdout.strip()
        print(f"  container: {self._container_id[:12]}", flush=True)
        self._wait_ready()
        return self

    def stop(self) -> None:
        if self._container_id:
            subprocess.run(
                ["sudo", "docker", "rm", "-f", self._container_id],
                capture_output=True,
            )
            self._container_id = None
            print(f"  vllm serve stopped", flush=True)

    def _wait_ready(self, timeout_s: int = 900) -> None:
        health_url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout_s
        last_log = time.time()
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=5) as r:
                    if r.status == 200:
                        elapsed = int(time.time() - (deadline - timeout_s))
                        print(f"  vllm serve ready ({elapsed}s)", flush=True)
                        return
            except Exception:
                pass
            if time.time() - last_log > 30:
                elapsed = int(time.time() - (deadline - timeout_s))
                print(f"  waiting for vllm serve... ({elapsed}s)", flush=True)
                last_log = time.time()
            time.sleep(5)
        raise RuntimeError(
            f"vllm serve did not become healthy within {timeout_s}s"
        )

    def __enter__(self) -> "VLLMEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── HTTP helpers ────────────────────────────────────────────────────────

    def _post(self, payload: dict) -> dict:
        url = f"http://localhost:{self.port}/v1/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    # ── public API ──────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate a response for the given messages; return text only."""
        model_id = self._lora_name or self.model
        resp = self._post({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        })
        return resp["choices"][0]["message"]["content"]

    def chat_with_logprobs(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate a response and return (text, per_token_logprobs).

        Requests logprobs=True so vLLM returns the log-probability of each
        emitted token.  Used by score_response_confidence — avoids a separate
        forward pass (prompt_logprobs is untested on the TPU backend).
        """
        model_id = self._lora_name or self.model
        resp = self._post({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 1,
        })
        choice = resp["choices"][0]
        text = choice["message"]["content"]
        lp_content = (choice.get("logprobs") or {}).get("content") or []
        token_logprobs = [
            e["logprob"] for e in lp_content if e.get("logprob") is not None
        ]
        return text, token_logprobs
