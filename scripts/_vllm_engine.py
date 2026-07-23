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
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple


DOCKER_IMAGE_TPU = "vllm/vllm-tpu:latest"
DOCKER_IMAGE_GPU = "vllm/vllm-openai:latest"
DEFAULT_PORT = 8000


def _detect_accelerator() -> str:
    """Return 'tpu' or 'gpu' based on what the host is.

    Checks (in order):
      1. ``BODHI_VLLM_ACCEL`` env var  ('tpu' / 'gpu')  — explicit override.
      2. ``PJRT_DEVICE`` env var  ('TPU')               — set on Cloud TPU VMs.
      3. ``/dev/vfio`` directory exists                 — TPU char devices live here.
      4. ``nvidia-smi`` exits 0                          — Nvidia GPU host.

    Default fallback is 'gpu' so a stock CUDA box "just works" without
    setting any env vars.  An explicit override is the only way to force
    TPU mode if both detections somehow fire.
    """
    explicit = os.environ.get("BODHI_VLLM_ACCEL", "").lower()
    if explicit in ("tpu", "gpu"):
        return explicit
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return "tpu"
    if os.path.exists("/dev/vfio") and any(
        d.isdigit() for d in os.listdir("/dev/vfio")
    ):
        return "tpu"
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True, capture_output=True, timeout=5,
        )
        return "gpu"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "gpu"  # default — let docker fail loudly if neither is present


def _detect_run_mode() -> str:
    """Return 'docker' or 'subprocess'.

    On TPU and bare-metal GPU hosts we run vLLM inside a Docker container
    (TPU image bundles libtpu, GPU image bundles CUDA + nccl deps without
    polluting the host's pip env).  Inside cloud GPU pods (RunPod, Lambda
    Labs container instances, k8s sidecars), the pod IS already a container
    and there is no Docker daemon to nest into — we have to run vLLM as a
    plain Python subprocess on the host.

    Detection order:
      1. ``BODHI_VLLM_MODE`` env var  ('docker' / 'subprocess')   — override.
      2. ``/.dockerenv`` exists                                    — we are
         already inside a container; default to subprocess.
      3. ``sudo docker info`` succeeds                             — daemon
         reachable via sudo (matches the launch path which uses
         ``sudo docker run``); use docker.
      4. Otherwise                                                 — subprocess.

    The probe uses ``sudo`` deliberately: the actual container start in
    ``_build_docker_cmd`` uses ``sudo docker run``, so a host where the
    user is not in the ``docker`` group but does have NOPASSWD sudo for
    docker (the typical Cloud TPU VM setup) reaches docker mode here too.
    """
    explicit = os.environ.get("BODHI_VLLM_MODE", "").lower()
    if explicit in ("docker", "subprocess"):
        return explicit
    if os.path.exists("/.dockerenv"):
        return "subprocess"
    try:
        subprocess.run(
            ["sudo", "-n", "docker", "info"],
            check=True, capture_output=True, timeout=5,
        )
        return "docker"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "subprocess"


def _auto_tp(model_name: str) -> int:
    """Heuristic: small models (≤8B) run fine on one chip; use all 8 for bigger ones.

    Override with BODHI_VLLM_TP (e.g. =1 when pinning a single GPU via
    CUDA_VISIBLE_DEVICES; =N to tensor-parallel across N idle GPUs). Needed on
    shared boxes: nvidia-smi -L lists ALL GPUs regardless of CUDA_VISIBLE_DEVICES,
    so a 24B model would otherwise pick TP=8 and fail against a 1-GPU pin.

    Uses a negative lookbehind so that multi-digit sizes like "14b" or "70b"
    don't accidentally match single-digit tags ("4b" inside "14b" would give
    a wrong TP=1 for a 14B model).
    """
    override = os.environ.get("BODHI_VLLM_TP")
    if override:
        return max(1, int(override))
    name = model_name.lower()
    # Match a standalone single-digit size tag, e.g. "4b" in "gemma-3-4b-it"
    # but NOT "4b" embedded in "14b" or "24b".
    if re.search(r"(?<!\d)[1-8]b(?!\w)", name):
        return 1
    # Cap by actual visible GPU count so single-GPU pods (RunPod, Lambda, etc.)
    # don't try TP=8 against a 1-GPU placement group and hang.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, timeout=5,
        )
        n = max(1, sum(1 for l in out.splitlines() if l.strip().startswith("GPU ")))
        return min(8, n)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
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
        max_model_len: int = 8192,
        lora_path: Optional[str] = None,
        port: int = DEFAULT_PORT,
        hf_token: Optional[str] = None,
        enforce_eager: bool = True,
    ):
        # vllm/vllm-tpu does NOT implement add_lora as of writing — the engine
        # boots, then crashes inside init_static_loras with NotImplementedError
        # ~30s into startup. Without this guard the caller's _wait_ready()
        # polls the dead container for the full 45-minute timeout. Fail fast
        # with a clear pointer to the GPU pod path the production pipeline
        # already uses for Stage 4 eval (see scripts/launch_gpu.sh + the
        # subprocess-mode addition in _detect_run_mode below).
        if lora_path and _detect_accelerator() == "tpu":
            raise NotImplementedError(
                "vllm/vllm-tpu does not support LoRA serving (NotImplementedError "
                "in add_lora). Run Stage 4 eval / latency benchmark on a GPU pod "
                "instead — see scripts/launch_gpu.sh. Train on TPU, eval on GPU "
                "is the supported topology."
            )

        self.model = model
        self.tp_size = tp_size if tp_size is not None else _auto_tp(model)
        self.max_model_len = max_model_len
        self.port = port
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._enforce_eager = enforce_eager
        # Honor the host's HF_HOME redirect (set by tpu/setup_tpu.sh to
        # /dev/shm/hf when no /mnt/cache disk is attached). Without this, the
        # vLLM Docker container's HF cache always bind-mounted to the boot
        # disk's ~/.cache/huggingface, even when the launcher's daemon had
        # HF_HOME pointing at tmpfs — Stage 1 medgemma (54G) + Stage 2
        # llama-grader (16G) + Stage 4 merged checkpoint (54G) saturated the
        # 100G boot disk every time. Bind-mounting the actual HF_HOME path
        # routes vLLM's downloads to wherever HF_HOME points (typically the
        # 700G /dev/shm tmpfs).
        self._hf_cache_host = os.environ.get("HF_HOME") or os.path.expanduser(
            "~/.cache/huggingface"
        )
        self._home_host = os.path.expanduser("~")
        # Resolve lora_path to absolute so the container mount is correct.
        self._lora_path_host = os.path.realpath(lora_path) if lora_path else None
        self._lora_name = "adapter" if lora_path else None
        self._container_id: Optional[str] = None
        # Subprocess mode lifecycle handle (only set when run_mode='subprocess')
        self._proc: Optional[subprocess.Popen] = None
        self._run_mode = _detect_run_mode()

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
            # max-num-seqs caps how many sequences the scheduler runs in
            # parallel. KV-cache footprint is roughly
            # max_num_seqs * max_model_len * 360 KB/token for medgemma-27b
            # (28 layers x 8 GQA KV heads x 256 dim x 2 bytes bf16). On a
            # v6e-8 (256 GB HBM, ~200 GB free after weights):
            #   max_model_len=4096, max_num_seqs=128 -> 188 GB  (was prior config)
            #   max_model_len=8192, max_num_seqs=64  -> 188 GB  (current; same fit)
            # We picked the second option after BODHI's two-pass started
            # producing prompts >2048 tokens (the analysis from pass 1
            # is fed back as input to pass 2), which would clip at
            # max_model_len=4096. 64 server slots is still well above
            # the 32-thread client cap so no client-side queuing.
            "--max-num-seqs", "64",
            # --enforce-eager: skip CUDA-graph pre-capture.  vllm 0.9 with
            # --enable-lora captures ~67 graph shapes and each takes ~2 min
            # on A100, totalling 130+ min before the first prompt is served.
            # Eager mode is slightly slower per-token at runtime but pays
            # off massively for short eval runs (200 prompts).  Default
            # ON; opt out with enforce_eager=False (e.g. latency benchmark
            # where graph-mode throughput matters and the long capture
            # cost is acceptable).
            *(["--enforce-eager"] if self._enforce_eager else []),
            *lora_args,
        ]

        # On TPU we mount /dev with --privileged so the container can talk to
        # libtpu via the chip char-devices.  On GPU we instead pass --gpus=all
        # which Docker translates to the right nvidia runtime args.  The same
        # vllm serve cmd works on either image.
        accel = _detect_accelerator()
        if accel == "tpu":
            device_args = ["--privileged"]
            image = DOCKER_IMAGE_TPU
        else:
            device_args = ["--gpus", "all"]
            image = DOCKER_IMAGE_GPU

        return [
            "sudo", "docker", "run", "-d",
            *device_args, "--net=host",
            "-v", "/dev/shm:/dev/shm", "--shm-size", "10gb",
            "-v", f"{self._hf_cache_host}:/hf_cache",
            "-v", f"{self._home_host}:/host_home",
            "-e", f"HF_HOME=/hf_cache",
            "-e", f"HF_TOKEN={self.hf_token}",
            "-e", "VLLM_LOGGING_LEVEL=WARNING",
            image,
            *serve_cmd,
        ]

    def _build_subprocess_cmd(self) -> List[str]:
        """Build a plain `vllm serve` cmd that runs on the host directly.

        Used inside cloud GPU pods (RunPod, Lambda Labs container instances,
        etc.) where the pod IS already a container and there is no Docker
        daemon to nest into.  vLLM must be pip-installed in the pod's
        Python env (launch_gpu.sh does this).  Paths are NOT translated —
        the lora adapter dir is read directly from disk.
        """
        lora_args: List[str] = []
        if self._lora_path_host:
            lora_args = [
                "--enable-lora",
                "--lora-modules", f"{self._lora_name}={self._lora_path_host}",
            ]
        return [
            "vllm", "serve", self.model,
            "--tensor-parallel-size", str(self.tp_size),
            "--max-model-len", str(self.max_model_len),
            "--dtype", "bfloat16",
            "--port", str(self.port),
            # Match docker-mode batching capacity; see comment in
            # _build_docker_cmd.
            "--max-num-seqs", "64",
            # Match the docker-mode default so the two run-modes behave
            # consistently: --enforce-eager is on by default to skip the
            # ~130-min CUDA-graph capture hang on LoRA, opt out for
            # latency benchmarking.
            *(["--enforce-eager"] if self._enforce_eager else []),
            *lora_args,
        ]

    def start(self) -> "VLLMEngine":
        lora_tag = f", lora={self._lora_name}" if self._lora_name else ""
        print(
            f"Starting vllm serve: {self.model} "
            f"(TP={self.tp_size}, port={self.port}{lora_tag}, "
            f"mode={self._run_mode})",
            flush=True,
        )
        if self._run_mode == "docker":
            cmd = self._build_docker_cmd()
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"docker run failed:\n{result.stderr.strip()}")
            self._container_id = result.stdout.strip()
            print(f"  container: {self._container_id[:12]}", flush=True)
        else:
            # subprocess mode — run vllm directly on the host, redirect
            # stdout / stderr to a log file so it doesn't pollute our terminal
            # (vllm is chatty and the chat-probe in _wait_ready already tells
            # us when it's ready).
            cmd = self._build_subprocess_cmd()
            log_path = os.path.expanduser(f"~/vllm_serve_{self.port}.log")
            log_fh = open(log_path, "w")
            env = {
                **os.environ,
                "HF_TOKEN": self.hf_token,
                "VLLM_LOGGING_LEVEL": "WARNING",
            }
            self._proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
                preexec_fn=os.setsid,  # new pgid so we can kill the whole tree
            )
            print(f"  subprocess pid={self._proc.pid}, log={log_path}", flush=True)
        self._wait_ready()
        return self

    def stop(self) -> None:
        if self._container_id:
            subprocess.run(
                ["sudo", "docker", "rm", "-f", self._container_id],
                capture_output=True,
            )
            self._container_id = None
            print(f"  vllm serve stopped (docker)", flush=True)
        elif self._proc is not None:
            # Kill the whole process group so any child workers spawned by
            # vllm (engine workers, log streams, etc.) go down cleanly.
            try:
                os.killpg(os.getpgid(self._proc.pid), 15)  # SIGTERM
                self._proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self._proc.pid), 9)  # SIGKILL
                except ProcessLookupError:
                    pass
            self._proc = None
            print(f"  vllm serve stopped (subprocess)", flush=True)

    def _wait_ready(self, timeout_s: int = 2700) -> None:
        # /health goes 200 the moment the FastAPI process is up, BEFORE the
        # chat-completions route is registered.  We learned the hard way that
        # firing 16 concurrent requests at a "healthy" server can get 800 fast
        # 404s in a row.  So: poll /health first, then send a one-token test
        # chat to confirm /v1/chat/completions is actually serving.
        health_url = f"http://localhost:{self.port}/health"
        chat_url   = f"http://localhost:{self.port}/v1/chat/completions"
        deadline = time.time() + timeout_s
        last_log = time.time()
        health_ok = False
        while time.time() < deadline:
            try:
                if not health_ok:
                    with urllib.request.urlopen(health_url, timeout=5) as r:
                        health_ok = r.status == 200
                if health_ok:
                    # Send a tiny probe to /v1/chat/completions to confirm
                    # the actual inference route is wired up.
                    probe = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    }
                    req = urllib.request.Request(
                        chat_url,
                        data=json.dumps(probe).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as r:
                        if r.status == 200:
                            elapsed = int(time.time() - (deadline - timeout_s))
                            print(f"  vllm serve ready ({elapsed}s)", flush=True)
                            return
            except Exception:
                pass
            if time.time() - last_log > 30:
                elapsed = int(time.time() - (deadline - timeout_s))
                state = "chat-routing" if health_ok else "starting"
                print(f"  waiting for vllm serve [{state}]... ({elapsed}s)", flush=True)
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
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate a response for the given messages; return text only.

        Default max_new_tokens=2048: HealthBench responses (especially
        the BODHI two-pass analysis) routinely run 4000+ chars. The
        previous default of 1024 capped about 22% of the responses
        mid-sentence (audit on the 4799-row Stage 1 output, 2026-04-30):
        responses ending mid-letter at the 4000-4900 char window were
        clearly truncated. Doubling the cap to 2048 gives ~8000 chars
        of headroom while still leaving room in max_model_len=8192
        for typical 1500-2000 token healthbench prompts plus BODHI's
        pass-1 analysis (which becomes pass-2 input).
        """
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
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate a response and return (text, per_token_logprobs).

        Requests logprobs=True so vLLM returns the log-probability of each
        emitted token.  Used by score_response_confidence — avoids a separate
        forward pass (prompt_logprobs is untested on the TPU backend).

        Default max_new_tokens matches chat() for consistency: 2048 to avoid
        mid-sentence truncation on long medical responses.
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
