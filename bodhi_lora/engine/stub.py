"""A CPU stand-in for VLLMEngine: the same surface, no server, no model.

Stub runs (python -m bodhi_lora run <cell> --stub-engine) and the tests use it to run
the prompt, protocol, grading and bookkeeping code of a stage on a CPU. `reply` is a
fixed string or a function of the messages, so a grader stub can return a parseable
grade. Every call is kept in `calls`.

Its tokenizer splits on spaces, one id per distinct word, so the probe's token
arithmetic runs without downloading a tokenizer. A raw continuation is `reply` called
with the prompt text as one user message.

It serves names as the server would: the base, "adapter" for one adapter (lora_path),
or each named LoRA module. route(name) checks the name, and `routes` keeps the served
name of each call, in the order of `calls`, so a test can see which weights a request
asked for.
"""

import threading
from typing import Callable, List, Optional, Tuple, Union

from bodhi_lora.engine.vllm import ADAPTER_NAME, Completion, Route, check_route

Reply = Union[str, Callable[[list], str]]


def _words(messages: list) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


class StubEngine:
    def __init__(self, model: str, reply: Reply = "STUB RESPONSE", served: Optional[dict] = None, **settings):
        """served: the settings a vLLM server would report (make_engine passes them), which
        settings() returns; without it, settings() returns the constructor's settings."""
        self.model = model
        self._reply = reply
        self._settings = {**(served or {"model": model, **settings}), "stub": True}
        lora_path = settings.get("lora_path")
        self.lora_modules = {ADAPTER_NAME: lora_path} if lora_path else dict(settings.get("lora_modules") or {})
        self._default = ADAPTER_NAME if lora_path else model
        self.calls: List[list] = []
        self.routes: List[str] = []
        # Protocols call from many threads; calls and routes stay in step.
        self._lock = threading.Lock()
        self._vocab: dict = {}
        self._words: List[str] = []

    def settings(self) -> dict:
        return dict(self._settings)

    def __enter__(self) -> "StubEngine":
        return self

    def __exit__(self, *_) -> None:
        pass

    def route(self, name: Optional[str]) -> Route:
        return Route(self, check_route(self.model, self.lora_modules, name))

    def _record(self, messages: list, model: Optional[str] = None) -> None:
        name = model or self._default
        if name != self.model and name not in self.lora_modules:
            raise ValueError(f"The model `{name}` does not exist.")
        with self._lock:
            self.calls.append(messages)
            self.routes.append(name)

    def complete(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 logprobs: bool = False, model: Optional[str] = None) -> Completion:
        self._record(messages, model)
        text = self._reply(messages) if callable(self._reply) else self._reply
        # Words stand in for tokens; the stub's counts only have to be plausible.
        n = len(text.split())
        return Completion(text=text, prompt_tokens=_words(messages), completion_tokens=n,
                          finish_reason="length" if n >= max_new_tokens else "stop",
                          token_logprobs=[])

    def chat(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0) -> str:
        return self.complete(messages, max_new_tokens, temperature, top_p).text

    def chat_with_logprobs(self, messages: list, max_new_tokens: int,
                           temperature: float = 0.0) -> Tuple[str, List[float]]:
        done = self.complete(messages, max_new_tokens, temperature, logprobs=True)
        return done.text, done.token_logprobs

    def tokenize(self, text: str) -> List[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._words)
                self._words.append(word)
            ids.append(self._vocab[word])
        return ids

    def detokenize(self, ids: List[int]) -> str:
        return " ".join(self._words[i] for i in ids)

    def continue_ids(self, prompt_ids: List[int], max_new_tokens: int,
                     temperature: float = 0.0) -> Completion:
        messages = [{"role": "user", "content": self.detokenize(prompt_ids)}]
        self._record(messages)
        text = self._reply(messages) if callable(self._reply) else self._reply
        n = len(text.split())
        return Completion(text=text, prompt_tokens=len(prompt_ids), completion_tokens=n,
                          finish_reason="length" if n >= max_new_tokens else "stop")
