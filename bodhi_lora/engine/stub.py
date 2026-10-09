"""A CPU stand-in for VLLMEngine: the same surface, no server, no model.

Stub runs (python -m bodhi_lora run <cell> --stub-engine) and the tests use it to run
the prompt, protocol, grading and bookkeeping code of a stage on a CPU. `reply` is a
fixed string or a function of the messages, so a grader stub can return a parseable
grade. Every call is kept in `calls`.

Its tokenizer splits on spaces, one id per distinct word, so the probe's token
arithmetic runs without downloading a tokenizer. A raw continuation is `reply` called
with the prompt text as one user message.
"""

from typing import Callable, List, Optional, Tuple, Union

from bodhi_lora.engine.vllm import Completion

Reply = Union[str, Callable[[list], str]]


def _words(messages: list) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


class StubEngine:
    def __init__(self, model: str, reply: Reply = "STUB RESPONSE", **settings):
        self.model = model
        self._reply = reply
        self._settings = {"model": model, "stub": True, **settings}
        self.calls: List[list] = []
        self._vocab: dict = {}
        self._words: List[str] = []

    def settings(self) -> dict:
        return dict(self._settings)

    def __enter__(self) -> "StubEngine":
        return self

    def __exit__(self, *_) -> None:
        pass

    def complete(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 logprobs: bool = False) -> Completion:
        self.calls.append(messages)
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
        self.calls.append(messages)
        text = self._reply(messages) if callable(self._reply) else self._reply
        n = len(text.split())
        return Completion(text=text, prompt_tokens=len(prompt_ids), completion_tokens=n,
                          finish_reason="length" if n >= max_new_tokens else "stop")
