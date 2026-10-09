"""The protocol interface: how a reply is made from a conversation at inference.

A protocol turns the chat turns of one prompt into the reply the user sees, with one or
more calls to an engine. Every protocol returns an Output that records what the reply
cost (calls, tokens) and whether any call hit its token limit, so the eval can report
length and truncation for every condition the same way.

The identity protocol (no protocol: the conditions "base" and "lora") makes one call
with the conversation as given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Decoding:
    """Sampling settings of every call a protocol makes for one reply."""

    # Token limit of each call.
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None


@dataclass(frozen=True)
class Output:
    response: str                   # the reply the user sees; the training target of a trace
    analysis: Optional[str]         # a wrapper's internal text, kept for audit
    calls: int
    prompt_tokens: int              # summed over calls
    completion_tokens: int          # summed over calls
    finish_length: bool             # some call stopped at its token limit
    metadata: dict = field(default_factory=dict)
    # Log-probabilities of the reply's tokens, when the reply is one call's output.
    token_logprobs: Optional[list] = None


class CallLog:
    """Sends a protocol's calls to the engine and adds up what they cost."""

    def __init__(self, engine, decoding: Decoding, logprobs: bool = False):
        self.engine = engine
        self.decoding = decoding
        self.logprobs = logprobs
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_length = False
        self.last = None

    def chat(self, messages: list) -> str:
        d = self.decoding
        done = self.engine.complete(messages, d.max_new_tokens, temperature=d.temperature,
                                    top_p=d.top_p, seed=d.seed, logprobs=self.logprobs)
        self.calls += 1
        self.prompt_tokens += done.prompt_tokens
        self.completion_tokens += done.completion_tokens
        self.finish_length = self.finish_length or done.finish_length
        self.last = done
        return done.text

    def output(self, response: str, analysis: Optional[str] = None, metadata: Optional[dict] = None,
               token_logprobs: Optional[list] = None) -> Output:
        return Output(response=response, analysis=analysis, calls=self.calls,
                      prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                      finish_length=self.finish_length, metadata=metadata or {},
                      token_logprobs=token_logprobs)


class Protocol:
    """A way to make a reply. Subclasses implement respond()."""

    id: Optional[str] = None
    # wrapper: its calls get the per-pass token limit; prompt and identity: the reply limit.
    kind: str = "identity"

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        raise NotImplementedError


class Identity(Protocol):
    """No protocol: one call with the conversation as given."""

    def respond(self, engine, messages: list, decoding: Decoding) -> Output:
        log = CallLog(engine, decoding, logprobs=True)
        text = log.chat(messages)
        return log.output(text, token_logprobs=log.last.token_logprobs)
