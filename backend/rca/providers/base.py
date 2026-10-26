"""LLM provider contract. Providers only turn a prompt into raw text; parsing and validation happen in RCAService,
so every provider is held to exactly the same schema and grounding checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.rca.prompt import RCAPrompt


class LLMError(Exception):
    """A provider failure with a stable code. Messages never contain credentials or prompt content.

    Codes: LLM_DISABLED, LLM_NOT_CONFIGURED, LLM_TIMEOUT, LLM_RATE_LIMITED, LLM_UNAVAILABLE, LLM_AUTH_FAILED,
           LLM_API_ERROR, LLM_REFUSED, LLM_OUTPUT_TRUNCATED, LLM_EMPTY_OUTPUT
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def generate_rca(self, prompt: RCAPrompt) -> LLMCompletion:
        """Return the model's raw JSON text, or raise LLMError."""
        ...

    def status(self) -> str:
        """'configured' or 'not_configured' - static check only, never calls the network."""
        ...
