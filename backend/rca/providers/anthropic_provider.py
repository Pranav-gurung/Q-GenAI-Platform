"""Anthropic Claude provider (LLM_PROVIDER=anthropic), official `anthropic` SDK.

Request: Messages API, system prompt + one user message, structured output via `output_config.format`
(JSON schema), `output_config.effort` (LLM_EFFORT, default low for latency), bounded `max_tokens` and timeout,
one SDK retry. No temperature: the API rejects sampling parameters on current models. Optional server-side refusal
fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`) re-runs a request a safety classifier
declines on Anthropic's recommended fallback model.
"""
from __future__ import annotations

import logging
import os

import anthropic

from backend.core.logging import QUIET_LOGGERS

for _name in QUIET_LOGGERS:  # the SDK logs full prompts at DEBUG; keep that out of logs even if the app sets DEBUG
    logging.getLogger(_name).setLevel(logging.WARNING)

from backend.core.config import Settings
from backend.rca.prompt import RCAPrompt
from backend.rca.providers.base import LLMCompletion, LLMError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, http_client=None, max_retries: int = 1) -> None:
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.effort = settings.llm_effort
        self.refusal_fallback = settings.llm_refusal_fallback
        secret = settings.llm_api_key
        key = (secret.get_secret_value() if hasattr(secret, "get_secret_value") else secret) or os.environ.get(
            "ANTHROPIC_API_KEY")
        self._client = (anthropic.Anthropic(api_key=key, timeout=settings.llm_timeout_seconds, max_retries=max_retries,
                                            **({"http_client": http_client} if http_client is not None else {}))
                        if key else None)

    def status(self) -> str:
        return "configured" if self._client else "not_configured"

    def generate_rca(self, prompt: RCAPrompt) -> LLMCompletion:
        if self._client is None:
            raise LLMError("LLM_NOT_CONFIGURED", "No API key configured (LLM_API_KEY or ANTHROPIC_API_KEY)")
        request = dict(
            model=self.model, max_tokens=self.max_tokens, system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": prompt.output_schema}})
        try:
            if self.refusal_fallback:
                response = self._client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                response = self._client.messages.create(**request)
        except anthropic.APITimeoutError as exc:  # subclass of APIConnectionError: must come first
            raise LLMError("LLM_TIMEOUT", "LLM request timed out") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("LLM_RATE_LIMITED", "LLM provider rate limit reached") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMError("LLM_AUTH_FAILED", f"LLM provider rejected the credentials (HTTP {exc.status_code})") from exc
        except anthropic.APIStatusError as exc:
            code = "LLM_UNAVAILABLE" if exc.status_code >= 500 else "LLM_API_ERROR"
            raise LLMError(code, f"LLM provider returned HTTP {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("LLM_UNAVAILABLE", "Could not reach the LLM provider") from exc

        if response.stop_reason == "refusal":
            raise LLMError("LLM_REFUSED", "The model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("LLM_OUTPUT_TRUNCATED", "Model output hit LLM_MAX_TOKENS before completing")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            raise LLMError("LLM_EMPTY_OUTPUT", "The model returned no text")
        usage = response.usage
        return LLMCompletion(text=text, model=response.model, stop_reason=response.stop_reason,
                             input_tokens=getattr(usage, "input_tokens", None),
                             output_tokens=getattr(usage, "output_tokens", None))
