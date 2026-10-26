"""LLM provider abstraction. Keys come from environment variables only and are never logged.

Providers use plain HTTPS via urllib (no vendor SDK dependency). Every failure mode maps to a typed
exception; nothing ever fabricates a successful response.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Protocol

from .config import Settings


class LLMError(Exception):
    code = "llm_error"
    http_status = 502


class LLMConfigurationError(LLMError):
    code, http_status = "llm_not_configured", 503


class LLMTimeoutError(LLMError):
    code, http_status = "llm_timeout", 504


class LLMRateLimitError(LLMError):
    code, http_status = "llm_rate_limited", 429

    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMUnavailableError(LLMError):
    code, http_status = "llm_unavailable", 503


class LLMOutputError(LLMError):
    code, http_status = "llm_output_invalid", 502


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> str: ...


class UnconfiguredProvider:
    name, model, configured = "none", "", False

    def generate(self, system: str, user: str) -> str:
        raise LLMConfigurationError("no LLM configured: set RCA_LLM_PROVIDER (anthropic|openai_compatible), "
                                    "RCA_LLM_MODEL and the provider API key environment variable")


def _post_json(url: str, headers: dict, body: dict, timeout: float, retries: int,
               opener: Callable = urllib.request.urlopen, sleep: Callable = time.sleep) -> dict:
    data = json.dumps(body).encode("utf-8")
    last: Optional[LLMError] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with opener(req, timeout=timeout) as resp:
                try:
                    return json.loads(resp.read().decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as e:
                    raise LLMOutputError("provider returned a non-JSON HTTP body") from e
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise LLMConfigurationError(f"provider rejected the credentials (HTTP {e.code})") from None
            if e.code == 429:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    ra_s = float(ra) if ra else None
                except ValueError:
                    ra_s = None
                last = LLMRateLimitError("provider rate limit reached (HTTP 429)", ra_s)
            elif e.code >= 500:
                last = LLMUnavailableError(f"provider error (HTTP {e.code})")
            else:
                raise LLMError(f"provider rejected the request (HTTP {e.code})") from None
        except (socket.timeout, TimeoutError):
            last = LLMTimeoutError(f"provider did not answer within {timeout:.0f}s")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                last = LLMTimeoutError(f"provider did not answer within {timeout:.0f}s")
            else:
                last = LLMUnavailableError(f"cannot reach provider: {type(e.reason).__name__}")
        if attempt < retries:
            delay = min(getattr(last, "retry_after", None) or (2 ** attempt), 10)
            sleep(delay)
    assert last is not None
    raise last


class AnthropicProvider:
    name, configured = "anthropic", True

    def __init__(self, s: Settings, opener=urllib.request.urlopen, sleep=time.sleep):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is not set")
        if not s.llm_model:
            raise LLMConfigurationError("RCA_LLM_MODEL is not set")
        self._key, self.model, self.s, self._opener, self._sleep = key, s.llm_model, s, opener, sleep
        self._url = (s.llm_base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"

    def generate(self, system: str, user: str) -> str:
        out = _post_json(self._url, {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
                         {"model": self.model, "max_tokens": self.s.llm_max_tokens, "temperature": 0, "system": system,
                          "messages": [{"role": "user", "content": user}]},
                         self.s.llm_timeout_s, self.s.llm_max_retries, self._opener, self._sleep)
        text = "".join(b.get("text", "") for b in out.get("content", []) if isinstance(b, dict) and b.get("type") == "text")
        if not text.strip():
            raise LLMOutputError("provider returned no text content")
        return text


class OpenAICompatibleProvider:
    name, configured = "openai_compatible", True

    def __init__(self, s: Settings, opener=urllib.request.urlopen, sleep=time.sleep):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMConfigurationError("OPENAI_API_KEY is not set")
        if not s.llm_model:
            raise LLMConfigurationError("RCA_LLM_MODEL is not set")
        self._key, self.model, self.s, self._opener, self._sleep = key, s.llm_model, s, opener, sleep
        self._url = (s.llm_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def generate(self, system: str, user: str) -> str:
        out = _post_json(self._url, {"Authorization": f"Bearer {self._key}"},
                         {"model": self.model, "max_tokens": self.s.llm_max_tokens, "temperature": 0,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                         self.s.llm_timeout_s, self.s.llm_max_retries, self._opener, self._sleep)
        try:
            text = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMOutputError("provider response has no choices[0].message.content") from e
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputError("provider returned empty content")
        return text


def get_provider(s: Settings) -> LLMProvider:
    """Never raises for 'none'; a configuration error surfaces only when generation is requested."""
    if s.llm_provider == "none":
        return UnconfiguredProvider()
    try:
        if s.llm_provider == "anthropic":
            return AnthropicProvider(s)
        if s.llm_provider == "openai_compatible":
            return OpenAICompatibleProvider(s)
    except LLMConfigurationError as e:
        failing = UnconfiguredProvider()
        failing.name, msg = s.llm_provider, str(e)

        def _raise(system, user, _m=msg):
            raise LLMConfigurationError(_m)
        failing.generate = _raise  # type: ignore[method-assign]
        return failing
    raise ValueError(f"unknown RCA_LLM_PROVIDER {s.llm_provider!r}")
