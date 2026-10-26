"""The real AnthropicProvider, exercised through the official SDK against a simulated HTTP transport.
No network, no API key: the transport plays the Messages API (success, malformed output and every failure class)."""
import json
import logging

import httpx2
import pytest

pytest.importorskip("anthropic")
import anthropic  # noqa: E402

from backend.core.config import Settings  # noqa: E402
from backend.rca.prompt import build_prompt  # noqa: E402
from backend.rca.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider  # noqa: E402
from backend.rca.providers.base import LLMError  # noqa: E402
from backend.rca.providers.mock_provider import build_output  # noqa: E402
from backend.rca.service import RCAService  # noqa: E402
from backend.schemas.rca import RCAStatus  # noqa: E402
from tests.rca.helpers import CANONICAL, trace_of  # noqa: E402

KEY = "sk-ant-test-SECRET-123456"
TRACE = trace_of(CANONICAL)


def message(text, stop_reason="end_turn"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 300}}


def provider(handler, **settings):
    s = Settings(_env_file=None, llm_enabled=True, llm_provider="anthropic", llm_api_key=KEY, **settings)
    client = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicProvider(s, http_client=client, max_retries=0), s


def good_handler(seen):
    def handler(request):
        seen.append(request)
        prompt_ev = json.loads(request.content)["messages"][0]["content"]
        from backend.rca.prompt import evidence_from_user_message
        return httpx2.Response(200, json=message(json.dumps(build_output(evidence_from_user_message(prompt_ev)))))
    return handler


def test_success_and_request_shape(caplog):
    seen = []
    p, s = provider(good_handler(seen))
    caplog.set_level(logging.DEBUG)
    rca = RCAService(s, provider=p).analyze(TRACE)
    assert rca.status is RCAStatus.COMPLETED and rca.generation.provider == "anthropic"
    assert rca.generation.model == "claude-opus-5-5" and rca.generation.input_tokens == 1200
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 8000
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body and "top_p" not in body  # sampling params are rejected by the API
    assert body["fallbacks"] == "default" and FALLBACK_BETA in seen[0].headers["anthropic-beta"]
    assert body["system"].startswith("You write root-cause analyses") and "<evidence>" in body["messages"][0]["content"]
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"  # sent to the provider only
    assert KEY not in caplog.text and "<evidence>" not in caplog.text  # no key, no prompt in logs


def test_fallback_can_be_disabled():
    seen = []
    p, _ = provider(good_handler(seen), llm_refusal_fallback=False)
    p.generate_rca(_prompt())
    assert "fallbacks" not in json.loads(seen[0].content) and "anthropic-beta" not in seen[0].headers


def _prompt():
    from backend.rca.evidence import build_evidence
    return build_prompt(build_evidence(TRACE))


@pytest.mark.parametrize("status,code", [(429, "LLM_RATE_LIMITED"), (401, "LLM_AUTH_FAILED"), (403, "LLM_AUTH_FAILED"),
                                         (400, "LLM_API_ERROR"), (404, "LLM_API_ERROR"), (500, "LLM_UNAVAILABLE"),
                                         (529, "LLM_UNAVAILABLE")])
def test_api_errors(status, code):
    p, _ = provider(lambda r: httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "boom"}}))
    with pytest.raises(LLMError) as exc:
        p.generate_rca(_prompt())
    assert exc.value.code == code and KEY not in exc.value.message


@pytest.mark.parametrize("exc_cls,code", [(httpx2.ReadTimeout, "LLM_TIMEOUT"), (httpx2.ConnectError, "LLM_UNAVAILABLE")])
def test_timeout_and_network(exc_cls, code):
    def handler(request):
        raise exc_cls("simulated", request=request)
    p, _ = provider(handler)
    with pytest.raises(LLMError) as exc:
        p.generate_rca(_prompt())
    assert exc.value.code == code


@pytest.mark.parametrize("stop,code", [("refusal", "LLM_REFUSED"), ("max_tokens", "LLM_OUTPUT_TRUNCATED")])
def test_stop_reasons(stop, code):
    p, _ = provider(lambda r: httpx2.Response(200, json=message("{}", stop_reason=stop)))
    with pytest.raises(LLMError) as exc:
        p.generate_rca(_prompt())
    assert exc.value.code == code


def test_malformed_response_is_rejected_safely():
    p, s = provider(lambda r: httpx2.Response(200, json=message("Sure! Here is the RCA: {not json")))
    rca = RCAService(s, provider=p).analyze(TRACE)
    assert rca.status is RCAStatus.INVALID_OUTPUT and rca.requires_review and rca.summary is None
    assert rca.generation.error_code == "LLM_INVALID_OUTPUT" and rca.defects[0].likely_causes == []
    assert rca.defects[0].facts.first_observed_stage.value == "PACKING"  # deterministic facts survive


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(_env_file=None, llm_enabled=True, llm_provider="anthropic")
    p = AnthropicProvider(s)
    assert p.status() == "not_configured"
    rca = RCAService(s, provider=p).analyze(TRACE)
    assert rca.status is RCAStatus.UNAVAILABLE and rca.generation.error_code == "LLM_NOT_CONFIGURED"


def test_service_maps_provider_failure_to_unavailable():
    p, s = provider(lambda r: httpx2.Response(503, json={"type": "error", "error": {"type": "overloaded", "message": "x"}}))
    rca = RCAService(s, provider=p).analyze(TRACE)
    assert rca.status is RCAStatus.UNAVAILABLE and rca.generation.error_code == "LLM_UNAVAILABLE" and rca.requires_review
