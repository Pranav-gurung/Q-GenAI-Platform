"""LLM providers for RCA. Only the factory imports a vendor SDK, and only when that provider is selected."""
from __future__ import annotations

from backend.core.config import Settings
from backend.rca.providers.base import LLMCompletion, LLMError, LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        from backend.rca.providers.mock_provider import MockProvider
        return MockProvider(model="mock-rca-1")
    try:
        from backend.rca.providers.anthropic_provider import AnthropicProvider
    except ImportError:  # optional extra not installed: the app still starts, RCA reports UNAVAILABLE
        return _MissingDependencyProvider(settings.llm_model)
    return AnthropicProvider(settings)


class _MissingDependencyProvider:
    name = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model

    def status(self) -> str:
        return "dependencies_missing"

    def generate_rca(self, prompt):
        raise LLMError("LLM_DEPENDENCIES_MISSING", "The anthropic SDK is not installed (pip install -e '.[llm]')")


__all__ = ["LLMCompletion", "LLMError", "LLMProvider", "build_provider"]
