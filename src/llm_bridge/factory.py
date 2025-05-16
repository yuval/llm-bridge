from __future__ import annotations

import logging
from typing import Optional, Type

from .anthropic_provider import AnthropicLLM
from .gemini_provider import GeminiLLM
from .openai_provider import OpenAILLM
from .llm import BaseAsyncLLM
from .providers import Provider, get_api_key

# map Provider enum to its LLM implementation
_LLM_REGISTRY: dict[Provider, Type[BaseAsyncLLM]] = {
    Provider.OPENAI: OpenAILLM,
    Provider.ANTHROPIC: AnthropicLLM,
    Provider.GEMINI: GeminiLLM,
}


def create_llm(
    provider: Provider,
    model: str,
    *,
    api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **provider_kwargs: object,
) -> BaseAsyncLLM:
    """
    Factory for creating any supported LLM.

    Args:
        provider: Which provider to use (OPENAI, ANTHROPIC, GEMINI).
        model:     Model identifier (e.g. "gemini-2.5-flash-preview-04-17").
        api_key:   Overrides automatic lookup; if omitted, pulled from env.
        logger:    Optional custom logger.
        **provider_kwargs: Any extra args to pass through (timeout, max_retries).

    Returns:
        An instance of BaseAsyncLLM ready for chat().
    """
    llm_cls = _LLM_REGISTRY.get(provider)
    if llm_cls is None:
        raise ValueError(f"Unsupported provider: {provider}")

    # auto‑fetch key if the user didn’t give one
    key = api_key or get_api_key(provider)
    return llm_cls(model, api_key=key, logger=logger, **provider_kwargs)