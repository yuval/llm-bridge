from __future__ import annotations

import logging
from typing import Type

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from llm_bridge.providers.anthropic import AnthropicLLM
from llm_bridge.providers.gemini import GeminiLLM
from llm_bridge.providers.base import BaseAsyncLLM
from llm_bridge.providers.openai import OpenAILLM

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
    api_key: str | None = None,
    client: AsyncOpenAI | AsyncAnthropic | None = None,
    logger: logging.Logger | None = None,
    **provider_kwargs: object,
) -> BaseAsyncLLM:
    """
    Factory for creating any supported LLM.

    Args:
        provider: Which provider to use (OPENAI, ANTHROPIC, GEMINI).
        model: Model identifier (e.g. "gemini-2.5-flash-preview-04-17").
        api_key: Overrides automatic lookup; if omitted, pulled from env.
        client: Optional pre-configured client instance to use.
            - For Provider.OPENAI: an AsyncOpenAI instance
            - For Provider.ANTHROPIC: an AsyncAnthropic instance
            - For Provider.GEMINI: currently only AsyncOpenAI instance is supported (OpenAI-compatible)
            If not provided, the relevant client with the default configuration will be used.
        logger: Optional custom logger.
        **provider_kwargs: Any extra args to pass through (timeout, max_retries).
    """
    try:
        llm_cls = _LLM_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc

    if client is not None:  # use caller‑supplied client verbatim
        return llm_cls.from_client(model, client, logger=logger, **provider_kwargs)

    key = api_key or get_api_key(provider)
    return llm_cls(model, api_key=key, logger=logger, **provider_kwargs)
