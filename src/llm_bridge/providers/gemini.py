from __future__ import annotations

import logging
from typing import Optional, Sequence, Type

from openai import AsyncOpenAI

from llm_bridge.responses import GeminiResponse
from llm_bridge.types.chat import BaseChatResponse, ChatMessage, ChatParams

from .base import BaseAsyncLLM, ChatResult
from .openai import OpenAIRequestAdapter


class GeminiLLM(BaseAsyncLLM):
    """
    Gemini LLM implementation using OpenAI-compatible API.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, logger=logger, name=name)
        
        # Use OpenAI client with Gemini's base URL
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # Reuse OpenAI's request adapter since Gemini uses OpenAI-compatible API
        self._adapter = OpenAIRequestAdapter()

    @property
    def wrapper_class(self) -> Type[BaseChatResponse]:
        """Response wrapper class for Gemini provider."""
        return GeminiResponse

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> ChatResult:
        """Core implementation for Gemini chat requests via OpenAI-compatible API."""
        
        # Build request arguments using OpenAI adapter
        args = {
            "model": self.model,
            "messages": self._adapter.build_messages(messages),
            **self._adapter.build_params(params)
        }
        
        self._log(f"Sending request to Gemini model {self.model} (Stream: {params.stream})")

        if params.stream:
            # Handle streaming
            return self._generic_stream(
                lambda: self._client.chat.completions.create(stream=True, **args)
            )
        else:
            # Non-streaming
            response = await self._client.chat.completions.create(**args)
            return response