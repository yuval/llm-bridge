from __future__ import annotations

import logging
from typing import Optional, Self, Sequence, Type

from openai import AsyncOpenAI

from llm_bridge.responses import GeminiResponse
from llm_bridge.types.chat import BaseChatResponse, ChatMessage, ChatParams

from .base import BaseAsyncLLM, ChatResult
from .openai import OpenAIRequestAdapter


_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiLLM(BaseAsyncLLM):
    """
    Gemini LLM implementation via the OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        base_url: str = _DEFAULT_GEMINI_BASE_URL,
    ) -> None:
        super().__init__(model=model, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._adapter = OpenAIRequestAdapter()

    # Alternate constructor
    @classmethod
    def from_client(
        cls,
        model: str,
        client: AsyncOpenAI,
        *,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
    ) -> Self:
        """
        Wrap an existing ``AsyncOpenAI`` client already configured
        with Gemini's base URL.
        """
        if not isinstance(client, AsyncOpenAI):
            raise TypeError(
                f"GeminiLLM.from_client expects AsyncOpenAI; got {type(client).__name__}"
            )

        self = cls.__new__(cls)  # bypass __init__
        BaseAsyncLLM.__init__(self, model=model, logger=logger, name=name)
        self.api_key = client.api_key
        self._client = client
        self._adapter = OpenAIRequestAdapter()
        return self

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
            **self._adapter.build_params(params, self.model),
        }

        self._log(
            f"Sending request to Gemini model {self.model} (Stream: {params.stream})"
        )

        if params.stream:
            # Handle streaming
            return self._generic_stream(
                lambda: self._client.chat.completions.create(stream=True, **args)
            )
        else:
            # Non-streaming
            response = await self._client.chat.completions.create(**args)
            return response
