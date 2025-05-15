from __future__ import annotations

import logging
from typing import Optional, Sequence

from openai import AsyncOpenAI

from .llm import BaseAsyncLLM, llm_call, StreamResult
from .responses import ResponseWrapper
from .chat_types import ChatMessage, ChatParams
from .openai_provider import OpenAIRequestAdapter, _parse_openai_content


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

    @llm_call(lambda **kwargs: ResponseWrapper(_parse_openai_content, **kwargs))
    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> StreamResult:
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
            stream = await self._client.chat.completions.create(stream=True, **args)
            
            async def generate_responses():
                async for chunk in stream:
                    yield ResponseWrapper(_parse_openai_content, llm_response=chunk)
            
            return generate_responses()
        else:
            # Non-streaming
            try:
                response = await self._client.chat.completions.create(**args)
                return ResponseWrapper(_parse_openai_content, llm_response=response)
            except Exception as e:
                self._log(f"Error in Gemini request: {e}", level=logging.ERROR)
                return ResponseWrapper(_parse_openai_content, error_message=str(e))