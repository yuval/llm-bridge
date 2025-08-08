from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional, Self, Sequence, Union, Type

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_bridge.providers.base import BaseAsyncLLM
from llm_bridge.responses import OpenAIResponse
from llm_bridge.types.tool import ToolCallResult
from llm_bridge.types.chat import BaseChatResponse, ChatMessage, ChatParams


class OpenAIRequestAdapter:
    """Adapter for converting generic requests to OpenAI format."""

    def build_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Convert generic ChatMessage to OpenAI's expected format."""
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg["role"]}

            # Handle content (required for most message types)
            if msg.get("content") is not None:
                openai_msg["content"] = msg["content"]

            # Handle tool calls (for assistant messages with function calls)
            if msg.get("tool_calls"):
                openai_msg["tool_calls"] = msg["tool_calls"]
                # OpenAI spec: content should be null when tool_calls is present
                if "content" not in openai_msg:
                    openai_msg["content"] = None

            # Handle function call (legacy format, still supported)
            if msg.get("function_call"):
                openai_msg["function_call"] = msg["function_call"]

            # Handle tool call ID (for tool response messages)
            if msg.get("tool_call_id"):
                openai_msg["tool_call_id"] = msg["tool_call_id"]

            # Handle name (for function/tool responses or to identify speakers)
            if msg.get("name"):
                openai_msg["name"] = msg["name"]

            # Ensure content is set for messages that require it
            if (
                "content" not in openai_msg
                and not openai_msg.get("tool_calls")
                and not openai_msg.get("function_call")
            ):
                openai_msg["content"] = ""

            openai_messages.append(openai_msg)

        return openai_messages

    def build_params(self, params: ChatParams, model: str) -> dict[str, Any]:
        """Convert ChatParams to OpenAI API parameters."""
        # Use enhanced params if available, otherwise create minimal dict
        if hasattr(params, "as_dict"):
            base_params = params.as_dict(exclude_none=True)
        else:
            # Fallback for basic ChatParams
            base_params = {}
            for attr in [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "response_format",
                "seed",
                "stop",
                "tools",
                "tool_choice",
                "user",
                "parallel_tool_calls",
            ]:
                value = getattr(params, attr, None)
                if value is not None:
                    base_params[attr] = value

        # Remove fields not accepted by API
        base_params.pop("stream", None)
        base_params.pop("extra_params", None)

        # Fold Responses-style extras into Chat Completions for convenience
        extras = getattr(params, "extra_params", None) or {}
        reasoning = extras.get("reasoning")
        if isinstance(reasoning, dict) and "effort" in reasoning and "reasoning_effort" not in base_params:
            base_params["reasoning_effort"] = reasoning["effort"]
        text_cfg = extras.get("text")
        if isinstance(text_cfg, dict) and "verbosity" in text_cfg and "verbosity" not in base_params:
            base_params["verbosity"] = text_cfg["verbosity"]

        # Handle max_tokens vs max_completion_tokens based on model
        if "max_tokens" in base_params and base_params["max_tokens"] is not None:
            # Models that require max_completion_tokens instead of max_tokens
            if self._requires_max_completion_tokens(model):
                base_params["max_completion_tokens"] = base_params.pop("max_tokens")

        # Add remaining extra_params (after we consumed reasoning/text)
        if extras:
            for k, v in extras.items():
                if k not in ("reasoning", "text"):  # already mapped
                    base_params.setdefault(k, v)

        return base_params

    def _requires_max_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens."""
        # GPT-5 and other newer models require max_completion_tokens
        newer_models = {
            "gpt-5",  # GPT-5 series
            "o1",  # O1 series models
            "o3",  # O3 series models
        }

        # Check if model starts with any of the newer model prefixes
        return any(model.startswith(prefix) for prefix in newer_models)

    def build_tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """
        Return a ChatMessage that OpenAI expects for a tool result.
        """
        return {
            "role": "tool",
            "tool_call_id": result.id,
            "content": str(result.content)
            if not isinstance(result.content, str)
            else result.content,
        }


class OpenAILLM(BaseAsyncLLM):
    """
    OpenAI LLM implementation (async‑only).

    Use ``OpenAILLM.from_client`` when you already have an ``AsyncOpenAI`` instance.
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
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
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
        Build an ``OpenAILLM`` around an already‑configured ``AsyncOpenAI`` client.
        """
        if not isinstance(client, AsyncOpenAI):
            raise TypeError(
                f"OpenAILLM.from_client expects AsyncOpenAI; got {type(client).__name__}"
            )

        self = cls.__new__(cls)  # bypass __init__
        BaseAsyncLLM.__init__(self, model=model, logger=logger, name=name)
        self.api_key = client.api_key
        self._client = client
        self._adapter = OpenAIRequestAdapter()
        return self

    @property
    def wrapper_class(self) -> Type[BaseChatResponse]:
        """Response wrapper class for OpenAI provider."""
        return OpenAIResponse

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Core implementation for OpenAI chat requests."""
        # Build request arguments using the adapter
        args = {
            "model": self.model,
            "messages": self._adapter.build_messages(messages),
            "stream": params.stream,
            **self._adapter.build_params(params, self.model),
        }

        self._log(
            f"Sending request to OpenAI model {self.model} (Stream: {params.stream})"
        )

        if params.stream:
            # Use the generic streaming implementation
            return self._generic_stream(
                lambda: self._client.chat.completions.create(**args)
            )
        else:
            # Non-streaming: direct API call
            response: ChatCompletion = await self._client.chat.completions.create(
                **args
            )
            return response


def create_openai_message_dict(wrapper: BaseChatResponse) -> Optional[dict[str, Any]]:
    """Create OpenAI message dictionary from response wrapper."""
    if wrapper.is_error or not wrapper.raw_response:
        return None

    message = wrapper.extract_by_path("choices.0.message")
    if not message:
        return None

    message_dict: dict[str, Any] = {"role": message.role}

    # Handle content
    if message.content:
        message_dict["content"] = message.content

    # Handle tool calls
    if message.tool_calls:
        message_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
        # OpenAI spec: content is null if tool_calls is present
        message_dict["content"] = None

    # Handle function call (legacy format)
    if message.function_call:
        message_dict["function_call"] = {
            "name": message.function_call.name,
            "arguments": message.function_call.arguments,
        }

    # Ensure content is set appropriately
    if (
        "content" not in message_dict
        and not message_dict.get("tool_calls")
        and not message_dict.get("function_call")
    ):
        message_dict["content"] = ""

    return message_dict
