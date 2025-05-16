from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional, Sequence, Union, Type

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_bridge.llm import BaseAsyncLLM
from llm_bridge.tool_types import ToolCallResult

from .responses import LLMResponseWrapper
from .chat_types import ChatMessage, ChatParams


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
            if ("content" not in openai_msg and 
                not openai_msg.get("tool_calls") and 
                not openai_msg.get("function_call")):
                openai_msg["content"] = ""
                
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    def build_params(self, params: ChatParams) -> dict[str, Any]:
        """Convert ChatParams to OpenAI API parameters."""
        # Use enhanced params if available, otherwise create minimal dict
        if hasattr(params, 'as_dict'):
            base_params = params.as_dict(exclude_none=True)
        else:
            # Fallback for basic ChatParams
            base_params = {}
            for attr in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
                        'presence_penalty', 'response_format', 'seed', 'stop', 
                        'tools', 'tool_choice', 'user', 'parallel_tool_calls']:
                value = getattr(params, attr, None)
                if value is not None:
                    base_params[attr] = value
        
        # Remove stream from params dict as it's handled separately
        base_params.pop('stream', None)
        
        # Add any extra_params
        if hasattr(params, 'extra_params') and params.extra_params:
            base_params.update(params.extra_params)
            
        return base_params
    
    def build_tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """
        Return a ChatMessage that OpenAI expects for a tool result.
        """
        return {
            "role": "tool",
            "tool_call_id": result.id,
            "content": str(result.content) if not isinstance(result.content, str) else result.content,
        }


class OpenAILLM(BaseAsyncLLM):
    """
    OpenAI LLM implementation (async-only).
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
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncOpenAI(api_key=self.api_key, timeout=timeout, max_retries=max_retries, base_url=base_url)
        self._adapter = OpenAIRequestAdapter()

    @property
    def wrapper_class(self) -> Type[LLMResponseWrapper]:
        """Response wrapper class for OpenAI provider."""
        from .provider_wrappers import OpenAIWrapper
        return OpenAIWrapper

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
            **self._adapter.build_params(params)
        }
        
        self._log(f"Sending request to OpenAI model {self.model} (Stream: {params.stream})")

        if params.stream:
            # Use the generic streaming implementation
            return self._generic_stream(
                lambda: self._client.chat.completions.create(**args)
            )
        else:
            # Non-streaming: direct API call
            response: ChatCompletion = await self._client.chat.completions.create(**args)
            return response


def create_openai_message_dict(wrapper: LLMResponseWrapper) -> Optional[dict[str, Any]]:
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
                    "arguments": tc.function.arguments
                }
            }
            for tc in message.tool_calls
        ]
        # OpenAI spec: content is null if tool_calls is present
        message_dict["content"] = None
    
    # Handle function call (legacy format)
    if message.function_call:
        message_dict["function_call"] = {
            "name": message.function_call.name,
            "arguments": message.function_call.arguments
        }
    
    # Ensure content is set appropriately
    if ("content" not in message_dict and 
        not message_dict.get("tool_calls") and 
        not message_dict.get("function_call")):
        message_dict["content"] = ""
        
    return message_dict