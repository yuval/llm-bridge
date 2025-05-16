from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional, Sequence, Type

from anthropic import AsyncAnthropic
from anthropic.types import Message

from .llm import BaseAsyncLLM, ChatResult
from .responses import LLMResponseWrapper
from .tool_types import ToolCallResult
from .chat_types import ChatMessage, ChatParams


class AnthropicRequestAdapter:
    """Adapter for converting generic requests to Anthropic format."""
    
    def build_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Convert generic ChatMessage to Anthropic's expected format."""
        anthropic_messages: list[dict[str, Any]] = []
        
        for msg in messages:
            anthropic_msg: dict[str, Any] = {"role": msg["role"]}
            
            # Handle content
            if msg.get("content") is not None:
                content = msg["content"]
                # Anthropic supports both string and list formats for content
                if isinstance(content, str):
                    anthropic_msg["content"] = content
                elif isinstance(content, list):
                    # Handle complex content (e.g., with images, tool results)
                    anthropic_msg["content"] = content
                else:
                    anthropic_msg["content"] = str(content)
            
            # Handle tool use responses (tool_call_id indicates this is a tool result)
            if msg.get("tool_call_id"):
                # This is a tool result message
                anthropic_msg["content"] = [{
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg.get("content", "")
                }]
                anthropic_msg["role"] = "user"  # Tool results are always user messages in Anthropic
                
            anthropic_messages.append(anthropic_msg)
        
        return anthropic_messages
    
    def build_params(self, params: ChatParams) -> dict[str, Any]:
        """Convert ChatParams to Anthropic API parameters."""
        # Use enhanced params if available, otherwise create minimal dict
        if hasattr(params, 'as_dict'):
            base_params = params.as_dict(exclude_none=True)
        else:
            # Fallback for basic ChatParams
            base_params = {}
            for attr in ['max_tokens', 'temperature', 'top_p', 'stop']:
                value = getattr(params, attr, None)
                if value is not None:
                    if attr == 'stop':
                        # Anthropic uses stop_sequences instead of stop
                        base_params["stop_sequences"] = value if isinstance(value, list) else [value]
                    else:
                        base_params[attr] = value
        
        # Anthropic requires max_tokens - provide a reasonable default if not specified
        if "max_tokens" not in base_params:
            base_params["max_tokens"] = 4096
        
        # Remove stream from params dict as it's handled separately
        base_params.pop('stream', None)
        
        # Handle tools (convert OpenAI format to Anthropic format if needed)
        if hasattr(params, 'tools') and params.tools is not None:
            anthropic_tools = []
            for tool in params.tools:
                if tool.get("type") == "function":
                    # Convert OpenAI function tool to Anthropic format
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
                # No support for other tool types yet (e.g., server side tools such as web search)
                else:
                    # Assume it's already in Anthropic format
                    anthropic_tools.append(tool)
            base_params["tools"] = anthropic_tools
        
        # Handle thinking mode (Anthropic-specific)
        if hasattr(params, 'extra_params') and params.extra_params:
            if "thinking" in params.extra_params:
                base_params["thinking"] = params.extra_params["thinking"]
            
            # Add any other extra_params
            for key, value in params.extra_params.items():
                if key not in base_params:  # Don't override existing parameters
                    base_params[key] = value
                    
        return base_params
    
    def build_tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """
        Return a ChatMessage that Anthropic expects for a tool_result.
        """
        return {
            "role": "user", #Anthropic mandates 'user' here
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.content,
                }
            ],
        }


class AnthropicLLM(BaseAsyncLLM):
    """
    Anthropic LLM implementation (async-only).
    Use SyncLLM wrapper for synchronous access.
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
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout, max_retries=max_retries, base_url=base_url)
        self._adapter = AnthropicRequestAdapter()

    @property
    def wrapper_class(self) -> Type[LLMResponseWrapper]:
        """Response wrapper class for Anthropic provider."""
        from .provider_wrappers import AnthropicWrapper
        return AnthropicWrapper

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> ChatResult:
        """Core implementation for Anthropic chat requests."""
        # Build request arguments using the adapter
        # Don't include 'stream' in args - it's handled separately
        all_messages = self._adapter.build_messages(messages)
        
        system_prompt = ""
        # Anthropic expects the system prompt as a top-level parameter, 
        # not a message in the messages list
        if all_messages and all_messages[0].get("role") == "system":
            system_prompt = all_messages.pop(0).get("content", "")

        args = {
            "model": self.model,
            "system": system_prompt,
            "messages": all_messages,
            **self._adapter.build_params(params)
        }
        
        self._log(f"Sending request to Anthropic model {self.model} (Stream: {params.stream})")

        if params.stream:
            # Anthropic requires special handling for streaming
            return self._anthropic_stream(args)
        else:
            # Non-streaming: direct API call
            response: Message = await self._client.messages.create(**args)
            return response

    async def _anthropic_stream(self, args: dict[str, Any]) -> AsyncGenerator[Any, None]:
        """Handle Anthropic-specific streaming with context manager."""
        async with self._client.messages.stream(**args) as stream:
            async for event in stream:
                yield event


def create_anthropic_message_dict(wrapper: LLMResponseWrapper) -> Optional[dict[str, Any]]:
    """Create Anthropic message dictionary from response wrapper."""
    if wrapper.is_error or not wrapper.raw_response:
        return None
    
    response = wrapper.raw_response
    
    # Check if it's a complete Message (not a streaming event)
    if hasattr(response, 'role') and hasattr(response, 'content'):
        message_dict: dict[str, Any] = {"role": response.role}
        
        # Handle content - Anthropic uses content blocks
        if response.content:
            # For single text block, simplify to string
            if len(response.content) == 1 and getattr(response.content[0], 'type', None) == "text":
                message_dict["content"] = response.content[0].text
            else:
                # Multiple blocks or non-text blocks, keep as array
                content_blocks = []
                for block in response.content:
                    block_type = getattr(block, 'type', None)
                    if block_type == "text":
                        content_blocks.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block_type == "tool_use":
                        content_blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block_type == "thinking":
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": block.thinking
                        })
                message_dict["content"] = content_blocks
        else:
            message_dict["content"] = ""
            
        return message_dict
    