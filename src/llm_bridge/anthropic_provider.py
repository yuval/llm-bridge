from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional, Sequence, Union

from anthropic import AsyncAnthropic
from anthropic.types import Message

from .llm import BaseAsyncLLM, llm_call, StreamResult, ResponseExtractor
from .responses import ResponseWrapper
from .chat_types import ChatMessage, ChatParams


#TODO: Fix streaming - first token is repeated for some reason
def _parse_anthropic_content(response: Union[Message, Any]) -> str:
    """Parse content from Anthropic response."""
    # First check if it's a complete Message with content blocks
    if hasattr(response, 'content') and response.content:
        # Complete message - extract text from content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, 'type') and block.type == "text" and hasattr(block, 'text'):
                text_parts.append(block.text)
        return "".join(text_parts)
    
    # Then check if it's a streaming event by looking for type attribute
    elif hasattr(response, 'type'):
        # Handle streaming events - only process text deltas
        if response.type == "content_block_delta":
            # Check if it's a text delta
            if hasattr(response, 'delta') and hasattr(response.delta, 'type') and response.delta.type == "text_delta":
                return response.delta.text
        # Handle direct text events (from older SDK versions)
        elif response.type == "text" and hasattr(response, 'text'):
            return response.text
        # For all other streaming events, return empty
        return ""
    
    return ""


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
                elif tool.get("type") == "web_search_20250305":
                    # Web search tool
                    anthropic_tools.append({
                        "name": "web_search",
                        "type": "web_search_20250305"
                    })
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


# Simplified type alias using the generic ResponseWrapper
AnthropicChatResponseWrapper = ResponseWrapper[Union[Message, Any]]


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
    ) -> None:
        super().__init__(model=model, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout, max_retries=max_retries)
        self._adapter = AnthropicRequestAdapter()

    @llm_call(lambda **kwargs: ResponseWrapper(_parse_anthropic_content, **kwargs))
    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> StreamResult:
        """Core implementation for Anthropic chat requests."""
        # Build request arguments using the adapter
        # Don't include 'stream' in args - it's handled separately
        args = {
            "model": self.model,
            "messages": self._adapter.build_messages(messages),
            **self._adapter.build_params(params)
        }
        
        self._log(f"Sending request to Anthropic model {self.model} (Stream: {params.stream})")

        if params.stream:
            # Anthropic requires special handling for streaming
            return self._anthropic_stream(args)
        else:
            # Non-streaming: direct API call
            response: Message = await self._client.messages.create(**args)
            return ResponseWrapper(_parse_anthropic_content, llm_response=response)

    async def _anthropic_stream(self, args: dict[str, Any]) -> AsyncGenerator[ResponseWrapper[Any], None]:
        """Handle Anthropic-specific streaming with context manager."""
        try:
            async with self._client.messages.stream(**args) as stream:
                async for event in stream:
                    yield ResponseWrapper(_parse_anthropic_content, llm_response=event)
        except Exception as e:
            self._log(f"Streaming error: {e}", level=logging.ERROR)
            yield ResponseWrapper(_parse_anthropic_content, error_message=str(e))


# Unified helper functions using ResponseExtractor

def get_anthropic_tool_uses(wrapper: ResponseWrapper) -> Optional[list]:
    """Extract tool uses from Anthropic response wrapper."""
    if wrapper.is_error or not wrapper.raw_response:
        return None
    
    response = wrapper.raw_response
    if isinstance(response, Message) and response.content:
        tool_uses = []
        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        return tool_uses if tool_uses else None
    return None


def get_anthropic_thinking(wrapper: ResponseWrapper) -> Optional[str]:
    """Extract thinking content from Anthropic response wrapper."""
    # Check for thinking in content blocks
    content = ResponseExtractor.extract_by_path(wrapper, "content")
    if content:
        for block in content:
            if getattr(block, 'type', None) == "thinking":
                return getattr(block, 'thinking', None)
    
    # Check for streaming thinking events
    response = wrapper.raw_response
    if hasattr(response, 'type') and response.type == "thinking":
        return getattr(response, 'thinking', None)
    
    return None


def get_anthropic_stop_reason(wrapper: ResponseWrapper) -> Optional[str]:
    """Extract stop reason from Anthropic response wrapper."""
    return ResponseExtractor.extract_by_path(wrapper, "stop_reason")


def get_anthropic_usage(wrapper: ResponseWrapper) -> Optional[dict]:
    """Extract usage statistics from Anthropic response wrapper."""
    return ResponseExtractor.extract_usage(wrapper, "anthropic")


def get_anthropic_web_search_results(wrapper: ResponseWrapper) -> Optional[list[dict]]:
    """Extract web search results from Anthropic response wrapper."""
    content = ResponseExtractor.extract_by_path(wrapper, "content")
    if content:
        for block in content:
            if (getattr(block, 'type', None) == "web_search_tool_result" and 
                hasattr(block, 'content')):
                return block.content
    return None


def create_anthropic_message_dict(wrapper: ResponseWrapper) -> Optional[dict[str, Any]]:
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
    
    return None