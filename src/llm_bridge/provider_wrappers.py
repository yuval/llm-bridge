from __future__ import annotations

from typing import Union, Optional, Any
import json

from .errors import UnsupportedResponseTypeError
from .responses import LLMResponseWrapper
from .tool_types import ToolCallRequest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from anthropic.types import Message


class OpenAIWrapper(LLMResponseWrapper):
    """OpenAI-specific response wrapper with built-in content extraction."""
    
    def __init__(
        self,
        raw_response: ChatCompletion | ChatCompletionChunk | None = None,
        *,
        error_message: str | None = None,
    ) -> None:
        if (raw_response is None) == (error_message is None):
            raise UnsupportedResponseTypeError(
                "Provide exactly one of 'raw_response' or 'error_message'."
            )
        self._response = raw_response
        self._error_message = error_message

    @property
    def is_error(self) -> bool:
        return self._error_message is not None

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def raw_response(self) -> ChatCompletion | ChatCompletionChunk | None:
        return self._response

    def get_response_content(self) -> str:
        if self.is_error or self._response is None:
            return ""
        return self._extract_content(self._response)

    def get_tool_calls(self) -> list[ToolCallRequest] | None:
        """Extract tool calls from OpenAI response."""
        if self.is_error or not self._response:
            return None

        calls: list[ToolCallRequest] = []

        if isinstance(self._response, ChatCompletion) and self._response.choices:
            message = self._response.choices[0].message
            if message.tool_calls:
                for tc in message.tool_calls:
                    # Handle both dict and string arguments
                    if isinstance(tc.function.arguments, dict):
                        arguments = tc.function.arguments
                    else:
                        # Parse JSON string arguments
                        try:
                            arguments = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            # Fallback to eval for backward compatibility (less safe)
                            arguments = eval(tc.function.arguments)
                    
                    calls.append(
                        ToolCallRequest(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments
                        )
                    )
        return calls or None

    def _extract_content(self, response: ChatCompletion | ChatCompletionChunk) -> str:
        """Extract content from OpenAI response."""
        content: Optional[str] = None
        if isinstance(response, ChatCompletionChunk):
            # Streaming chunk
            if response.choices and response.choices[0].delta:
                content = response.choices[0].delta.content
        else:
            # Complete response
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
        return content if content is not None else ""

    def raise_for_error(self) -> None:
        if self.is_error:
            raise RuntimeError(self._error_message)


class AnthropicWrapper(LLMResponseWrapper):
    """Anthropic-specific response wrapper with built-in content extraction."""
    
    def __init__(
        self,
        raw_response: Union[Message, Any] | None = None,
        *,
        error_message: str | None = None,
    ) -> None:
        if (raw_response is None) == (error_message is None):
            raise UnsupportedResponseTypeError(
                "Provide exactly one of 'raw_response' or 'error_message'."
            )
        self._response = raw_response
        self._error_message = error_message

    @property
    def is_error(self) -> bool:
        return self._error_message is not None

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def raw_response(self) -> Union[Message, Any] | None:
        return self._response

    def get_response_content(self) -> str:
        if self.is_error or self._response is None:
            return ""
        return self._extract_content(self._response)

    def get_tool_calls(self) -> list[ToolCallRequest] | None:
        """Extract tool calls from Anthropic response."""
        if self.is_error or not self._response:
            return None

        calls: list[ToolCallRequest] = []

        if isinstance(self._response, Message) and self._response.content:
            for block in self._response.content:
                if block.type == "tool_use":
                    calls.append(
                        ToolCallRequest(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )
        return calls or None

    def _extract_content(self, response: Union[Message, Any]) -> str:
        """Extract content from Anthropic response."""
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

    def raise_for_error(self) -> None:
        if self.is_error:
            raise RuntimeError(self._error_message)


class GeminiWrapper(OpenAIWrapper):  # Gemini uses same response types as OpenAI
    """Gemini-specific response wrapper (reuses OpenAI extraction logic)."""
    pass