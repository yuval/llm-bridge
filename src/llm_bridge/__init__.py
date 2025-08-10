"""
LLM Bridge - Unified interface for multiple LLM providers.
"""

from .client import (
    BaseAsyncLLM,
    OpenAILLM,
    AnthropicLLM,
    GeminiLLM,
    create_llm,
)
from .response import ChatResponse
from .params import ChatMessage
from .types import ToolCallRequest, ToolCallResult
from .provider import Provider, get_api_key
from .adapters.anthropic import ephemeral

__version__ = "0.1.0"

__all__ = [
    "BaseAsyncLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "GeminiLLM",
    "create_llm",
    "ChatResponse",
    "ChatMessage",
    "ToolCallRequest",
    "ToolCallResult",
    "Provider",
    "get_api_key",
    "ephemeral",
]
