"""
LLM Bridge - Unified interface for multiple LLM providers.
"""

from .providers.base import BaseAsyncLLM
from .providers.anthropic import AnthropicLLM, ephemeral
from .providers.openai import OpenAILLM
from .providers.gemini import GeminiLLM
from .responses import OpenAIResponse, AnthropicResponse, GeminiResponse
from .types.chat import ChatMessage, ChatParams, BaseChatResponse
from .types.tool import ToolCallRequest, ToolCallResult
from .factory import create_llm
from .providers import Provider, get_api_key

__version__ = "0.1.0"

__all__ = [
    "BaseAsyncLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "GeminiLLM",
    "ephemeral",
    "OpenAIResponse",
    "AnthropicResponse",
    "GeminiResponse",
    "ChatMessage",
    "ChatParams",
    "BaseChatResponse",
    "ToolCallRequest",
    "ToolCallResult",
    "create_llm",
    "Provider",
    "get_api_key",
]