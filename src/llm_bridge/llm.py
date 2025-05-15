"""Base classes for LLM implementations and sync wrapper."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Protocol, Sequence, TypeVar, Union
from dataclasses import dataclass, asdict

from .chat_types import ChatMessage, ChatParams
from .responses import LLMResponseWrapper
from .errors import _classify_error

__all__ = ["BaseAsyncLLM", "llm_call", "RequestAdapter", "ResponseExtractor"]

T = TypeVar('T', bound=LLMResponseWrapper)
# Type for streaming responses
StreamResult = Union[LLMResponseWrapper, AsyncGenerator[LLMResponseWrapper, None]]


def llm_call(wrapper_factory: Callable[[], T]) -> Callable[[Callable], Callable[..., T]]:
    """
    Decorator that centralizes error handling for LLM provider calls.
    
    Args:
        wrapper_factory: Factory function that creates the appropriate response wrapper
                        (e.g., lambda: ResponseWrapper(parse_function)) for success cases.
                        For errors, the factory should create a wrapper with error_message.
        
    Returns:
        Decorated function that catches exceptions and returns appropriate error wrappers
    """
    def decorator(func: Callable) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(self, *args, **kwargs) -> T:
            try:
                # Execute the original function
                result = await func(self, *args, **kwargs)
                return result
            except Exception as e:
                # Determine error type and create appropriate message
                error_message = _classify_error(e, self.logger)
                
                # Create error wrapper using the same factory but with error_message
                # This assumes the wrapper factory can handle error_message kwarg
                return wrapper_factory(error_message=error_message)
        return wrapper
    return decorator


class RequestAdapter(Protocol):
    """Protocol for adapting generic chat messages and params to provider-specific format."""
    
    def build_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Convert generic ChatMessage list to provider-specific format."""
        ...
    
    def build_params(self, params: ChatParams) -> dict[str, Any]:
        """Convert generic ChatParams to provider-specific parameters."""
        ...


class ResponseExtractor:
    """Generic extractor for common response fields across different providers."""
    
    @staticmethod
    def extract_by_path(wrapper: LLMResponseWrapper, path: str, default: Any = None) -> Any:
        """
        Extract a value from the response using a dot-notation path.
        
        Args:
            wrapper: The response wrapper
            path: Dot-notation path like "choices.0.message.content"
            default: Default value if path doesn't exist
            
        Returns:
            The extracted value or default
        """
        if wrapper.is_error or not wrapper.raw_response:
            return default
            
        current = wrapper.raw_response
        for part in path.split('.'):
            if part.isdigit():
                # Handle array indexing
                try:
                    current = current[int(part)]
                except (IndexError, TypeError):
                    return default
            else:
                # Handle object attribute/key access
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
        return current
    
    @staticmethod
    def extract_usage(wrapper: LLMResponseWrapper, provider: str) -> dict[str, Any] | None:
        """Extract usage statistics in a unified format."""
        if wrapper.is_error or not wrapper.raw_response:
            return None
            
        if provider == "openai":
            usage = ResponseExtractor.extract_by_path(wrapper, "usage")
            if usage:
                return {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(usage, 'completion_tokens', None),
                    "total_tokens": getattr(usage, 'total_tokens', None)
                }
        elif provider == "anthropic":
            usage = ResponseExtractor.extract_by_path(wrapper, "usage")
            if usage:
                result = {
                    "input_tokens": getattr(usage, 'input_tokens', None),
                    "output_tokens": getattr(usage, 'output_tokens', None)
                }
                # Add server tool use if available
                if hasattr(usage, 'server_tool_use') and usage.server_tool_use:
                    result["server_tool_use"] = {
                        "web_search_requests": getattr(usage.server_tool_use, 'web_search_requests', None)
                    }
                return result
        return None


@dataclass
class EnhancedChatParams(ChatParams):
    """Enhanced ChatParams with utility methods."""
    
    def as_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary, optionally excluding None values.
        
        Args:
            exclude_none: If True, exclude fields with None values
            
        Returns:
            Dictionary representation of the params
        """
        result = asdict(self)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result


class BaseAsyncLLM(ABC):
    """
    Base class for all LLM implementations. All implementations are async-first.
    """

    def __init__(
        self,
        model: str,
        *,
        logger: logging.Logger | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initializes the base LLM client.

        Args:
            model: The identifier of the LLM model to be used.
            logger: Optional logger instance. If None, a logger named after
                    this module will be used.
            name: Optional name for this component, used in logging.
                  If None, defaults to the concrete class's name.
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.name = name if name is not None else self.__class__.__name__

    @abstractmethod
    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> StreamResult:
        """
        Core asynchronous implementation for sending chat messages to the LLM.
        This method must be implemented by subclasses.

        Args:
            messages: A sequence of chat messages forming the conversation history.
            params: Parameters for the chat completion request.

        Returns:
            An LLMResponseWrapper for non-streaming requests, or
            AsyncGenerator[LLMResponseWrapper, None] for streaming requests.
        """
        ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: ChatParams | None = None,
    ) -> StreamResult:
        """
        Sends a chat request to the LLM and returns the response asynchronously.

        Args:
            messages: A sequence of chat messages.
            params: Optional parameters for the chat completion. If None,
                    default ChatParams will be used.

        Returns:
            An LLMResponseWrapper for non-streaming requests, or
            AsyncGenerator[LLMResponseWrapper, None] for streaming requests.
        """
        final_params = params if params is not None else ChatParams()
        return await self._chat_impl(messages, final_params)

    async def _generic_stream(
        self,
        make_stream: Callable[..., Any],
        parse_func: Callable,
    ) -> AsyncGenerator[LLMResponseWrapper, None]:
        """
        Generic streaming implementation that can be used by all providers.
        
        Args:
            make_stream: Async callable that creates and returns the stream
            parse_func: Function to parse individual stream chunks
            
        Yields:
            ResponseWrapper instances for each chunk
        """
        try:
            stream = await make_stream()
            async for chunk in stream:
                # Skip empty chunks if needed (provider-specific logic can be handled in make_stream)
                yield LLMResponseWrapper(parse_func, llm_response=chunk)
        except Exception as e:
            error_message = _classify_error(e, self.logger)
            yield LLMResponseWrapper(parse_func, error_message=error_message)

    def _log(self, message: str) -> None:
        self.logger.info(f"[{self.name}] {message}")
