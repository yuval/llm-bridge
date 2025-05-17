"""Base classes for LLM implementations and sync wrapper."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Optional, Protocol, Sequence, Type, TypeVar, Union

from llm_bridge._exceptions import classify_error
from llm_bridge.types.chat import BaseChatResponse, ChatMessage, ChatParams


__all__ = ["BaseAsyncLLM", "RequestAdapter"]

T = TypeVar('T', bound=BaseChatResponse)
# Type for streaming responses
ChatResult = Union[BaseChatResponse, AsyncGenerator[BaseChatResponse, None]]


class RequestAdapter(Protocol):
    """Protocol for adapting generic chat messages and params to provider-specific format."""
    
    def build_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Convert generic ChatMessage list to provider-specific format."""
        ...
    
    def build_params(self, params: ChatParams) -> dict[str, Any]:
        """Convert generic ChatParams to provider-specific parameters."""
        ...  

class BaseAsyncLLM(ABC):
    """
    Base class for all LLM implementations. All implementations are async-first.
    """

    def __init__(
        self,
        model: str,
        *,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
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
        self.base_url = base_url

    @abstractmethod
    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: ChatParams,
    ) -> Union[Any, AsyncGenerator[Any, None]]:
        """
        Core asynchronous implementation for sending chat messages to the LLM.
        This method must be implemented by subclasses.

        Args:
            messages: A sequence of chat messages forming the conversation history.
            params: Parameters for the chat completion request.

        Returns:
            A raw provider response for non-streaming requests, or
            AsyncGenerator yielding raw provider responses for streaming requests.
        """
        ...

    @property
    @abstractmethod
    def wrapper_class(self) -> Type[BaseChatResponse]:
        """LLMResponseWrapper subclass for this provider."""
        ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: ChatParams | None = None,
    ) -> Union[BaseChatResponse, AsyncGenerator[BaseChatResponse, None]]:
        """
        Send chat, catching exceptions and wrapping responses uniformly.

        Returns a single wrapper or an async generator of wrappers.
        """
        final_params = params or ChatParams()
        try:
            raw = await self._chat_impl(messages, final_params)
        except Exception as exc:
            return self._wrap_error(exc)

        if isinstance(raw, AsyncGenerator):
            return self._wrap_stream(raw)
        return self._wrap_response(raw)

    def _wrap_error(self, exc: Exception) -> BaseChatResponse:
        """Wrap exception into an error response wrapper."""
        msg = classify_error(exc, self.logger)
        return self.wrapper_class(error_message=msg)

    def _wrap_response(self, raw: Any) -> BaseChatResponse:
        """Wrap raw provider response into a response wrapper."""
        return self.wrapper_class(raw_response=raw)

    def _wrap_stream(
        self, raw_stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[BaseChatResponse, None]:
        """Wrap an async stream of raw responses into wrapped responses."""
        async def gen() -> AsyncGenerator[BaseChatResponse, None]:
            try:
                async for item in raw_stream:
                    yield self.wrapper_class(raw_response=item)
            except Exception as exc:
                yield self._wrap_error(exc)

        return gen()
    
    async def _generic_stream(
        self, make_stream: Callable[[], AsyncGenerator[Any, None]]
    ) -> AsyncGenerator[Any, None]:
        async for chunk in await make_stream():
            yield chunk


    def _log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, f"[{self.name}] {message}")