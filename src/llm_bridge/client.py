"""
LLM clients with unified chat() and stream() methods.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Optional,
    Protocol,
    Self,
    Sequence,
    Union,
)

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from anthropic.types import Message

from llm_bridge.errors import classify_error
from llm_bridge.response import ChatResponse
from llm_bridge.params import normalize_params, ChatMessage
from llm_bridge.types import ToolCallResult
from llm_bridge.adapters import (
    OpenAIRequestAdapter,
    AnthropicRequestAdapter,
    GeminiRequestAdapter,
)
from llm_bridge.provider import Provider, get_api_key


class RequestAdapter(Protocol):
    """Protocol for adapting between generic chat format and provider-specific format."""

    def to_provider(
        self, messages: Sequence[ChatMessage], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert generic messages and normalized params to provider-specific request format."""
        ...

    def from_provider(self, raw: Any) -> ChatResponse:
        """Convert provider response to unified ChatResponse."""
        ...

    def stream_text(self, raw_chunk: Any) -> ChatResponse:
        """Extract content from a streaming chunk and return as ChatResponse."""
        ...

    def assistant_message_from(self, raw: Any) -> ChatMessage:
        """Convert a provider response to a provider-specific assistant ChatMessage."""
        ...

    def tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """Convert a ToolCallResult to a provider-specific ChatMessage."""
        ...


class BaseAsyncLLM(ABC):
    """
    Abstract base class for async-first LLM wrappers.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: dict[str, Any],
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
    def adapter(self) -> RequestAdapter:
        """Request adapter for this provider."""
        ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Send chat request and return a single response.

        Returns a ChatResponse object.
        """
        normalized_params = normalize_params(params)
        normalized_params["stream"] = False  # Ensure non-streaming

        try:
            raw = await self._chat_impl(messages, normalized_params)
            return self.adapter.from_provider(raw)
        except Exception as exc:
            return self._wrap_error(exc)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Send chat request and return a stream of response chunks.

        Yields ChatResponse objects for each chunk.
        """
        normalized_params = normalize_params(params)
        normalized_params["stream"] = True  # Ensure streaming

        try:
            raw_result = await self._chat_impl(messages, normalized_params)
            if hasattr(raw_result, "__anext__"):
                # It's an async generator, iterate through it
                async for chunk in raw_result:
                    yield self.adapter.stream_text(chunk)
            else:
                # It's a single response, wrap and yield it
                response = self.adapter.from_provider(raw_result)
                yield response
        except Exception as exc:
            yield self._wrap_error(exc)

    def _wrap_error(self, exc: Exception) -> ChatResponse:
        """Wrap exception into an error response."""
        msg = classify_error(exc, self.logger)
        return ChatResponse(content="", error=str(msg))

    async def _generic_stream(
        self, make_stream: Callable[[], AsyncGenerator[Any, None]]
    ) -> AsyncGenerator[Any, None]:
        async for chunk in make_stream():
            yield chunk

    def _log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, f"[{self.name}] {message}")

    # --- lifecycle ---------------------------------------------------------
    async def aclose(self) -> None:
        """
        Close underlying async HTTP clients to avoid cleanup after the loop closes.
        Safe to call multiple times.
        """
        client = getattr(self, "_client", None)
        close = getattr(client, "aclose", None)
        if close:
            await close()

    async def __aenter__(self) -> "BaseAsyncLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


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
        super().__init__(model=model, api_key=api_key, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
        )
        self._adapter = OpenAIRequestAdapter()

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
        BaseAsyncLLM.__init__(
            self, model=model, api_key=client.api_key or "", logger=logger, name=name
        )
        self.api_key = client.api_key or ""
        self._client = client
        self._adapter = OpenAIRequestAdapter()
        return self

    @property
    def adapter(self) -> RequestAdapter:
        """Request adapter for OpenAI provider."""
        return self._adapter

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: dict[str, Any],
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Core implementation for OpenAI chat requests."""
        # Use adapter to build request
        request_data = self._adapter.to_provider(messages, params)

        args = {
            "model": self.model,
            "stream": params["stream"],
            **request_data,
        }

        # Handle special fields that need passthrough
        passthrough_keys = ("verbosity", "reasoning_effort")
        extra_body = {}
        for k in passthrough_keys:
            if k in args:
                extra_body[k] = args.pop(k)
        if extra_body:
            args["extra_body"] = {**args.get("extra_body", {}), **extra_body}

        self._log(
            f"Sending request to OpenAI model {self.model} (Stream: {params['stream']})"
        )

        if params["stream"]:
            # For streaming, we need to return an AsyncGenerator directly
            return await self._client.chat.completions.create(**args)
        else:
            response: ChatCompletion = await self._client.chat.completions.create(
                **args
            )
            return response


class AnthropicLLM(BaseAsyncLLM):
    """
    Anthropic LLM implementation (async‑only).

    Use ``AnthropicLLM.from_client`` when you already have an ``AsyncAnthropic`` instance.
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
        super().__init__(model=model, api_key=api_key, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
        )
        self._adapter = AnthropicRequestAdapter()

    @classmethod
    def from_client(
        cls,
        model: str,
        client: AsyncAnthropic,
        *,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
    ) -> Self:
        """
        Wrap an existing ``AsyncAnthropic`` client.
        """
        if not isinstance(client, AsyncAnthropic):
            raise TypeError(
                f"AnthropicLLM.from_client expects AsyncAnthropic; got {type(client).__name__}"
            )

        self = cls.__new__(cls)  # bypass __init__
        BaseAsyncLLM.__init__(
            self, model=model, api_key=client.api_key or "", logger=logger, name=name
        )
        self.api_key = client.api_key or ""
        self._client = client
        self._adapter = AnthropicRequestAdapter()
        return self

    @property
    def adapter(self) -> RequestAdapter:
        """Request adapter for Anthropic provider."""
        return self._adapter

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: dict[str, Any],
    ) -> Union[Message, AsyncGenerator[Any, None]]:
        """Core implementation for Anthropic chat requests."""
        # Use adapter to build request
        request_data = self._adapter.to_provider(messages, params)

        args = {
            "model": self.model,
            **request_data,
        }

        self._log(
            f"Sending request to Anthropic model {self.model} (Stream: {params['stream']})"
        )

        if params["stream"]:
            return self._anthropic_stream(args)
        else:
            response: Message = await self._client.messages.create(**args)
            return response

    async def _anthropic_stream(
        self, args: dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """Handle Anthropic-specific streaming with context manager."""
        async with self._client.messages.stream(**args) as stream:
            async for event in stream:
                yield event


_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiLLM(BaseAsyncLLM):
    """
    Gemini LLM implementation via the OpenAI-compatible endpoint.
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
        base_url: str = _DEFAULT_GEMINI_BASE_URL,
    ) -> None:
        super().__init__(model=model, api_key=api_key, logger=logger, name=name)
        self.api_key = api_key
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._adapter = GeminiRequestAdapter()

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
        Wrap an existing ``AsyncOpenAI`` client already configured
        with Gemini's base URL.
        """
        if not isinstance(client, AsyncOpenAI):
            raise TypeError(
                f"GeminiLLM.from_client expects AsyncOpenAI; got {type(client).__name__}"
            )

        self = cls.__new__(cls)  # bypass __init__
        BaseAsyncLLM.__init__(
            self, model=model, api_key=client.api_key or "", logger=logger, name=name
        )
        self.api_key = client.api_key or ""
        self._client = client
        self._adapter = GeminiRequestAdapter()
        return self

    @property
    def adapter(self) -> RequestAdapter:
        """Request adapter for Gemini provider."""
        return self._adapter

    async def _chat_impl(
        self,
        messages: Sequence[ChatMessage],
        params: dict[str, Any],
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Core implementation for Gemini chat requests via OpenAI-compatible API."""
        # Use OpenAI adapter since Gemini uses OpenAI-compatible API
        request_data = self._adapter.to_provider(messages, params)

        args = {
            "model": self.model,
            "stream": params["stream"],
            **request_data,
        }

        # Handle special fields that need passthrough
        passthrough_keys = ("verbosity", "reasoning_effort")
        extra_body = {}
        for k in passthrough_keys:
            if k in args:
                extra_body[k] = args.pop(k)
        if extra_body:
            args["extra_body"] = {**args.get("extra_body", {}), **extra_body}

        self._log(
            f"Sending request to Gemini model {self.model} (Stream: {params['stream']})"
        )

        if params["stream"]:
            # For streaming, we need to return an AsyncGenerator directly
            return await self._client.chat.completions.create(**args)
        else:
            response: ChatCompletion = await self._client.chat.completions.create(
                **args
            )
            return response


# Factory for creating LLM instances

_LLM_REGISTRY = {
    Provider.OPENAI: OpenAILLM,
    Provider.ANTHROPIC: AnthropicLLM,
    Provider.GEMINI: GeminiLLM,
}


def create_llm(
    provider: Provider,
    model: str,
    *,
    api_key: str | None = None,
    client: AsyncOpenAI | AsyncAnthropic | None = None,
    logger: logging.Logger | None = None,
    **provider_kwargs: Any,
) -> BaseAsyncLLM:
    """
    Factory for creating any supported LLM.

    Args:
        provider: Which provider to use (OPENAI, ANTHROPIC, GEMINI).
        model: Model identifier (e.g. "gemini-2.5-flash-preview-04-17").
        api_key: Overrides automatic lookup; if omitted, pulled from env.
        client: Optional pre-configured client instance to use.
            - For Provider.OPENAI: an AsyncOpenAI instance
            - For Provider.ANTHROPIC: an AsyncAnthropic instance
            - For Provider.GEMINI: currently only AsyncOpenAI instance is supported (OpenAI-compatible)
            If not provided, the relevant client with the default configuration will be used.
        logger: Optional custom logger.
        **provider_kwargs: Any extra args to pass through (timeout, max_retries).
    """
    try:
        llm_cls = _LLM_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc

    if client is not None:  # use caller‑supplied client verbatim
        if hasattr(llm_cls, "from_client"):
            return llm_cls.from_client(model, client, logger=logger, **provider_kwargs)
        else:
            raise ValueError(
                f"Provider {provider} does not support client-based initialization"
            )

    key = api_key or get_api_key(provider)
    return llm_cls(model=model, api_key=key, logger=logger, **provider_kwargs)
