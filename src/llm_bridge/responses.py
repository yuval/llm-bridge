
"""Defines the contract and base implementation for LLM response wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

from .errors import UnsupportedResponseTypeError

__all__ = ["LLMResponseWrapper", "ResponseWrapper"]

T_RawResponse = TypeVar("T_RawResponse")


class LLMResponseWrapper(ABC):
    """
    Abstract base class that all LLM response wrappers must implement.
    Defines a consistent interface for handling responses, including error cases.
    """

    @property
    @abstractmethod
    def is_error(self) -> bool:
        ...

    @property
    @abstractmethod
    def error_message(self) -> Optional[str]:
        ...

    @abstractmethod
    def get_response_content(self) -> str:
        ...

    @abstractmethod
    def raise_for_error(self) -> None:
        ...


class ResponseWrapper(Generic[T_RawResponse], LLMResponseWrapper):
    """
    Generic response wrapper using an injected parser to extract content.
    """

    def __init__(
        self,
        parse_content: Callable[[T_RawResponse], str],
        *,
        llm_response: Optional[T_RawResponse] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if (llm_response is None) == (error_message is None):
            raise UnsupportedResponseTypeError(
                "Provide exactly one of 'llm_response' or 'error_message'."
            )
        self._response = llm_response
        self._error_message = error_message
        self._parse_content = parse_content

    @property
    def is_error(self) -> bool:
        return self._error_message is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def raw_response(self) -> Optional[T_RawResponse]:
        return self._response

    def get_response_content(self) -> str:
        if self.is_error or self._response is None:
            return ""
        
        try:
            return self._parse_content(self._response)
        except Exception as e:
            # If parsing fails, return empty string and log the error
            # You might want to log this error or handle it differently
            return ""

    def raise_for_error(self) -> None:
        if self.is_error:
            raise RuntimeError(self._error_message or "Unknown LLM error")

    def __bool__(self) -> bool:
        return not self.is_error

    def __iter__(self):
        yield self.get_response_content()

    def __repr__(self) -> str:
        if self.is_error:
            return f"{self.__class__.__name__}(error_message={self._error_message!r})"
        try:
            content = self.get_response_content()
            preview = content[:75] + "..." if len(content) > 75 else content
        except Exception:
            preview = f"raw_response='{str(self._response)[:75]}...'"
        return f"{self.__class__.__name__}(response_preview={preview!r})"