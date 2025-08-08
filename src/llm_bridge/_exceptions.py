"""
Translate noisy provider tracebacks into a unified `LLMBridgeError`, while
preserving the original exception for full tracebacks.
"""

from __future__ import annotations

import importlib
import logging
from typing import Final, Type, Optional

__all__: tuple[str, ...] = ("LLMBridgeError", "classify_error")


class LLMBridgeError(RuntimeError):
    """Public bridge‐level exception.

    Attributes:
        original_exc: The underlying provider exception.
    """

    original_exc: Exception

    def __init__(self, message: str, original_exc: Exception) -> None:
        super().__init__(message)
        self.original_exc = original_exc
        self.__cause__ = original_exc


def _import_exception(path: str) -> Type[Exception]:
    """Dynamically import an exception type, falling back to Exception."""
    module_name, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError):
        return Exception


OpenAI_APIError: Final = _import_exception("openai.APIError")
OpenAI_APIConnectionError: Final = _import_exception("openai.APIConnectionError")
OpenAI_RateLimitError: Final = _import_exception("openai.RateLimitError")

Anthropic_APIError: Final = _import_exception("anthropic.APIError")
Anthropic_APIConnectionError: Final = _import_exception("anthropic.APIConnectionError")
Anthropic_RateLimitError: Final = _import_exception("anthropic.RateLimitError")

API_ERRORS: Final[tuple[Type[Exception], ...]] = (
    OpenAI_APIError,
    Anthropic_APIError,
)

CONN_ERRORS: Final[tuple[Type[Exception], ...]] = (
    OpenAI_APIConnectionError,
    Anthropic_APIConnectionError,
    TimeoutError,
    ConnectionError,
)

RATE_LIMIT_ERRORS: Final[tuple[Type[Exception], ...]] = (
    OpenAI_RateLimitError,
    Anthropic_RateLimitError,
)


def classify_error(
    exc: Exception,
    logger: Optional[logging.Logger] = None,
) -> LLMBridgeError:
    """Wrap an SDK exception in LLMBridgeError with a friendly, concise message."""
    log = logger or logging.getLogger("llm_bridge.exceptions")

    if isinstance(exc, RATE_LIMIT_ERRORS):
        msg = "Rate‑limit exceeded – please retry later"
    elif isinstance(exc, CONN_ERRORS):
        msg = "Connection problem – unable to reach the LLM provider"
    elif isinstance(exc, API_ERRORS):
        msg = "Provider reported an internal error"
    else:
        msg = exc.__class__.__name__

    log.warning("Wrapping provider exception", extra={"exc": exc})
    return LLMBridgeError(f"{msg}: {exc}", exc)
