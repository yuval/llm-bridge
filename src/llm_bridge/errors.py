
from __future__ import annotations
import logging

try:
    from openai import APIError as OpenAIAPIError, APIConnectionError as OpenAIConnectionError
    from anthropic import APIError as AnthropicAPIError, APIConnectionError as AnthropicConnectionError
except ImportError:
    OpenAIAPIError = OpenAIConnectionError = Exception

__all__ = ["UnsupportedResponseTypeError", "NonToolCallError", "LLMBridgeError"]


class UnsupportedResponseTypeError(ValueError):
    """Raised when an unsupported response type or combination of parameters is provided."""
    pass


class NonToolCallError(Exception):
    """Raised when tool getters are called on a non-tool call."""
    pass


class LLMBridgeError(Exception):
    """Base exception for all LLM provider errors."""
    pass


def _classify_error(exception: Exception, logger: logging.Logger) -> str:
    """
    Classifies an exception and returns an appropriate error message.
    
    Args:
        exception: The caught exception
        logger: Logger for recording the error
        
    Returns:
        Formatted error message string
    """
    # Handle specific provider exceptions first
    if isinstance(exception, (OpenAIAPIError, AnthropicAPIError)):
        status_info = getattr(exception, 'status_code', 'unknown')
        msg = f"API error ({status_info}): {str(exception)}"
        logger.error(msg)
        return msg
    elif isinstance(exception, (OpenAIConnectionError, AnthropicConnectionError)):
        msg = f"Connection error: {str(exception)}"
        logger.error(msg)
        return msg
    
    # Fallback to string matching for broader compatibility
    error_type = type(exception).__name__
    error_message = str(exception)
    
    if "APIConnectionError" in error_type or "RemoteProtocolError" in error_type:
        msg = f"Connection error: {error_message}"
        logger.error(msg)
        return msg
    elif "RateLimitError" in error_type:
        msg = f"Rate limit error: {error_message}"
        logger.error(msg)
        return msg
    elif "APIStatusError" in error_type or "APIError" in error_type:
        status_info = getattr(exception, 'status_code', 'unknown')
        msg = f"API error ({status_info}): {error_message}"
        logger.error(msg)
        return msg
    elif "HTTPError" in error_type:
        msg = f"HTTP error: {error_message}"
        logger.error(msg)
        return msg
    else:
        # Generic error handling
        msg = f"Unhandled {error_type}: {error_message}"
        logger.exception(msg)  # Use exception for stack trace
        return msg