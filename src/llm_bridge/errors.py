from __future__ import annotations
import logging

try:
    from openai import APIError as OpenAIAPIError
    from anthropic import APIError as AnthropicAPIError
except ImportError:
    OpenAIAPIError = OpenAIConnectionError = Exception

__all__ = ["UnsupportedResponseTypeError", "NonToolCallError"]


class UnsupportedResponseTypeError(ValueError):
    """Raised when an unsupported response type or combination of parameters is provided."""
    pass


class NonToolCallError(Exception):
    """Raised when tool getters are called on a non-tool call."""
    pass


# Simple mapping of error patterns to error types
ERROR_TYPE_PATTERNS = {
    # Connection-related errors
    "ConnectionError": "Connection error",
    "APIConnectionError": "Connection error", 
    "RemoteProtocolError": "Connection error",
    "HTTPError": "HTTP error",
    # API-related errors
    "APIError": "API error",
    "APIStatusError": "API error",
    "RateLimitError": "Rate limit error",
}


def classify_error(exception: Exception, logger: logging.Logger) -> str:
    """
    Classifies an exception and returns an appropriate error message.
    
    Args:
        exception: The caught exception
        logger: Logger for recording the error
        
    Returns:
        Formatted error message string
    """
    error_type = type(exception).__name__
    error_message = str(exception)
    
    # Check if we have a specific handler for known provider exceptions
    if isinstance(exception, (OpenAIAPIError, AnthropicAPIError)):
        status_info = getattr(exception, 'status_code', 'unknown')
        msg = f"API error ({status_info}): {error_message}"
        logger.error(msg)
        return msg
    
    # Look for patterns in the error type name
    for pattern, prefix in ERROR_TYPE_PATTERNS.items():
        if pattern in error_type:
            msg = f"{prefix}: {error_message}"
            logger.error(msg)
            return msg
    
    # Fallback for everything else
    msg = f"{error_type}: {error_message}"
    logger.exception(msg)  # Use exception for stack trace on unknown errors
    return msg