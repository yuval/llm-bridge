"""Enhanced chat types with utility methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Optional, Union

from llm_bridge.types.tool import ToolCallRequest


@dataclass
class ChatParams:
    """Parameters for chat completion requests with utility methods."""

    # Core parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False

    # Tool parameters
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None

    # Other parameters
    response_format: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    # Reasoning models (GPT-5, o-series)
    reasoning_effort: Optional[str] = None 
    verbosity: Optional[str] = None

    # Provider-specific parameters
    extra_params: Optional[dict[str, Any]] = None

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

    def copy(self, **kwargs) -> "ChatParams":
        """
        Create a copy of this ChatParams with optional overrides.

        Args:
            **kwargs: Field values to override

        Returns:
            New ChatParams instance with overrides applied
        """
        # Get current values as dict
        current = self.as_dict(exclude_none=False)
        # Apply overrides
        current.update(kwargs)
        # Create new instance
        return ChatParams(**current)


# Type alias for chat messages
ChatMessage = dict[str, Any]


class BaseChatResponse(ABC):
    """
    Abstract base class that all LLM response wrappers must implement.
    Defines a consistent interface for handling responses, including error cases.
    """

    @property
    @abstractmethod
    def is_error(self) -> bool: ...

    @property
    @abstractmethod
    def error_message(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def raw_response(self):
        """Access to the underlying provider response object."""
        ...

    @abstractmethod
    def get_response_content(self) -> str: ...

    # @abstractmethod
    # def get_usage(self) -> dict[str, Any] | None:
    #     """Return raw usage information from the provider response, or None if unavailable."""
    #     ...

    @abstractmethod
    def raise_for_error(self) -> None: ...

    @abstractmethod
    def get_tool_calls(self) -> list["ToolCallRequest"] | None:
        """
        Extract tool calls from the response.

        Returns:
            List of ToolCallRequest objects if tool calls are present, None otherwise.
        """
        ...

    def extract_by_path(self, path: str, default: Any = None) -> Any:
        """
        Extract a value from the response using a dot-notation path.

        Args:
            path: Dot-notation path like "choices.0.message.content"
            default: Default value if path doesn't exist

        Returns:
            The extracted value or default
        """
        if self.is_error or not self.raw_response:
            return default

        current = self.raw_response
        for part in path.split("."):
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

    def __bool__(self) -> bool:
        return not self.is_error

    def __iter__(self):
        yield self.get_response_content()

    def __repr__(self) -> str:
        if self.is_error:
            return f"{self.__class__.__name__}(error_message={self.error_message!r})"
        try:
            content = self.get_response_content()
            preview = content[:75] + "..." if len(content) > 75 else content
        except Exception:
            preview = f"raw_response='{str(self.raw_response)[:75]}...'"
        return f"{self.__class__.__name__}(response_preview={preview!r})"
