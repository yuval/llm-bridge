from abc import ABC, abstractmethod
from typing import Optional, Any

__all__ = ["LLMResponseWrapper"]


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

    @property
    @abstractmethod
    def raw_response(self):
        """Access to the underlying provider response object."""
        ...

    @abstractmethod
    def get_response_content(self) -> str:
        ...

    @abstractmethod
    def raise_for_error(self) -> None:
        ...

    @abstractmethod  
    def get_tool_calls(self) -> list['ToolCallRequest'] | None:
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