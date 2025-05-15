"""Enhanced chat types with utility methods."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union


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