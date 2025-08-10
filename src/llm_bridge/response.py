from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from llm_bridge.types import ToolCallRequest


@dataclass
class ChatResponse:
    """Unified response object for all LLM providers."""

    content: str
    tool_calls: list[ToolCallRequest] | None = None
    raw: Any = None
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def raise_for_error(self) -> None:
        if self.is_error:
            raise RuntimeError(self.error)
