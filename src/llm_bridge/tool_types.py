"""
Provider‑neutral dataclasses for client‑side tool use.

They are intentionally minimal: everything provider‑specific lives in adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ToolCallRequest", "ToolCallResult"]


@dataclass(slots=True)
class ToolCallRequest:
    """A model‑agnostic request emitted by the LLM to call a local tool."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallResult:
    """Payload to send back to the LLM after the tool finished running."""
    id: str                     # must match the request id
    content: str | dict[str, Any]
