"""Anthropic adapter for pure request/response transformations."""

from __future__ import annotations

from typing import Any, Sequence

from anthropic.types import Message

from llm_bridge.response import ChatResponse
from llm_bridge.types import ToolCallResult, ToolCallRequest
from llm_bridge.params import ChatMessage


def ephemeral(text: str) -> dict[str, Any]:
    """Return a text block marked for Anthropic's 5‑minute *ephemeral* prompt cache."""
    return {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }


class AnthropicRequestAdapter:
    """Adapter for converting between generic format and Anthropic format."""

    def to_provider(
        self, messages: Sequence[ChatMessage], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert generic messages and params to Anthropic request format."""
        # Build messages
        anthropic_messages: list[dict[str, Any]] = []
        system_prompt = ""

        for msg in messages:
            # Extract system prompt from messages
            if msg["role"] == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    # Handle list of content blocks (including cache control)
                    system_prompt = content
                else:
                    system_prompt = str(content)
                continue

            anthropic_msg: dict[str, Any] = {"role": msg["role"]}

            # Handle content
            if msg.get("content") is not None:
                content = msg["content"]
                if isinstance(content, str):
                    anthropic_msg["content"] = content
                elif isinstance(content, list):
                    anthropic_msg["content"] = content
                else:
                    anthropic_msg["content"] = str(content)

            # Handle tool use responses
            if msg.get("tool_call_id"):
                anthropic_msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg.get("content", ""),
                    }
                ]
                anthropic_msg["role"] = "user"

            anthropic_messages.append(anthropic_msg)

        # Build params from normalized dict
        base_params = dict(params)  # Copy the normalized params

        # Remove fields not accepted by API
        base_params.pop("stream", None)

        # Handle extra params
        extras = base_params.pop("extra", {})

        # Anthropic requires max_tokens
        if "max_tokens" not in base_params:
            base_params["max_tokens"] = 4096

        # Handle stop sequences
        if "stop" in base_params:
            stop = base_params.pop("stop")
            base_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Handle tools
        if "tools" in base_params and base_params["tools"]:
            anthropic_tools = []
            for tool in base_params["tools"]:
                if tool.get("type") == "function":
                    func = tool["function"]
                    anthropic_tools.append(
                        {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {}),
                        }
                    )
                else:
                    anthropic_tools.append(tool)
            base_params["tools"] = anthropic_tools

        # Add extra params to base_params
        for k, v in extras.items():
            base_params.setdefault(k, v)

        return {"system": system_prompt, "messages": anthropic_messages, **base_params}

    def from_provider(self, raw: Message) -> ChatResponse:
        """Convert Anthropic response to unified ChatResponse."""
        content = ""
        tool_calls = None

        if raw.content:
            text_parts = []
            tool_calls_list = []

            for block in raw.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls_list.append(
                        ToolCallRequest(
                            id=block.id,
                            name=block.name,
                            arguments=dict(block.input) if hasattr(block.input, "items") else {},
                        )
                    )

            content = "".join(text_parts)
            tool_calls = tool_calls_list if tool_calls_list else None

        return ChatResponse(content=content, tool_calls=tool_calls, raw=raw)

    def stream_text(self, raw_chunk: Any) -> ChatResponse:
        """Extract content from Anthropic streaming chunk."""
        content = ""

        # Handle streaming events
        if hasattr(raw_chunk, "type"):
            if raw_chunk.type == "content_block_delta":
                if (
                    hasattr(raw_chunk, "delta")
                    and hasattr(raw_chunk.delta, "type")
                    and raw_chunk.delta.type == "text_delta"
                ):
                    content = raw_chunk.delta.text
            elif raw_chunk.type == "text" and hasattr(raw_chunk, "text"):
                content = raw_chunk.text

        return ChatResponse(content=content, raw=raw_chunk)

    def assistant_message_from(self, raw: Message) -> ChatMessage:
        """Convert Anthropic response to assistant ChatMessage."""
        chat_message: ChatMessage = {"role": "assistant"}
        
        if not raw.content:
            chat_message["content"] = ""
            return chat_message
        
        # Extract text and tool use blocks
        text_parts = []
        tool_use_blocks = []
        
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input) if hasattr(block.input, "items") else {}
                })
        
        # Build content - could be string or list depending on whether there are tool uses
        if tool_use_blocks:
            # Mix of text and tool uses - use content list format
            content_list = []
            if text_parts:
                content_list.append({"type": "text", "text": "".join(text_parts)})
            content_list.extend(tool_use_blocks)
            chat_message["content"] = content_list
        else:
            # Just text content
            chat_message["content"] = "".join(text_parts)
        
        return chat_message

    def tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """Convert ToolCallResult to Anthropic ChatMessage."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.content,
                }
            ],
        }
