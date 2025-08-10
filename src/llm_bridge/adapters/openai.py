"""OpenAI adapter for pure request/response transformations."""

from __future__ import annotations

import json
from typing import Any, Sequence

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_bridge.response import ChatResponse
from llm_bridge.types import ToolCallResult, ToolCallRequest
from llm_bridge.params import ChatMessage


class OpenAIRequestAdapter:
    """Adapter for converting between generic format and OpenAI format."""

    def to_provider(
        self, messages: Sequence[ChatMessage], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert generic messages and params to OpenAI request format."""
        # Build messages
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg["role"]}

            # Handle content (required for most message types)
            if msg.get("content") is not None:
                openai_msg["content"] = msg["content"]

            # Handle tool calls (for assistant messages with function calls)
            if msg.get("tool_calls"):
                openai_msg["tool_calls"] = msg["tool_calls"]
                # OpenAI spec: content should be null when tool_calls is present
                if "content" not in openai_msg:
                    openai_msg["content"] = None

            # Handle function call (legacy format, still supported)
            if msg.get("function_call"):
                openai_msg["function_call"] = msg["function_call"]

            # Handle tool call ID (for tool response messages)
            if msg.get("tool_call_id"):
                openai_msg["tool_call_id"] = msg["tool_call_id"]

            # Handle name (for function/tool responses or to identify speakers)
            if msg.get("name"):
                openai_msg["name"] = msg["name"]

            # Ensure content is set for messages that require it
            if (
                "content" not in openai_msg
                and not openai_msg.get("tool_calls")
                and not openai_msg.get("function_call")
            ):
                openai_msg["content"] = ""

            openai_messages.append(openai_msg)

        # Build params from normalized dict
        base_params = dict(params)  # Copy the normalized params

        # Remove fields not accepted by API
        base_params.pop("stream", None)

        # Handle extra params
        extras = base_params.pop("extra", {})

        # Add extra params to base_params
        for k, v in extras.items():
            base_params.setdefault(k, v)

        return {"messages": openai_messages, **base_params}

    def from_provider(self, raw: ChatCompletion) -> ChatResponse:
        """Convert OpenAI response to unified ChatResponse."""
        content = ""
        tool_calls = None

        if raw.choices and raw.choices[0].message:
            message = raw.choices[0].message
            content = message.content or ""

            # Extract tool calls
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    raw_args = tc.function.arguments
                    arguments = {}

                    if isinstance(raw_args, dict):
                        arguments = raw_args
                    elif isinstance(raw_args, str) and raw_args.strip():
                        try:
                            arguments = json.loads(raw_args)
                        except json.JSONDecodeError:
                            continue  # Skip malformed tool calls

                    tool_calls.append(
                        ToolCallRequest(
                            id=tc.id, name=tc.function.name, arguments=arguments
                        )
                    )

        return ChatResponse(content=content, tool_calls=tool_calls, raw=raw)

    def stream_text(self, raw_chunk: ChatCompletionChunk) -> ChatResponse:
        """Extract content from streaming chunk."""
        content = ""
        if raw_chunk.choices and raw_chunk.choices[0].delta:
            content = raw_chunk.choices[0].delta.content or ""

        return ChatResponse(content=content, raw=raw_chunk)

    def assistant_message_from(self, raw: ChatCompletion) -> ChatMessage:
        """Convert OpenAI response to assistant ChatMessage."""
        if not raw.choices or not raw.choices[0].message:
            return {"role": "assistant", "content": ""}
        
        message = raw.choices[0].message
        chat_message: ChatMessage = {"role": "assistant"}
        
        # Add content if present
        if message.content:
            chat_message["content"] = message.content
        
        # Add tool calls if present
        if message.tool_calls:
            chat_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
            # OpenAI spec: content should be null when tool_calls is present
            if "content" not in chat_message:
                chat_message["content"] = None
        
        return chat_message

    def tool_result_message(self, result: ToolCallResult) -> ChatMessage:
        """Convert ToolCallResult to OpenAI ChatMessage."""
        return {
            "role": "tool",
            "tool_call_id": result.id,
            "content": str(result.content)
            if not isinstance(result.content, str)
            else result.content,
        }
