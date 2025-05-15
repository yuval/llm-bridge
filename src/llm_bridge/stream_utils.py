"""Shared streaming utilities for LLM providers."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, TypeVar, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)

__all__ = ["aggregate_openai_stream", "aggregate_openai_stream_sync"]

T = TypeVar('T')


async def aggregate_openai_stream(
    chunks: AsyncGenerator[ChatCompletionChunk, None],
    model: str,
) -> ChatCompletion:
    """
    Pure function that aggregates a stream of ChatCompletionChunks into a single ChatCompletion.
    
    Args:
        chunks: Async generator of ChatCompletionChunk objects
        model: Model name for the aggregated response
        
    Returns:
        A complete ChatCompletion object with aggregated content
    """
    full_content = ""
    tool_calls_agg: List[Dict[str, Any]] = []
    final_role: Optional[str] = "assistant"
    finish_reason: Optional[str] = "stop"
    system_fingerprint: Optional[str] = None
    first_chunk_id: Optional[str] = None
    created_timestamp: Optional[int] = None

    async for chunk in chunks:
        if not first_chunk_id and chunk.id:
            first_chunk_id = chunk.id
        if not created_timestamp and chunk.created:
            created_timestamp = chunk.created
        if chunk.system_fingerprint:
            system_fingerprint = chunk.system_fingerprint

        if not chunk.choices:
            continue
        
        delta = chunk.choices[0].delta
        if delta.role:
            final_role = delta.role
        if delta.content:
            full_content += delta.content
        
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                while len(tool_calls_agg) <= tc_chunk.index:
                    tool_calls_agg.append({
                        "id": "", "type": "function", 
                        "function": {"name": "", "arguments": ""}
                    })
                
                agg_tc = tool_calls_agg[tc_chunk.index]
                if tc_chunk.id: 
                    agg_tc["id"] = tc_chunk.id
                if tc_chunk.type: 
                    agg_tc["type"] = tc_chunk.type
                if tc_chunk.function:
                    if tc_chunk.function.name: 
                        agg_tc["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments: 
                        agg_tc["function"]["arguments"] += tc_chunk.function.arguments
        
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    # Build final tool calls
    final_tool_calls: List[ChatCompletionMessageToolCall] = []
    if tool_calls_agg:
        for tc_data in tool_calls_agg:
            if not tc_data["id"] or not tc_data["function"]["name"]:
                continue
            final_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc_data["id"],
                    type=cast(Literal["function"], tc_data["type"]),
                    function=ChatCompletionMessageToolCallFunction(
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"]
                    )
                )
            )

    message_content = full_content if not final_tool_calls else None
    if not full_content and not final_tool_calls and final_role == "assistant":
        message_content = ""

    choice = Choice(
        finish_reason=cast(
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"], 
            finish_reason
        ),
        index=0,
        message=ChatCompletionMessage(
            role=cast(Literal["assistant"], final_role),
            content=message_content,
            tool_calls=final_tool_calls if final_tool_calls else None
        )
    )
    
    return ChatCompletion(
        id=first_chunk_id or "aggregated_stream",
        choices=[choice],
        created=created_timestamp or 0,
        model=model,
        object="chat.completion",
        system_fingerprint=system_fingerprint
    )


def aggregate_openai_stream_sync(
    chunks_gen: Any,  # Generator that yields ChatCompletionChunk
    model: str,
) -> ChatCompletion:
    """
    Synchronous wrapper around the async aggregation function.
    
    Args:
        chunks_gen: Synchronous generator of ChatCompletionChunk objects
        model: Model name for the aggregated response
        
    Returns:
        A complete ChatCompletion object with aggregated content
    """
    async def async_gen_wrapper():
        for chunk in chunks_gen:
            yield chunk
    
    return asyncio.run(aggregate_openai_stream(async_gen_wrapper(), model))