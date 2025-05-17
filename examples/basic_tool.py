from __future__ import annotations

import argparse
import asyncio
import logging

from llm_bridge.factory import create_llm
from llm_bridge.providers import Provider
from llm_bridge.providers.anthropic import AnthropicRequestAdapter
from llm_bridge.providers.openai import OpenAIRequestAdapter, create_openai_message_dict
from llm_bridge.types.chat import ChatMessage, ChatParams
from llm_bridge.types.tool import ToolCallRequest, ToolCallResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Anthropic‐style tool definition
ANTHROPIC_WEATHER_TOOL: dict[str, object] = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["location"],
    },
}

# OpenAI/Gemini‐style function schema

OPENAI_WEATHER_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": ["string", "null"],
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Units (celsius or fahrenheit); pass null for default.",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

def run_local_tool(req: ToolCallRequest) -> str:
    """Stub implementation of get_weather."""
    # imagine we call a real weather API here
    return "15 °C, mostly cloudy"


async def single_tool_roundtrip(provider: Provider, model: str) -> None:
    """
    Run a single tool‐calling roundtrip with the given provider + model.
    
    1) Send user prompt
    2) Let model emit a tool call
    3) Execute stub tool, re‐inject call + result
    4) Ask model to finish using tool result
    """
    tools = (
        [ANTHROPIC_WEATHER_TOOL]
        if provider == Provider.ANTHROPIC
        else [OPENAI_WEATHER_TOOL]
    )

    llm = create_llm(provider, model)
    messages: list[ChatMessage] = [
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]
    params = ChatParams(tools=tools)

    # Step 1 → get first response
    rsp1 = await llm.chat(messages, params=params)

    # Step 2 → extract tool/function calls using the new wrapper method
    calls = rsp1.get_tool_calls()

    if not calls:
        logger.warning(f"Model answered directly: {rsp1.get_response_content()}")
        return

    # Step 3 → for each call, re‐inject the call then the result
    for call in calls:
        if provider == Provider.ANTHROPIC:
            adapter = AnthropicRequestAdapter()
            # re‐inject tool_use
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    ],
                }
            )
            # inject result block
            messages.append(
                adapter.build_tool_result_message(
                    ToolCallResult(call.id, run_local_tool(call))
                )
            )
        else:
            adapter = OpenAIRequestAdapter()
            # re‐inject function‐call message
            call_msg = create_openai_message_dict(rsp1)
            if call_msg:
                messages.append(call_msg)
            # inject function result
            messages.append(
                adapter.build_tool_result_message(
                    ToolCallResult(call.id, run_local_tool(call))
                )
            )

    # Step 4 → final completion
    rsp2 = await llm.chat(messages, params=ChatParams(tools=tools))
    logger.info("%s says: %s", provider.value.capitalize(), rsp2.get_response_content())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=[p.value for p in Provider],
        default=Provider.ANTHROPIC.value,
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-haiku-20241022" # "gpt-4.1-nano-2025-04-14", "gemini-2.0-flash-lite", "claude-3-5-haiku-20241022"
    )
    args = parser.parse_args()

    asyncio.run(single_tool_roundtrip(Provider(args.provider), args.model))