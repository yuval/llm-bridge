# llm-bridge

**llm-bridge** offers a unified async interface for calling chat-based LLMs across OpenAI, Anthropic, and Gemini. It standardises message formats, tool usage, and response handling, while exposing provider-specific features when needed.

> ⚠️ This project is still a **Work in Progress**. Some features (like web search tools) are not yet implemented.

---

## 🔧 Installation

```bash
poetry install

# Set environment variables for your API keys: (at least one is required)

# OpenAI
export OPENAI_API_KEY=sk-...
# Anthropic
export ANTHROPIC_API_KEY=...
# Gemini (OpenAI-compatible)
export GEMINI_API_KEY=...
```

# Basic Usage

## Simple Chat Example

```
import asyncio
from llm_bridge import ChatParams
from llm_bridge.factory import create_llm
from llm_bridge.providers import Provider
from llm_bridge.responses import BaseChatResponse


async def chat_example():
    openai_llm = create_llm(Provider.OPENAI, "gpt-4.1-nano-2025-04-14")
    anthropic_llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")
    gemini_llm = create_llm(Provider.GEMINI, "gemini-2.0-flash-lite")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your name?"},
    ]

    params = ChatParams(max_tokens=1000, temperature=0.7)

    openai_response: BaseChatResponse = await openai_llm.chat(messages, params=params)
    anthropic_response: BaseChatResponse = await anthropic_llm.chat(messages, params=params)
    gemini_response: BaseChatResponse = await gemini_llm.chat(messages, params=params)

    print("OpenAI: ", openai_response.get_response_content())
    print("Anthropic: ", anthropic_response.get_response_content())
    print("Gemini: ", gemini_response.get_response_content())


if __name__ == "__main__":
    asyncio.run(chat_example())
```

## Streaming Example

```
params = ChatParams(stream=True)
stream = await llm.chat(messages, params=params)
async for chunk in stream:
    print(chunk.get_response_content(), end="", flush=True)
```

## Tool Calling

```
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

params = ChatParams(tools=tools)

```

See examples/basic_tool.py for a full working demo.

## Structured Output (JSON Schema)

```
params = ChatParams(response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "math_solution",
        "schema": {...},
        "strict": True
    }
})
response = await llm.chat(messages, params=params)
```

See examples/structured_output.py for an example solving equations step-by-step.

# Limitations

❌ Only Chat completions is currently supported

❌ Native Gemini client (google-generativeai) is not yet supported

❌ Web search and other server-side tools are not yet implemented
