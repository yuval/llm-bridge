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

```
import asyncio
from llm_bridge import Provider, create_llm, ChatParams

async def main():
    llm = create_llm(Provider.OPENAI, "gpt-4")
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    response = await llm.chat(messages)
    print(response.get_response_content())

asyncio.run(main())
```

# Streaming Example

```
params = ChatParams(stream=True)
stream = await llm.chat(messages, params=params)
async for chunk in stream:
    print(chunk.get_response_content(), end="", flush=True)
```

# Tool Calling

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

# Structured Output (JSON Schema)

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


# License

MIT © 2025 Yuval Merhav