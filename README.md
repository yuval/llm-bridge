# llm-bridge

**llm-bridge** offers a unified async interface for calling chat-based LLMs across OpenAI, Anthropic, and Gemini. It standardizes message formats, tool usage, and response handling, while exposing provider-specific features when needed.

## ✨ Key Features

- **Unified API**: Single interface across OpenAI, Anthropic, and Gemini
- **Simple Parameters**: Pass plain dictionaries - no wrapper classes needed  
- **Consistent Responses**: Same `ChatResponse` object for all providers
- **Provider-Agnostic Tools**: No conditional logic needed for different providers
- **Type Safety**: Full TypeScript-style type hints
- **Streaming Support**: Unified streaming interface
- **Error Handling**: Consistent error wrapping across providers

> ⚠️ This project is still a **Work in Progress**. Some features (like web search tools) are not yet implemented.

---

## 🔧 Installation

```bash
poetry install

# Set environment variables for your API keys (at least one is required):
# You can also include these in an .env file

# OpenAI
export OPENAI_API_KEY=sk-...
# Anthropic  
export ANTHROPIC_API_KEY=...
# Gemini (OpenAI-compatible)
export GEMINI_API_KEY=...
```

## 🚀 Basic Usage

Creating an LLM client is simple:

```python
from llm_bridge import create_llm, Provider

# Create any provider with the same interface
anthropic_llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")
openai_llm = create_llm(Provider.OPENAI, "gpt-4o-mini")  
gemini_llm = create_llm(Provider.GEMINI, "gemini-2.0-flash-lite")
```

## 🤖 Simple Chat Example

```python
import asyncio
from llm_bridge import create_llm, Provider

async def chat_example():
    llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your name?"},
    ]
    
    # Simple dict for parameters - no wrapper classes needed!
    response = await llm.chat(messages, params={
        "max_tokens": 1000, 
        "temperature": 0.7
    })
    
    print(f"Response: {response.content}")
    print(f"Error: {response.is_error}")

if __name__ == "__main__":
    asyncio.run(chat_example())
```

## 📡 Streaming Example

```python
async def streaming_example():
    llm = create_llm(Provider.OPENAI, "gpt-4o-mini")
    
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Use the dedicated stream method
    async for chunk in llm.stream(messages, params={"temperature": 0.8}):
        print(chunk.content, end="", flush=True)
```

## 🛠️ Tool Calling (Provider-Agnostic)

The unified adapter system eliminates provider-specific logic:

```python
from llm_bridge import create_llm, Provider, ToolCallResult

async def tool_example():
    # Works the same for ANY provider!
    llm = create_llm(Provider.OPENAI, "gpt-4o-mini")  # or ANTHROPIC, GEMINI
    
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
    
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    
    # Initial request with tools
    response = await llm.chat(messages, params={"tools": tools})
    
    if response.tool_calls:
        # Unified adapter - no provider checks needed!
        adapter = llm.adapter
        
        # Add assistant message with tool calls
        assistant_msg = adapter.assistant_message_from(response.raw)
        messages.append(assistant_msg)
        
        # Execute tools and add results
        for call in response.tool_calls:
            result = f"Weather in {call.arguments['location']}: 72°F, sunny"
            result_msg = adapter.tool_result_message(ToolCallResult(call.id, result))
            messages.append(result_msg)
        
        # Final response
        final_response = await llm.chat(messages, params={"tools": tools})
        print(final_response.content)
```

See `examples/basic_tool.py` for a full working demonstration.

## 📊 Structured Output (JSON Schema)

```python
response = await llm.chat(messages, params={
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "math_solution",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "number"}
                },
                "required": ["steps", "answer"]
            },
            "strict": True
        }
    }
})
```

See `examples/structured_output.py` for a complete example.

## ⚡ Provider-Specific Features

### Anthropic Prompt Caching

Reduce latency and costs with Anthropic's prompt caching:

```python
from llm_bridge.adapters.anthropic import ephemeral

messages = [
    {
        "role": "system", 
        "content": [ephemeral("Your large system prompt here...")]
    },
    {"role": "user", "content": "Your question"}
]

response = await anthropic_llm.chat(messages, params={"max_tokens": 500})

# Cache stats available in raw response
print(response.raw.usage)
```

### OpenAI Reasoning Models

Use reasoning models with effort and verbosity controls:

```python
response = await openai_llm.chat(messages, params={
    "reasoning_effort": "high",
    "verbosity": "medium", 
    "max_tokens": 2500
})
```

## 🔄 Using Custom SDK Clients

You can configure clients yourself and pass them to the factory:

```python
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# Custom Anthropic client
anthropic_client = AsyncAnthropic(
    api_key="your-key",
    timeout=30.0,
    max_retries=3
)
llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022", client=anthropic_client)

# Custom OpenAI client  
openai_client = AsyncOpenAI(timeout=60.0)
llm = create_llm(Provider.OPENAI, "gpt-4o-mini", client=openai_client)
```

## 🚨 Error Handling

```python
response = await llm.chat(messages)

if response.is_error:
    print(f"Error occurred: {response.error}")
    # Optionally raise an exception
    response.raise_for_error()
else:
    print(f"Success: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
```

## 🏗️ Architecture

**llm-bridge** uses a clean adapter pattern:

- **Clients** (`llm.chat()`, `llm.stream()`) - Unified interface for all providers
- **Adapters** - Handle provider-specific request/response transformations  
- **Normalization** - Standard parameter handling with provider-specific extras
- **Unified Responses** - Single `ChatResponse` class across all providers

This eliminates conditional logic in your code and makes adding new providers seamless.

## 📚 Examples

The `examples/` directory contains comprehensive demonstrations:

- `basic_chat.py` - Simple chat across all providers
- `basic_tool.py` - Tool calling with unified adapters
- `structured_output.py` - JSON schema responses  
- `anthropic_prompt_caching.py` - Prompt caching with Anthropic
- `openai_reasoning.py` - Reasoning models with OpenAI

## 🏷️ Limitations

❌ Only chat completions are currently supported  
❌ Native Gemini client (google-generativeai) is not yet supported  
❌ Web search and other server-side tools are not yet implemented

## 📄 License

This project is licensed under the MIT License.