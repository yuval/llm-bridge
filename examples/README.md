# Examples

This directory contains comprehensive examples demonstrating the llm-bridge unified API across different providers and use cases.

## 🚀 Basic Provider Examples

### Provider-Specific Basics
- **`basic_openai.py`** - OpenAI/ChatGPT examples with async context managers
- **`basic_anthropic.py`** - Anthropic Claude examples with proper client cleanup
- **`basic_gemini.py`** - Google Gemini examples via OpenAI-compatible endpoint

Each shows:
- Basic chat with default and custom clients
- Streaming responses  
- Async context manager usage for proper cleanup
- Consistent prompts and error handling

## 🛠️ Advanced Use Cases

### Tool Calling
- **`basic_tool.py`** - **Provider-agnostic tool calling** using unified adapters
- Shows the same code working across OpenAI, Anthropic, and Gemini
- Demonstrates `adapter.assistant_message_from()` and `adapter.tool_result_message()`

### Structured Output  
- **`structured_output.py`** - JSON schema responses for structured data
- Mathematical problem solving with step-by-step format

### Provider-Specific Features
- **`anthropic_prompt_caching.py`** - Reduce costs with Anthropic's prompt caching
- **`openai_reasoning.py`** - OpenAI reasoning models (gpt-5) with effort/verbosity controls

## 🎯 Key Patterns Demonstrated

### 1. Async Context Manager Pattern
```python
# Proper client cleanup across all providers
async with create_llm(Provider.OPENAI, "gpt-4o-mini") as llm:
    response = await llm.chat(messages, params={
        "max_completion_tokens": 150,  # Note: reasoning models use max_completion_tokens
        "temperature": 0.7,
        "extra": {"reasoning_effort": "medium"}  # Provider-specific params
    })
```

### 2. Provider-Agnostic Tool Calling
```python
# Works identically for OpenAI, Anthropic, Gemini
async with create_llm(provider, model) as llm:
    adapter = llm.adapter
    assistant_msg = adapter.assistant_message_from(response.raw)
    tool_msg = adapter.tool_result_message(ToolCallResult(call.id, result))
```

### 3. Consistent Error Handling
```python
if response.is_error:
    print(f"Error: {response.error}")
else:
    print(f"Response: {response.content}")
```

### 4. Streaming with Cleanup
```python
async with create_llm(provider, model) as llm:
    async for chunk in llm.stream(messages, params={"temperature": 0.8}):
        print(chunk.content, end="", flush=True)
```

## 📋 Running Examples

Each example can be run independently:

```bash
# Basic provider examples
python examples/basic_openai.py
python examples/basic_anthropic.py  
python examples/basic_gemini.py

# Advanced examples
python examples/basic_tool.py --provider openai --model gpt-4o-mini
python examples/structured_output.py
python examples/anthropic_prompt_caching.py
python examples/openai_reasoning.py
```

## 🔧 Setup Requirements

Set at least one API key:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## 🎉 Benefits Showcased

- **No Provider Conditionals**: Same code works across all providers
- **Proper Resource Management**: Async context managers prevent "Event loop is closed" errors
- **Simple Parameters**: Plain dictionaries instead of wrapper classes
- **Unified Responses**: Consistent `ChatResponse` objects
- **Clean Tool Handling**: Provider-agnostic message creation
- **Easy Streaming**: Dedicated `stream()` method with automatic cleanup
- **Provider Features**: Access to unique capabilities when needed (caching, reasoning, etc.)