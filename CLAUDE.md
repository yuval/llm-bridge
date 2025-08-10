# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llm-bridge** provides a unified async interface for chat-based LLMs across OpenAI, Anthropic, and Gemini providers. The library standardizes message formats, tool usage, and response handling while exposing provider-specific features when needed.

## Key Architecture

### Simplified Public API
- Two main methods: `client.chat()` and `client.stream()` that return unified `ChatResponse` objects
- Users pass plain `dict[str, Any]` for parameters (no `ChatParams` wrapper needed)
- Factory pattern via `create_llm()` for creating LLM instances
- Provider registry maps `Provider` enum values to concrete implementations

### Adapter Pattern for Provider Logic
- All providers inherit from `BaseAsyncLLM` abstract base class in `client.py`
- Each provider implements `_chat_impl()` and has an associated `adapter` property
- Adapters handle transformation between generic format and provider-specific formats
- Clean separation: business logic in clients, transformations in adapters

### Unified Response System
- Single `ChatResponse` dataclass replaces provider-specific response wrappers
- Consistent interface: `content`, `tool_calls`, `raw`, `error` properties
- Unified error handling across all providers
- Streaming returns the same `ChatResponse` objects

### Parameter Normalization
- `normalize_params()` function processes user parameters
- Standard keys (temperature, max_tokens, etc.) handled uniformly
- Provider-specific parameters moved to `extra` dict
- Defaults applied consistently (stream=False, extra={})

### Adapter Protocol
- `RequestAdapter` protocol defines consistent interface
- Methods: `to_provider()`, `from_provider()`, `stream_text()`, `assistant_message_from()`, `tool_result_message()`
- Enables provider-agnostic message creation
- No conditional logic needed in user code

### Type System
- `ChatMessage` is a flexible `dict[str, Any]` type alias
- `ChatResponse` unified dataclass for all responses
- `ToolCallRequest` and `ToolCallResult` for tool interactions

## Development Commands

### Setup
```bash
poetry install
```

### Linting and Formatting
```bash
# Run linter with auto-fix
ruff check --fix .

# Format code
ruff format .

# Check types
mypy .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_filename.py

# Run with verbose output
pytest -v
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Environment Variables

At least one provider API key is required:
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key  
- `GEMINI_API_KEY` - Gemini API key (OpenAI-compatible endpoint)

## Provider-Specific Features

### Anthropic
- Prompt caching support via `ephemeral()` helper function
- Cache statistics available in raw response object

### OpenAI
- Full OpenAI API compatibility
- Structured outputs via JSON schema in `response_format`
- Support for reasoning models with `reasoning_effort` and `verbosity`

### Gemini
- Uses OpenAI-compatible endpoints (not native Google client)
- Standard OpenAI API features supported

## Code Patterns

### Creating LLM Clients
```python
# Basic usage
llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")

# With custom client
client = AsyncAnthropic(api_key="...", timeout=30.0)
llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022", client=client)
```

### Making Requests
```python
# Simple chat request
messages = [{"role": "user", "content": "Hello"}]
response = await llm.chat(messages, params={"temperature": 0.7})
print(response.content)

# With provider-specific parameters
response = await llm.chat(messages, params={
    "temperature": 0.7,
    "max_tokens": 100,
    "extra": {"reasoning_effort": "high"}  # OpenAI-specific
})
```

### Streaming
```python
messages = [{"role": "user", "content": "Tell me a story"}]
async for chunk in llm.stream(messages, params={"temperature": 0.8}):
    print(chunk.content, end="", flush=True)
```

### Tool Usage with Unified Adapters
```python
# Provider-agnostic tool call handling
response = await llm.chat(messages, params={"tools": tools})

if response.tool_calls:
    adapter = llm.adapter
    
    # Add assistant message with tool calls
    assistant_msg = adapter.assistant_message_from(response.raw)
    messages.append(assistant_msg)
    
    # Add tool results
    for call in response.tool_calls:
        result = execute_tool(call)  # Your tool execution
        result_msg = adapter.tool_result_message(ToolCallResult(call.id, result))
        messages.append(result_msg)
    
    # Continue conversation
    final_response = await llm.chat(messages, params={"tools": tools})
```

### Error Handling
```python
response = await llm.chat(messages)

if response.is_error:
    print(f"Error: {response.error}")
    # Or raise exception
    response.raise_for_error()
else:
    print(response.content)
```

## Repository Structure

- `src/llm_bridge/` - Main package
  - `client.py` - LLM implementations and factory
  - `adapters/` - Provider-specific transformation logic
    - `openai.py` - OpenAI request/response adapter
    - `anthropic.py` - Anthropic request/response adapter  
    - `gemini.py` - Gemini adapter (re-exports OpenAI)
  - `response.py` - Unified ChatResponse dataclass
  - `params.py` - Parameter normalization functions
  - `types.py` - Type definitions (tool types, ChatMessage)
  - `provider.py` - Provider enum and utilities
  - `errors.py` - Error classification
- `examples/` - Usage examples including prompt caching, tools, structured output
- `tests/` - Test Suite with parameter normalization and adapter tests

## Migration Guide

### From Old API to New API

**Parameters:**
```python
# Old
params = ChatParams(temperature=0.7, max_tokens=100)
response = await llm.chat(messages, params=params)

# New  
params = {"temperature": 0.7, "max_tokens": 100}
response = await llm.chat(messages, params=params)
```

**Response Access:**
```python
# Old
content = response.get_response_content()
tool_calls = response.get_tool_calls()

# New
content = response.content
tool_calls = response.tool_calls
```

**Message Creation:**
```python
# Old - Provider-specific conditionals
if provider == Provider.ANTHROPIC:
    messages.append(create_anthropic_message(...))
else:
    messages.append(create_openai_message(...))

# New - Unified adapter methods
adapter = llm.adapter
assistant_msg = adapter.assistant_message_from(response.raw)
tool_msg = adapter.tool_result_message(tool_result)
```

## Current Limitations

- Only chat completions supported (no embeddings/completions)
- Gemini uses OpenAI-compatible endpoints only
- Web search and server-side tools not yet implemented