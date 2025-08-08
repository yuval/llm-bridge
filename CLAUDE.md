# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llm-bridge** provides a unified async interface for chat-based LLMs across OpenAI, Anthropic, and Gemini providers. The library standardizes message formats, tool usage, and response handling while exposing provider-specific features when needed.

## Key Architecture

### Factory Pattern
- `create_llm()` in `factory.py` is the main entry point for creating LLM instances
- Supports both default client creation and accepting pre-configured SDK clients
- Provider registry maps `Provider` enum values to concrete implementations

### Provider Implementation Pattern
- All providers inherit from `BaseAsyncLLM` abstract base class in `providers/base.py`
- Each provider implements `_chat_impl()` and defines a `wrapper_class` property
- Providers transform generic `ChatMessage`/`ChatParams` to provider-specific formats
- Response wrappers (extending `BaseChatResponse`) provide unified response handling

### Response Wrapper System
- All responses go through provider-specific wrappers (`OpenAIResponse`, `AnthropicResponse`, `GeminiResponse`)
- Uniform interface for content extraction, tool calls, error handling, and streaming
- Error responses are wrapped consistently across all providers

### Type System
- `ChatMessage` is a flexible `dict[str, Any]` type alias
- `ChatParams` dataclass with utility methods (`copy()`, `as_dict()`)
- `BaseChatResponse` abstract base ensures consistent response interface

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
- Cache statistics available in response (`cache_creation_input_tokens`, `cache_read_input_tokens`)

### OpenAI
- Full OpenAI API compatibility
- Structured outputs via JSON schema in `response_format`

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

### Error Handling
- All errors are caught and wrapped into error response objects
- Check `response.is_error` before processing
- Use `response.raise_for_error()` to propagate exceptions

### Streaming
```python
params = ChatParams(stream=True)
stream = await llm.chat(messages, params=params)
async for chunk in stream:
    print(chunk.get_response_content(), end="")
```

## Repository Structure

- `src/llm_bridge/` - Main package
  - `factory.py` - LLM creation factory
  - `providers/` - Provider implementations (base, openai, anthropic, gemini)
  - `types/` - Type definitions (chat, tool)
  - `responses.py` - Response wrapper implementations
- `examples/` - Usage examples including prompt caching, tools, structured output
- `tests/` - Test Suite (currently minimal structure)

## Current Limitations

- Only chat completions supported (no embeddings/completions)
- Gemini uses OpenAI-compatible endpoints only
- Web search and server-side tools not yet implemented