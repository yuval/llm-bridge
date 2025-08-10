"""Gemini adapter for pure request/response transformations.

Since Gemini uses OpenAI-compatible endpoints, we just re-export the OpenAI adapter.
"""

from .openai import OpenAIRequestAdapter

# Gemini uses the OpenAI-compatible API, so it's the same adapter
GeminiRequestAdapter = OpenAIRequestAdapter
