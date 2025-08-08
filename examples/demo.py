#!/usr/bin/env -S poetry run python
"""This script demonstrates three common scenarios:

1. A *standard* non‑streaming chat completion.
2. A *streaming* chat completion, yielding incremental deltas.
3. Accessing the *raw* OpenAI response object (e.g. to inspect headers / ids).

Execute directly, supplying an ``OPENAI_API_KEY`` in your environment.

Example::

    $ OPENAI_API_KEY=sk‑… ./openai_chat_examples.py

"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

from llm_bridge import ChatParams, Provider, create_llm

_MODEL: Final[str] = "gpt-4o-mini"
_LOGGER = logging.getLogger("examples.openai_chat_examples")
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def standard_request() -> None:
    """Run a single, non‑streaming chat completion and print the content."""
    _LOGGER.info("----- standard request -----")

    llm = create_llm(Provider.OPENAI, _MODEL)
    messages = [
        {"role": "user", "content": "Say this is a test"},
    ]

    response = await llm.chat(messages)
    _LOGGER.info(response.get_response_content())


async def streaming_request() -> None:
    """Stream a completion and print tokens as they arrive."""
    _LOGGER.info("----- streaming request -----")

    llm = create_llm(Provider.OPENAI, _MODEL)
    messages = [
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ]
    params = ChatParams(stream=True)

    stream = await llm.chat(messages, params=params)
    async for chunk in stream:  # type: ignore[func-returns-value]
        if chunk.is_error:
            continue  # ignore errors for brevity – production code should handle
        print(chunk.get_response_content(), end="", flush=True)
    print()


async def raw_response_headers() -> None:
    """Retrieve the raw :class:`openai.types.ChatCompletion` and show its metadata."""
    _LOGGER.info("----- raw response headers test -----")

    llm = create_llm(Provider.OPENAI, _MODEL)
    messages = [
        {"role": "user", "content": "Say this is a test"},
    ]

    wrapper = await llm.chat(messages)
    completion = wrapper.raw_response  # the original OpenAI object

    # The OpenAI SDK does not surface HTTP headers on the completion object, but
    # you can still access identifiers such as ``id`` and ``model`` for tracing.
    _LOGGER.info("request_id: %s", getattr(completion, "id", "<missing>"))
    _LOGGER.info(wrapper.get_response_content())


async def _main() -> None:
    await standard_request()
    await streaming_request()
    await raw_response_headers()


if __name__ == "__main__":
    asyncio.run(_main())
