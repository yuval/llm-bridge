"""
Example: Using OpenAI reasoning models with reasoning_effort and verbosity.

Demonstrates:
1. Varying reasoning_effort levels.
2. Varying verbosity levels.

Requirements:
- OPENAI_API_KEY environment variable
"""

import asyncio
from llm_bridge import create_llm, Provider, ChatResponse


async def reasoning_with_efforts():
    print("=== Reasoning Effort Levels ===\n")

    async with create_llm(Provider.OPENAI, "gpt-5-nano-2025-08-07") as llm:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant skilled at mathematical reasoning.",
            },
            {
                "role": "user",
                "content": (
                    "A train leaves Station A at 2:00 PM traveling at 60 mph toward Station B. "
                    "The stations are 180 miles apart. At what time will the train reach Station B?"
                ),
            },
        ]

        for effort in ["minimal", "medium", "high"]:
            print(f"\n--- Effort: {effort} ---")
            params = dict(
                max_completion_tokens=2500, reasoning_effort=effort, verbosity="medium"
            )

            try:
                resp: ChatResponse = await llm.chat(messages, params=params)
                print(
                    resp.content
                    if not resp.is_error
                    else f"Error: {resp.error}"
                )
            except Exception as e:
                print(f"Exception: {e}")


async def reasoning_with_verbosity():
    print("\n\n=== Verbosity Levels ===\n")

    async with create_llm(Provider.OPENAI, "gpt-5-nano-2025-08-07") as llm:
        messages = [
            {"role": "system", "content": "You are a logical reasoning assistant."},
            {
                "role": "user",
                "content": (
                    "In a certain code, HELLO is written as IFMMP. "
                    "Using the same coding pattern, how would WORLD be written?"
                ),
            },
        ]

        for verbosity in ["low", "medium", "high"]:
            print(f"\n--- Verbosity: {verbosity} ---")
            params = dict(
                max_completion_tokens=2500, reasoning_effort="minimal", verbosity=verbosity
            )

            try:
                resp: ChatResponse = await llm.chat(messages, params=params)
                print(
                    resp.content
                    if not resp.is_error
                    else f"Error: {resp.error}"
                )
            except Exception as e:
                print(f"Exception: {e}")


async def main():
    """Run all reasoning examples in sequence."""
    await reasoning_with_efforts()
    await reasoning_with_verbosity()

if __name__ == "__main__":
    print("OpenAI Reasoning Models Demo")
    print("=" * 50)
    
    asyncio.run(main())
