import asyncio
from llm_bridge import ChatParams
from llm_bridge.factory import create_llm
from llm_bridge.providers import Provider


async def basic_chat_example():
    openai_llm = create_llm(Provider.OPENAI, "gpt-4.1-nano-2025-04-14")
    anthropic_llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")

    messages = [
        {"role": "user", "content": "What's your name?"},
    ]

    params = ChatParams(max_tokens=1000, temperature=0.7)

    openai_response = await openai_llm.chat(messages, params=params)
    anthropic_response = await anthropic_llm.chat(messages, params=params)

    print("OpenAI: ", openai_response.get_response_content())
    print("Anthropic: ", anthropic_response.get_response_content())


async def streaming_example():
    llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")

    messages = [
        {"role": "user", "content": "Tell me a short story about AI, not longer than one paragraph."},
    ]

    params = ChatParams(stream=True, max_tokens=500)

    response_stream = await llm.chat(messages, params=params)

    print("Streaming response:")
    async for chunk in response_stream:
        if not chunk.is_error:
            content = chunk.get_response_content()
            print(content, end='', flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(basic_chat_example())
    print("\n" + "=" * 50 + "\n")
    asyncio.run(streaming_example())
