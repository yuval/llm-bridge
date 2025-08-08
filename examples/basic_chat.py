import asyncio

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from llm_bridge import ChatParams
from llm_bridge.factory import create_llm
from llm_bridge.providers import Provider, get_api_key
from llm_bridge.responses import BaseChatResponse


async def chat_example_default_client():
    openai_llm = create_llm(Provider.OPENAI, "gpt-5-nano-2025-08-07")
    anthropic_llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")
    gemini_llm = create_llm(Provider.GEMINI, "gemini-2.0-flash-lite")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your name?"},
    ]

    params = ChatParams(max_tokens=1000, temperature=0.7)

    openai_response: BaseChatResponse = await openai_llm.chat(messages, params=params)
    anthropic_response: BaseChatResponse = await anthropic_llm.chat(
        messages, params=params
    )
    gemini_response: BaseChatResponse = await gemini_llm.chat(messages, params=params)

    print("OpenAI: ", openai_response.get_response_content())
    print("Anthropic: ", anthropic_response.get_response_content())
    print("Gemini: ", gemini_response.get_response_content())


async def chat_example_pass_client():
    openai_client = AsyncOpenAI(max_retries=3, timeout=10)  # Example OpenAI client
    anthropic_client = AsyncAnthropic()
    gemini_client = AsyncOpenAI(
        api_key=get_api_key(Provider.GEMINI),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    openai_llm = create_llm(
        Provider.OPENAI, "gpt-5-nano-2025-08-07", client=openai_client
    )
    anthropic_llm = create_llm(
        Provider.ANTHROPIC, "claude-3-5-haiku-20241022", client=anthropic_client
    )
    gemini_llm = create_llm(
        Provider.GEMINI, "gemini-2.0-flash-lite", client=gemini_client
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your name?"},
    ]

    params = ChatParams(max_tokens=1000, temperature=0.7)

    openai_response: BaseChatResponse = await openai_llm.chat(messages, params=params)
    anthropic_response: BaseChatResponse = await anthropic_llm.chat(
        messages, params=params
    )
    gemini_response: BaseChatResponse = await gemini_llm.chat(messages, params=params)

    print("OpenAI: ", openai_response.get_response_content())
    print("Anthropic: ", anthropic_response.get_response_content())
    print("Gemini: ", gemini_response.get_response_content())

    print("OpenAI: ", openai_response.raw_response.usage)
    print("Anthropic: ", anthropic_response.raw_response.usage)
    print("Gemini: ", gemini_response.raw_response.usage)


if __name__ == "__main__":
    asyncio.run(chat_example_default_client())
    asyncio.run(chat_example_pass_client())
