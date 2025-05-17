import asyncio
from llm_bridge import ChatParams
from llm_bridge.factory import create_llm
from llm_bridge.providers import Provider
from llm_bridge.responses import BaseChatResponse


async def chat_example():
    openai_llm = create_llm(Provider.OPENAI, "gpt-4.1-nano-2025-04-14")
    anthropic_llm = create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022")
    gemini_llm = create_llm(Provider.GEMINI, "gemini-2.0-flash-lite")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your name?"},
    ]

    params = ChatParams(max_tokens=1000, temperature=0.7)

    openai_response: BaseChatResponse = await openai_llm.chat(messages, params=params)
    anthropic_response: BaseChatResponse = await anthropic_llm.chat(messages, params=params)
    gemini_response: BaseChatResponse = await gemini_llm.chat(messages, params=params)

    print("OpenAI: ", openai_response.get_response_content())
    print("Anthropic: ", anthropic_response.get_response_content())
    print("Gemini: ", gemini_response.get_response_content())


if __name__ == "__main__":
    asyncio.run(chat_example())
