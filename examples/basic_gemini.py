"""Basic Gemini chat example demonstrating the unified llm-bridge interface."""

import asyncio
from openai import AsyncOpenAI
from llm_bridge import create_llm, Provider, get_api_key


async def basic_gemini_example():
    """Simple chat example using default Gemini client."""
    print("=== Basic Gemini Chat Example ===")
    
    # Create LLM with default client and use as async context manager
    async with create_llm(Provider.GEMINI, "gemini-2.0-flash-lite") as llm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
        ]
        
        # Simple dict parameters - no wrapper classes needed!
        response = await llm.chat(messages, params={
            "max_tokens": 150,
            "temperature": 0.7
        })
        
        if response.is_error:
            print(f"❌ Error: {response.error}")
        else:
            print(f"🤖 Gemini Response: {response.content}")
            print(f"📊 Usage: {response.raw.usage}")


async def custom_gemini_client_example():
    """Chat example using a custom configured Gemini client."""
    print("\n=== Custom Gemini Client Example ===")
    
    # Gemini uses OpenAI-compatible endpoints, so we use AsyncOpenAI client
    custom_client = AsyncOpenAI(
        api_key=get_api_key(Provider.GEMINI),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=30.0,
        max_retries=2
    )
    
    # Pass the configured client to llm-bridge and use as async context manager
    async with create_llm(Provider.GEMINI, "gemini-2.0-flash-exp", client=custom_client) as llm:
        messages = [
            {"role": "user", "content": "What is the capital of Spain?"}
        ]
        
        response = await llm.chat(messages, params={
            "max_tokens": 200,
            "temperature": 0.3
        })
        
        if response.is_error:
            print(f"❌ Error: {response.error}")
        else:
            print(f"🤖 Gemini Response: {response.content}")


async def streaming_gemini_example():
    """Demonstrate streaming responses with Gemini."""
    print("\n=== Streaming Gemini Example ===")
    
    # Use async context manager for automatic cleanup
    async with create_llm(Provider.GEMINI, "gemini-2.0-flash-lite") as llm:
        messages = [
            {"role": "user", "content": "Write a short poem about coding."}
        ]
        
        print("🎵 Streaming poem: ", end="")
        async for chunk in llm.stream(messages, params={"temperature": 0.8}):
            if not chunk.is_error:
                print(chunk.content, end="", flush=True)
        print()  # New line after streaming


async def main():
    """Run all examples in sequence."""
    await basic_gemini_example()
    await custom_gemini_client_example()
    await streaming_gemini_example()

if __name__ == "__main__":
    asyncio.run(main())