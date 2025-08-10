"""Basic OpenAI chat example demonstrating the unified llm-bridge interface."""

import asyncio
from openai import AsyncOpenAI
from llm_bridge import create_llm, Provider, get_api_key

MODEL_NAME="gpt-4.1-nano"

async def basic_openai_example():
    """Simple chat example using default OpenAI client."""
    print("=== Basic OpenAI Chat Example ===")
    
    # Create LLM with default client and use as async context manager
    async with create_llm(Provider.OPENAI, MODEL_NAME) as llm:
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
            print(f"🤖 OpenAI Response: {response.content}")
            print(f"📊 Usage: {response.raw.usage}")


async def custom_openai_client_example():
    """Chat example using a custom configured OpenAI client."""
    print("\n=== Custom OpenAI Client Example ===")
    
    # Configure your own OpenAI client
    custom_client = AsyncOpenAI(
        api_key=get_api_key(Provider.OPENAI),  # Or set directly
        timeout=30.0,
        max_retries=3
    )
    
    # Pass the configured client to llm-bridge and use as async context manager
    async with create_llm(Provider.OPENAI, MODEL_NAME, client=custom_client) as llm:
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
            print(f"🤖 OpenAI Response: {response.content}")


async def streaming_openai_example():
    """Demonstrate streaming responses with OpenAI."""
    print("\n=== Streaming OpenAI Example ===")
    
    # Use async context manager for automatic cleanup
    async with create_llm(Provider.OPENAI, MODEL_NAME) as llm:
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
    await basic_openai_example()
    await custom_openai_client_example()
    await streaming_openai_example()

if __name__ == "__main__":
    asyncio.run(main())
