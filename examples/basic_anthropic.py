"""Basic Anthropic chat example demonstrating the unified llm-bridge interface."""

import asyncio
from anthropic import AsyncAnthropic
from llm_bridge import create_llm, Provider, get_api_key


async def basic_anthropic_example():
    """Simple chat example using default Anthropic client."""
    print("=== Basic Anthropic Chat Example ===")
    
    # Create LLM with default client and use as async context manager
    async with create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022") as llm:
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
            print(f"🤖 Anthropic Response: {response.content}")
            print(f"📊 Usage: {response.raw.usage}")


async def custom_anthropic_client_example():
    """Chat example using a custom configured Anthropic client."""
    print("\n=== Custom Anthropic Client Example ===")
    
    # Configure your own Anthropic client
    custom_client = AsyncAnthropic(
        api_key=get_api_key(Provider.ANTHROPIC),  # Or set directly
        timeout=60.0,
        max_retries=2
    )
    
    # Pass the configured client to llm-bridge and use as async context manager
    async with create_llm(Provider.ANTHROPIC, "claude-3-5-sonnet-20241022", client=custom_client) as llm:
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
            print(f"🤖 Anthropic Response: {response.content}")


async def streaming_anthropic_example():
    """Demonstrate streaming responses with Anthropic."""
    print("\n=== Streaming Anthropic Example ===")
    
    # Use async context manager for automatic cleanup
    async with create_llm(Provider.ANTHROPIC, "claude-3-5-haiku-20241022") as llm:
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
    await basic_anthropic_example()
    await custom_anthropic_client_example()
    await streaming_anthropic_example()

if __name__ == "__main__":
    asyncio.run(main())