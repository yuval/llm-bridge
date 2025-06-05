#!/usr/bin/env -S poetry run python
"""
Simple Anthropic Prompt Caching Example

Demonstrates prompt caching with ephemeral() to reduce latency and costs.
Execute with: ANTHROPIC_API_KEY=sk-... python examples/prompt_caching.py
"""
import asyncio
import logging

from llm_bridge import ChatParams, Provider, create_llm
from llm_bridge.providers.anthropic import ephemeral
from llm_bridge.responses import AnthropicResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_large_system_prompt():
    """Generate a large system prompt to ensure caching threshold is met."""
    base_prompt = "You are an expert code reviewer. "
    
    # Generate repetitive content to reach token threshold
    topics = [
        "software architecture patterns", "security vulnerabilities", "performance optimization",
        "code quality standards", "testing methodologies", "database design principles",
        "API design best practices", "error handling strategies", "documentation standards",
        "deployment procedures", "monitoring techniques", "scalability considerations"
    ]
    
    detailed_sections = []
    for i, topic in enumerate(topics * 1):  # Repeat to ensure length
        section = f"""
        Section {i+1}: Analysis of {topic}
        
        When reviewing code for {topic}, consider the following comprehensive checklist:
        - Examine the implementation approach and architectural decisions
        - Evaluate potential risks and mitigation strategies  
        - Assess performance implications and optimization opportunities
        - Review compliance with industry standards and best practices
        - Analyze maintainability and long-term sustainability factors
        - Consider integration points and dependency management
        - Validate error handling and edge case coverage
        - Review documentation completeness and accuracy
        
        Key principles for {topic}:
        - Prioritize clarity and simplicity in design choices
        - Ensure robust error handling and graceful degradation
        - Implement comprehensive logging and monitoring capabilities
        - Follow established conventions and community standards
        - Design for testability and automated verification
        - Consider security implications at every layer
        - Plan for scalability and future requirements
        - Document assumptions and architectural decisions
        """
        detailed_sections.append(section)
    
    return base_prompt + "".join(detailed_sections)


async def main():
    llm = create_llm(Provider.ANTHROPIC, "claude-3-haiku-20240307")
    params = ChatParams(max_tokens=500, temperature=0.7)

    # Generate large system prompt to meet caching threshold
    system_prompt = create_large_system_prompt()
    
    # First request: creates cache
    # Note: empheral() is just a helper that returns this dict for a given text:
    # {
    #     "type": "text",
    #     "text": text,
    #     "cache_control": {"type": "ephemeral"},
    # }
    # This will enable the default 5-minute caching window (i.e., cache dies after 5 minutes of inactivity)
    #
    # Prompt caching references the entire prompt - tools, system, and messages (in that order) up to and 
    # including the block designated with cache_control. You can call ephemeral() on any text message to cache
    # everything up to that point. 
    # Recommended reading: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    messages = [
        {"role": "system", "content": [ephemeral(system_prompt)]},
        {"role": "user", "content": "Review: def add(a, b): return a + b"}
    ]
    
    response1: AnthropicResponse = await llm.chat(messages, params=params)
    logger.info(f"Cache created: {response1.cache_creation_input_tokens} tokens")
    
    # Second request: uses cache
    messages[1]["content"] = "Review: def divide(a, b): return a / b"
    
    response2: AnthropicResponse = await llm.chat(messages, params=params)
    logger.info(f"Cache read: {response2.cache_read_input_tokens} tokens")
    
    print(f"\nResponse: {response2.get_response_content()}")


if __name__ == "__main__":
    asyncio.run(main())