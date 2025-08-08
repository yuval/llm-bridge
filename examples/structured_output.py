"""
This example demonstrates how to get structured JSON output from LLMs
by solving a mathematical equation step-by-step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from pydantic import BaseModel

from llm_bridge import ChatParams, Provider, create_llm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Step(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str


# JSON Schema for the response format
MATH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["explanation", "output"],
                "additionalProperties": False,
            },
        },
        "final_answer": {"type": "string"},
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": False,
}


async def solve_math_with_json_output(equation: str):
    """Solve a mathematical equation with structured JSON output."""

    llm = create_llm(Provider.OPENAI, "gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": "You are a mathematical assistant. Solve the given equation step by step.",
        },
        {"role": "user", "content": f"Solve this equation step by step: {equation}"},
    ]

    # Set up parameters with JSON schema response format
    params = ChatParams(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_solution",
                "schema": MATH_RESPONSE_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.1,  # Lower temperature for more consistent output
        max_tokens=1000,
    )

    logger.info(f"Solving '{equation}' with structured JSON output")

    try:
        response = await llm.chat(messages, params=params)

        # Parse the JSON response
        content = response.get_response_content()
        parsed_response = json.loads(content)

        # Create Pydantic model for validation
        math_response = MathResponse(**parsed_response)

        print(f"\nSolution for: {equation}")
        print("Steps:")
        for i, step in enumerate(math_response.steps, 1):
            print(f"  {i}. {step.explanation}")
            print(f"     Result: {step.output}")

        print(f"\nFinal Answer: {math_response.final_answer}")

        return math_response

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {response.get_response_content()}")
        return None
    except Exception as e:
        logger.error(f"Error during solution: {e}")
        return None


async def main():
    # Example 1: Linear equation
    await solve_math_with_json_output("3x + 7 = 16")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Quadratic equation
    await solve_math_with_json_output("x^2 - 5x + 6 = 0")


if __name__ == "__main__":
    asyncio.run(main())
