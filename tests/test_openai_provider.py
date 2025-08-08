"""Tests for OpenAI provider."""

from llm_bridge.providers.openai import OpenAIRequestAdapter
from llm_bridge.types.chat import ChatParams


def test_max_tokens_parameter_conversion():
    """Test that max_tokens is correctly converted to max_completion_tokens for newer models."""
    adapter = OpenAIRequestAdapter()

    # Test with GPT-4 (should keep max_tokens)
    params = ChatParams(max_tokens=100)
    result = adapter.build_params(params, "gpt-4-turbo")
    assert "max_tokens" in result
    assert "max_completion_tokens" not in result
    assert result["max_tokens"] == 100
    print("✅ GPT-4 test passed")

    # Test with GPT-5 (should convert to max_completion_tokens)
    params = ChatParams(max_tokens=100)
    result = adapter.build_params(params, "gpt-5")
    assert "max_tokens" not in result
    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 100
    print("✅ GPT-5 test passed")


def test_requires_max_completion_tokens():
    """Test the model detection logic."""
    adapter = OpenAIRequestAdapter()

    # Older models should return False
    assert not adapter._requires_max_completion_tokens("gpt-4")
    assert not adapter._requires_max_completion_tokens("gpt-4-turbo")
    assert not adapter._requires_max_completion_tokens("gpt-3.5-turbo")
    print("✅ Older model detection tests passed")

    # Newer models should return True
    assert adapter._requires_max_completion_tokens("gpt-5")
    assert adapter._requires_max_completion_tokens("gpt-5-turbo")
    assert adapter._requires_max_completion_tokens("o1")
    assert adapter._requires_max_completion_tokens("o1-preview")
    assert adapter._requires_max_completion_tokens("o3-mini")
    print("✅ Newer model detection tests passed")


def test_reasoning_and_verbosity_top_level_and_responses_style():
    adapter = OpenAIRequestAdapter()
    # Top-level fields win
    params = ChatParams(reasoning_effort="minimal", verbosity="low")
    out = adapter.build_params(params, "gpt-5")
    assert out["reasoning_effort"] == "minimal"
    assert out["verbosity"] == "low"

    # Responses-style fallback
    params = ChatParams(extra_params={"reasoning": {"effort": "high"}, "text": {"verbosity": "high"}})
    out = adapter.build_params(params, "gpt-5")
    assert out["reasoning_effort"] == "high"
    assert out["verbosity"] == "high"


def test_extra_params_not_leaked():
    adapter = OpenAIRequestAdapter()
    params = ChatParams(extra_params={"foo": "bar"})
    out = adapter.build_params(params, "gpt-4o-mini")
    assert "extra_params" not in out
    assert out["foo"] == "bar"

if __name__ == "__main__":
    import sys
    test_names = [n for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name in sorted(test_names):
        print(f"Running {name}...")
        try:
            globals()[name]()
        except AssertionError as e:
            print(f"{name} FAILED: {e}")
            sys.exit(1)
    print("All tests passed.")