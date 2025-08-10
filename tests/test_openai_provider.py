"""Tests for OpenAI provider."""

from llm_bridge.adapters.openai import OpenAIRequestAdapter
from llm_bridge.params import normalize_params


def test_to_provider_basic():
    """Test basic to_provider functionality."""
    adapter = OpenAIRequestAdapter()
    messages = [{"role": "user", "content": "Hello"}]
    params = normalize_params({"temperature": 0.7, "max_tokens": 100})

    result = adapter.to_provider(messages, params)

    assert "messages" in result
    assert "temperature" in result
    assert "max_tokens" in result
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 100
    print("✅ Basic to_provider test passed")


def test_message_conversion():
    """Test message format conversion."""
    adapter = OpenAIRequestAdapter()
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]
    params = normalize_params({})

    result = adapter.to_provider(messages, params)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    print("✅ Message conversion test passed")


def test_stream_exclusion():
    """Test that stream parameter is excluded."""
    adapter = OpenAIRequestAdapter()
    messages = [{"role": "user", "content": "test"}]
    params = normalize_params({"stream": True, "temperature": 0.7})

    result = adapter.to_provider(messages, params)

    assert "stream" not in result
    assert result["temperature"] == 0.7
    print("✅ Stream exclusion test passed")


if __name__ == "__main__":
    import sys

    test_names = [
        n for n, f in list(globals().items()) if n.startswith("test_") and callable(f)
    ]
    for name in sorted(test_names):
        print(f"Running {name}...")
        try:
            globals()[name]()
        except AssertionError as e:
            print(f"{name} FAILED: {e}")
            sys.exit(1)
    print("All tests passed.")
