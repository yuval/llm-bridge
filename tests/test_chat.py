"""Comprehensive test suite for parameter normalization and adapters."""

import pytest

from llm_bridge.params import normalize_params
from llm_bridge.adapters.openai import OpenAIRequestAdapter


class TestParamsNormalization:
    """Test parameter normalization functionality."""

    def test_basic_params_normalization(self):
        """Test basic parameter normalization with core parameters."""
        params = normalize_params(
            {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "stream": True,
            }
        )

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.2
        assert params["stream"] is True

    def test_extra_params_handling(self):
        """Test handling of extra parameters."""
        params = normalize_params(
            {"temperature": 0.7, "reasoning_effort": "minimal", "verbosity": "low"}
        )

        assert params["temperature"] == 0.7
        assert params["extra"]["reasoning_effort"] == "minimal"
        assert params["extra"]["verbosity"] == "low"

        # Test with different values
        params = normalize_params({"reasoning_effort": "high", "verbosity": "verbose"})

        assert params["extra"]["reasoning_effort"] == "high"
        assert params["extra"]["verbosity"] == "verbose"

    def test_none_values_handling(self):
        """Test handling of None values in normalization."""
        params = normalize_params(
            {"temperature": 0.7, "max_tokens": None, "reasoning_effort": "minimal"}
        )

        assert params["temperature"] == 0.7
        assert params["max_tokens"] is None
        assert params["extra"]["reasoning_effort"] == "minimal"

        # Test with empty params
        params = normalize_params(None)
        assert params["stream"] is False
        assert params["extra"] == {}

    def test_existing_extra_dict_merge(self):
        """Test merging with existing extra dict."""
        params = normalize_params(
            {
                "temperature": 0.7,
                "reasoning_effort": "minimal",
                "extra": {"verbosity": "high", "custom": "value"},
            }
        )

        # Check that both moved and explicit extra params are present
        assert params["temperature"] == 0.7
        assert params["extra"]["reasoning_effort"] == "minimal"
        assert params["extra"]["verbosity"] == "high"
        assert params["extra"]["custom"] == "value"

    def test_stream_default(self):
        """Test stream defaults to False."""
        params = normalize_params({"temperature": 0.7})
        assert params["stream"] is False

        # Explicit stream should override
        params = normalize_params({"stream": True})
        assert params["stream"] is True

    def test_tool_parameters(self):
        """Test tool-related parameters."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        tool_choice = {"type": "function", "function": {"name": "test"}}

        params = normalize_params(
            {"tools": tools, "tool_choice": tool_choice, "parallel_tool_calls": True}
        )

        assert params["tools"] == tools
        assert params["tool_choice"] == tool_choice
        assert params["parallel_tool_calls"] is True

    def test_response_format_parameter(self):
        """Test response_format parameter for structured outputs."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test_schema", "schema": {"type": "object"}},
        }

        params = normalize_params({"response_format": response_format})
        assert params["response_format"] == response_format

    def test_empty_normalization(self):
        """Test normalization with empty/None input."""
        params = normalize_params({})

        assert params["stream"] is False
        assert params["extra"] == {}
        assert "temperature" not in params
        assert "max_tokens" not in params

        params = normalize_params(None)
        assert params["stream"] is False
        assert params["extra"] == {}

    def test_edge_case_values(self):
        """Test edge cases and boundary values."""
        # Test with zero values
        params = normalize_params(
            {
                "temperature": 0.0,
                "max_tokens": 0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        )

        assert params["temperature"] == 0.0
        assert params["max_tokens"] == 0
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0

        # Test with empty collections
        params = normalize_params({"tools": [], "stop": []})

        assert params["tools"] == []
        assert params["stop"] == []


class TestOpenAIRequestAdapter:
    """Test OpenAI request adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create OpenAIRequestAdapter instance for testing."""
        return OpenAIRequestAdapter()

    def test_to_provider_basic_functionality(self, adapter):
        """Test basic to_provider functionality."""
        messages = [{"role": "user", "content": "Hello"}]
        params = normalize_params({"temperature": 0.7, "max_tokens": 100})

        result = adapter.to_provider(messages, params)

        assert "messages" in result
        assert "temperature" in result
        assert "max_tokens" in result
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert "stream" not in result

    def test_to_provider_message_conversion(self, adapter):
        """Test message format conversion."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        params = normalize_params({})

        result = adapter.to_provider(messages, params)

        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_to_provider_tool_calls(self, adapter):
        """Test tool call message handling."""
        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": {"a": 2, "b": 2}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ]
        params = normalize_params({})

        result = adapter.to_provider(messages, params)

        assert len(result["messages"]) == 3
        assert result["messages"][1]["tool_calls"] is not None
        assert result["messages"][2]["tool_call_id"] == "call_1"

    def test_stream_parameter_exclusion(self, adapter):
        """Test that stream parameter is excluded from API parameters."""
        messages = [{"role": "user", "content": "test"}]
        params = normalize_params({"stream": True, "temperature": 0.7})

        result = adapter.to_provider(messages, params)

        assert "stream" not in result
        assert result["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
