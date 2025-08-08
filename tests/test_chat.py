"""Comprehensive test suite for chat.py focusing on ChatParams provider-specific parameters."""

import pytest
from typing import Any, Dict

from llm_bridge.types.chat import ChatParams
from llm_bridge.providers.openai import OpenAIRequestAdapter


class TestChatParams:
    """Test ChatParams dataclass functionality and provider-specific parameter handling."""

    def test_basic_params_initialization(self):
        """Test basic ChatParams initialization with core parameters."""
        params = ChatParams(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            stream=True
        )
        
        assert params.temperature == 0.7
        assert params.max_tokens == 100
        assert params.top_p == 0.9
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.2
        assert params.stream is True

    def test_reasoning_and_verbosity_params(self):
        """Test GPT-5 and o-series specific reasoning parameters."""
        params = ChatParams(
            reasoning_effort="minimal",
            verbosity="low"
        )
        
        assert params.reasoning_effort == "minimal"
        assert params.verbosity == "low"
        
        # Test with different values
        params = ChatParams(
            reasoning_effort="high",
            verbosity="verbose"
        )
        
        assert params.reasoning_effort == "high"
        assert params.verbosity == "verbose"

    def test_as_dict_method(self):
        """Test ChatParams.as_dict() method with exclude_none parameter."""
        params = ChatParams(
            temperature=0.7,
            max_tokens=None,
            reasoning_effort="minimal"
        )
        
        # Test excluding None values (default)
        result = params.as_dict()
        assert "temperature" in result
        assert "max_tokens" not in result
        assert "reasoning_effort" in result
        assert result["temperature"] == 0.7
        assert result["reasoning_effort"] == "minimal"
        
        # Test including None values
        result = params.as_dict(exclude_none=False)
        assert "temperature" in result
        assert "max_tokens" in result
        assert "reasoning_effort" in result
        assert result["max_tokens"] is None

    def test_copy_method(self):
        """Test ChatParams.copy() method with overrides."""
        original = ChatParams(
            temperature=0.7,
            max_tokens=100,
            reasoning_effort="minimal"
        )
        
        # Test copy without overrides
        copied = original.copy()
        assert copied.temperature == 0.7
        assert copied.max_tokens == 100
        assert copied.reasoning_effort == "minimal"
        assert copied is not original  # Different instance
        
        # Test copy with overrides
        modified = original.copy(temperature=0.9, reasoning_effort="high")
        assert modified.temperature == 0.9
        assert modified.max_tokens == 100  # Unchanged
        assert modified.reasoning_effort == "high"
        assert original.temperature == 0.7  # Original unchanged

    def test_extra_params_functionality(self):
        """Test extra_params for provider-specific parameters."""
        extra_params = {
            "custom_param": "value",
            "provider_specific": True,
            "nested": {"key": "value"}
        }
        
        params = ChatParams(
            temperature=0.7,
            extra_params=extra_params
        )
        
        assert params.extra_params == extra_params
        
        # Test in as_dict
        result = params.as_dict()
        assert result["extra_params"] == extra_params

    def test_tool_parameters(self):
        """Test tool-related parameters."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        tool_choice = {"type": "function", "function": {"name": "test"}}
        
        params = ChatParams(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=True
        )
        
        assert params.tools == tools
        assert params.tool_choice == tool_choice
        assert params.parallel_tool_calls is True

    def test_response_format_parameter(self):
        """Test response_format parameter for structured outputs."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object"}
            }
        }
        
        params = ChatParams(response_format=response_format)
        assert params.response_format == response_format

    def test_all_optional_params_none_by_default(self):
        """Test that all parameters are optional and default to None or appropriate defaults."""
        params = ChatParams()
        
        assert params.temperature is None
        assert params.max_tokens is None
        assert params.top_p is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.stream is False  # Default to False
        assert params.tools is None
        assert params.tool_choice is None
        assert params.response_format is None
        assert params.seed is None
        assert params.stop is None
        assert params.user is None
        assert params.parallel_tool_calls is None
        assert params.reasoning_effort is None
        assert params.verbosity is None
        assert params.extra_params is None

    def test_edge_case_values(self):
        """Test edge cases and boundary values."""
        # Test with zero values
        params = ChatParams(
            temperature=0.0,
            max_tokens=0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        result = params.as_dict()
        assert result["temperature"] == 0.0
        assert result["max_tokens"] == 0
        assert result["frequency_penalty"] == 0.0
        assert result["presence_penalty"] == 0.0
        
        # Test with empty collections
        params = ChatParams(
            tools=[],
            stop=[]
        )
        
        result = params.as_dict()
        assert result["tools"] == []
        assert result["stop"] == []


class TestOpenAIProviderSpecificParams:
    """Test OpenAI provider-specific parameter handling through OpenAIRequestAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create OpenAIRequestAdapter instance for testing."""
        return OpenAIRequestAdapter()

    def test_max_tokens_to_max_completion_tokens_conversion(self, adapter):
        """Test conversion of max_tokens to max_completion_tokens for newer models."""
        params = ChatParams(max_tokens=100)
        
        # Test with older model (should keep max_tokens)
        result = adapter.build_params(params, "gpt-4-turbo")
        assert "max_tokens" in result
        assert "max_completion_tokens" not in result
        assert result["max_tokens"] == 100
        
        # Test with GPT-5 (should convert to max_completion_tokens)
        result = adapter.build_params(params, "gpt-5")
        assert "max_tokens" not in result
        assert "max_completion_tokens" in result
        assert result["max_completion_tokens"] == 100
        
        # Test with o3-mini (should convert to max_completion_tokens)
        result = adapter.build_params(params, "o3-mini")
        assert "max_tokens" not in result
        assert "max_completion_tokens" in result
        assert result["max_completion_tokens"] == 100

    def test_requires_max_completion_tokens_detection(self, adapter):
        """Test the model detection logic for max_completion_tokens requirement."""
        # Older models should return False
        assert not adapter._requires_max_completion_tokens("gpt-4")
        assert not adapter._requires_max_completion_tokens("gpt-4-turbo")
        assert not adapter._requires_max_completion_tokens("gpt-3.5-turbo")
        assert not adapter._requires_max_completion_tokens("gpt-4o")
        assert not adapter._requires_max_completion_tokens("gpt-4o-mini")
        
        # GPT-5 series should return True
        assert adapter._requires_max_completion_tokens("gpt-5")
        assert adapter._requires_max_completion_tokens("gpt-5-turbo")
        assert adapter._requires_max_completion_tokens("gpt-5-mini")
        
        # O1 series should return True
        assert adapter._requires_max_completion_tokens("o1")
        assert adapter._requires_max_completion_tokens("o1-preview")
        assert adapter._requires_max_completion_tokens("o1-mini")
        
        # O3 series should return True
        assert adapter._requires_max_completion_tokens("o3")
        assert adapter._requires_max_completion_tokens("o3-mini")
        assert adapter._requires_max_completion_tokens("o3-turbo")

    def test_reasoning_effort_parameter_handling(self, adapter):
        """Test reasoning_effort parameter for GPT-5 and reasoning models."""
        # Test direct reasoning_effort parameter
        params = ChatParams(reasoning_effort="minimal")
        result = adapter.build_params(params, "gpt-5")
        assert result["reasoning_effort"] == "minimal"
        
        # Test different values
        params = ChatParams(reasoning_effort="high")
        result = adapter.build_params(params, "gpt-5")
        assert result["reasoning_effort"] == "high"
        
        # Test with o1 model
        params = ChatParams(reasoning_effort="minimal")
        result = adapter.build_params(params, "o1-preview")
        assert result["reasoning_effort"] == "minimal"

    def test_verbosity_parameter_handling(self, adapter):
        """Test verbosity parameter for supported models."""
        params = ChatParams(verbosity="low")
        result = adapter.build_params(params, "gpt-5")
        assert result["verbosity"] == "low"
        
        params = ChatParams(verbosity="verbose")
        result = adapter.build_params(params, "gpt-5")
        assert result["verbosity"] == "verbose"

    def test_responses_style_parameter_mapping(self, adapter):
        """Test mapping from responses-style extra_params to top-level parameters."""
        # Test reasoning effort mapping from extra_params
        params = ChatParams(
            extra_params={
                "reasoning": {"effort": "high"},
                "text": {"verbosity": "verbose"}
            }
        )
        result = adapter.build_params(params, "gpt-5")
        assert result["reasoning_effort"] == "high"
        assert result["verbosity"] == "verbose"
        
        # Test that top-level parameters take precedence over extra_params
        params = ChatParams(
            reasoning_effort="minimal",  # Top-level
            verbosity="low",  # Top-level
            extra_params={
                "reasoning": {"effort": "high"},  # Should be ignored
                "text": {"verbosity": "verbose"}  # Should be ignored
            }
        )
        result = adapter.build_params(params, "gpt-5")
        assert result["reasoning_effort"] == "minimal"
        assert result["verbosity"] == "low"

    def test_extra_params_inclusion(self, adapter):
        """Test that extra_params are included in the final parameters (except consumed ones)."""
        params = ChatParams(
            extra_params={
                "custom_param": "value",
                "another_param": 42,
                "reasoning": {"effort": "high"},  # Will be consumed
                "text": {"verbosity": "low"},  # Will be consumed
                "nested_config": {"key": "value"}  # Should remain
            }
        )
        result = adapter.build_params(params, "gpt-5")
        
        # Consumed params should be transformed to top-level parameters
        assert result["reasoning_effort"] == "high"
        assert result["verbosity"] == "low"
        
        # Other extra_params should be included
        assert result["custom_param"] == "value"
        assert result["another_param"] == 42
        assert result["nested_config"] == {"key": "value"}
        
        # Original extra_params structure should not be in result
        assert "extra_params" not in result
        assert "reasoning" not in result
        assert "text" not in result

    def test_stream_parameter_exclusion(self, adapter):
        """Test that stream parameter is excluded from API parameters."""
        params = ChatParams(stream=True, temperature=0.7)
        result = adapter.build_params(params, "gpt-4")
        
        assert "stream" not in result
        assert result["temperature"] == 0.7

    def test_combined_parameters_scenario(self, adapter):
        """Test a realistic scenario with multiple parameter types."""
        params = ChatParams(
            temperature=0.8,
            max_tokens=500,
            top_p=0.9,
            reasoning_effort="medium",
            verbosity="detailed",
            tools=[{"type": "function", "function": {"name": "calculator"}}],
            tool_choice="auto",
            response_format={"type": "json_object"},
            extra_params={
                "custom_setting": True,
                "provider_config": {"timeout": 30}
            }
        )
        
        # Test with GPT-5 (should convert max_tokens)
        result = adapter.build_params(params, "gpt-5")
        
        assert result["temperature"] == 0.8
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 500
        assert result["top_p"] == 0.9
        assert result["reasoning_effort"] == "medium"
        assert result["verbosity"] == "detailed"
        assert result["tools"] == [{"type": "function", "function": {"name": "calculator"}}]
        assert result["tool_choice"] == "auto"
        assert result["response_format"] == {"type": "json_object"}
        assert result["custom_setting"] is True
        assert result["provider_config"] == {"timeout": 30}
        
        # Ensure excluded parameters
        assert "stream" not in result
        assert "extra_params" not in result

    def test_none_values_exclusion(self, adapter):
        """Test that None values are properly excluded from API parameters."""
        params = ChatParams(
            temperature=0.7,
            max_tokens=None,
            reasoning_effort=None,
            verbosity="low"
        )
        
        result = adapter.build_params(params, "gpt-5")
        
        assert result["temperature"] == 0.7
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result
        assert "reasoning_effort" not in result
        assert result["verbosity"] == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])