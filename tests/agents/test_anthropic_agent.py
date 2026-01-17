"""
tests/agents/test_anthropic_agent.py

Contains unit tests for the `AnthropicAgent` class (`src.agents.anthropic_agent.py`).

These tests mirror the OpenAI agent tests but handle Anthropic API differences:
- Messages API instead of Responses API
- Different response structure (response.content[0].text)
- Token usage (input_tokens + output_tokens)
- JSON mode via prompt engineering

Tests verify:
    - Successful processing of valid API responses
    - Correct retry logic for RateLimitError and APIError
    - Correct exception handling for malformed/missing response data
    - Correct logging of retry attempts
    - Initialization error handling
    - Token usage tracking and metrics
    - Edge cases and error scenarios

Key Testing Techniques:
    - Asynchronous testing with `pytest-asyncio`
    - Fixtures for creating test instances
    - Patching async methods (`agent.client.messages.create`)
    - Testing expected exceptions using `pytest.raises`
    - Capturing log output using `caplog`
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIError, RateLimitError

from src.agents.anthropic_agent import AnthropicAgent


def mock_anthropic_response(content_dict, usage_data=None):
    """
    Creates a mock Anthropic Messages API response.

    Structure: response.content[0].text (less nesting than OpenAI)

    Args:
        content_dict (dict): The dictionary to serialize into the mock response text.
        usage_data (dict, optional): Usage data (input_tokens, output_tokens).

    Returns:
        MagicMock: A mock object ready for use with `patch`.
    """
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_resp.content = [mock_content]

    if usage_data:
        mock_usage = MagicMock()
        for key, value in usage_data.items():
            setattr(mock_usage, key, value)
        mock_resp.usage = mock_usage
    else:
        mock_resp.usage = None

    return mock_resp


@pytest.fixture
def mock_config():
    """Provides a mock configuration for testing."""
    return {
        "anthropic": {
            "api_key": "sk-ant-test-key-1234567890abcdef",
            "model_name": "claude-3-haiku-20240307",
            "max_tokens": 256,
            "temperature": 0.2,
        },
        "anthropic_api": {
            "retry": {
                "max_attempts": 3,
                "backoff_factor": 2
            }
        },
    }


@pytest.fixture
def agent(mock_config):
    """Provides a fresh AnthropicAgent instance for each test."""
    with patch("src.agents.anthropic_agent.config", mock_config):
        return AnthropicAgent()


# Test initialization and configuration

def test_initialization_with_valid_config(mock_config):
    """Test agent initializes correctly with valid config."""
    with patch("src.agents.anthropic_agent.config", mock_config):
        agent = AnthropicAgent()

        assert agent.model == "claude-3-haiku-20240307"
        assert agent.max_tokens == 256
        assert agent.temperature == 0.2
        assert agent.retry_attempts == 3
        assert agent.backoff_factor == 2


def test_initialization_without_api_key():
    """Test agent raises ValueError if API key is missing."""
    with patch("src.agents.anthropic_agent.config", {"anthropic": {"api_key": ""}}):
        with pytest.raises(ValueError, match="Anthropic API key is not set"):
            AnthropicAgent()


def test_initialization_with_none_api_key():
    """Test agent raises ValueError if API key is None."""
    with patch("src.agents.anthropic_agent.config", {"anthropic": {"api_key": None}}):
        with pytest.raises(ValueError, match="Anthropic API key is not set"):
            AnthropicAgent()


# Test provider name methods

def test_get_provider_name(agent):
    """Test get_provider_name returns 'anthropic'."""
    assert agent.get_provider_name() == "anthropic"


def test_get_model_name(agent):
    """Test get_model_name returns configured model."""
    assert agent.get_model_name() == "claude-3-haiku-20240307"


# Test Phase 2 capability flags

def test_supports_prompt_caching(agent):
    """Test that Anthropic agent supports prompt caching."""
    assert agent.supports_prompt_caching() is True


def test_supports_batch_api(agent):
    """Test that Anthropic agent supports batch API."""
    assert agent.supports_batch_api() is True


# Test successful API calls

@pytest.mark.asyncio
async def test_successful_call(agent):
    """Test successful API call with valid response."""
    response_content = {
        "function_type": "interrogative",
        "confidence": "high"
    }

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_anthropic_response(response_content)

        response = await agent.call_model("Test prompt")

        assert response == response_content
        mock_create.assert_called_once()

        # Verify API was called with correct structure
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "claude-3-haiku-20240307"
        assert "messages" in call_args[1]
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "Test prompt"
        assert "system" in call_args[1]
        assert "JSON" in call_args[1]["system"]


@pytest.mark.asyncio
async def test_successful_call_with_token_usage(agent):
    """Test token usage tracking with Anthropic's format (input + output)."""
    response_content = {"function_type": "declarative"}
    usage_data = {"input_tokens": 100, "output_tokens": 50}

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.anthropic_agent.metrics_tracker") as mock_metrics:

        mock_create.return_value = mock_anthropic_response(
            response_content, usage_data
        )

        response = await agent.call_model("Test prompt")

        assert response == response_content

        # Verify total tokens = input + output (150)
        mock_metrics.add_tokens.assert_called_once_with(150)
        mock_metrics.increment_api_calls.assert_called_once()


# Test error handling

@pytest.mark.asyncio
async def test_json_decode_error_returns_empty_dict(agent):
    """Test that JSON decode errors return {} gracefully."""
    # Mock response with invalid JSON
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"function_type": "invalid json'  # Malformed
    mock_resp.content = [mock_content]

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.anthropic_agent.metrics_tracker") as mock_metrics:

        mock_create.return_value = mock_resp

        response = await agent.call_model("Test prompt")

        assert response == {}
        mock_metrics.increment_errors.assert_called_once()


@pytest.mark.asyncio
async def test_empty_response_content_raises_value_error(agent):
    """Test that empty content list raises ValueError."""
    mock_resp = MagicMock()
    mock_resp.content = []  # Empty content

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_resp

        with pytest.raises(ValueError, match="No content received from Anthropic API"):
            await agent.call_model("Test prompt")


@pytest.mark.asyncio
async def test_empty_text_raises_value_error(agent):
    """Test that empty text content raises ValueError."""
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = ""  # Empty string
    mock_resp.content = [mock_content]

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_resp

        with pytest.raises(ValueError, match="Received empty response from Anthropic API"):
            await agent.call_model("Test prompt")


# Test retry logic

@pytest.mark.asyncio
async def test_retry_on_rate_limit_error(agent):
    """Test retry logic with exponential backoff for RateLimitError."""
    response_content = {"function_type": "declarative"}
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # RateLimitError requires response= as keyword-only argument
            mock_response = MagicMock()
            raise RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        else:
            return mock_anthropic_response(response_content)

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = side_effect

        response = await agent.call_model("Test prompt")

        assert response == response_content
        assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_api_error(agent):
    """Test retry logic for generic APIError."""
    response_content = {"function_type": "declarative"}
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # APIError requires request as positional argument
            mock_request = MagicMock()
            raise APIError("API error", mock_request, body=None)
        else:
            return mock_anthropic_response(response_content)

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = side_effect

        response = await agent.call_model("Test prompt")

        assert response == response_content
        assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_max_retry_exceeded_raises_error(agent):
    """Test that max retries exhausted raises the original error."""
    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.anthropic_agent.metrics_tracker") as mock_metrics:

        # APIError requires request as positional argument
        mock_request = MagicMock()
        mock_create.side_effect = APIError("Persistent error", mock_request, body=None)

        with pytest.raises(APIError):
            await agent.call_model("Test prompt")

        assert mock_create.call_count == 3  # max_attempts
        mock_metrics.increment_errors.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_unexpected_exception(agent):
    """Test retry logic for unexpected exceptions."""
    response_content = {"function_type": "declarative"}
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Unexpected error")
        else:
            return mock_anthropic_response(response_content)

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = side_effect

        response = await agent.call_model("Test prompt")

        assert response == response_content
        assert mock_create.call_count == 2


# Test logging

@pytest.mark.asyncio
async def test_retry_logging(agent, caplog):
    """Test that retry attempts are logged."""
    caplog.set_level(logging.INFO)

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # APIError requires request as positional argument
            mock_request = MagicMock()
            raise APIError("Temporary error", mock_request, body=None)
        else:
            return mock_anthropic_response({"function_type": "declarative"})

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = side_effect

        await agent.call_model("Test prompt")

        assert "Retrying after" in caplog.text


# Test edge cases

@pytest.mark.asyncio
async def test_no_usage_data_logs_debug(agent, caplog):
    """Test that missing usage data logs debug message but doesn't fail."""
    caplog.set_level(logging.DEBUG)
    response_content = {"function_type": "declarative"}

    # Mock response without usage data
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_content)
    mock_resp.content = [mock_content]
    mock_resp.usage = None

    with patch.object(
        agent.client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_resp

        response = await agent.call_model("Test prompt")

        assert response == response_content
        assert "Token usage data not available" in caplog.text
