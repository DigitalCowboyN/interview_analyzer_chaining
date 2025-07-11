"""
tests/test_agent.py

Contains unit tests for the `OpenAIAgent` class (`src.agents.agent.py`).

These tests verify:
    - Successful processing of valid API responses.
    - Correct retry logic for `RateLimitError` and generic `APIError`.
    - Correct exception handling for malformed/missing API response data.
    - Correct logging of retry attempts.
    - Initialization error handling.
    - Token usage tracking and metrics.
    - Edge cases and error scenarios.

Key Testing Techniques Used:
    - Asynchronous testing with `pytest-asyncio` (`@pytest.mark.asyncio`).
    - Fixtures (`agent`) for creating test instances.
    - Patching asynchronous methods (`agent.client.responses.create`) with `AsyncMock`.
    - Testing expected exceptions using `pytest.raises`.
    - Capturing log output using the `caplog` fixture.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError, RateLimitError

from src.agents.agent import OpenAIAgent

# This marks all tests in this file as asynchronous tests.
pytestmark = pytest.mark.asyncio


def mock_response(content_dict, usage_data=None):
    """
    Creates a mock `openai.responses.CreateResponse` object structure.

    Mimics the expected nested structure: `response.output[0].content[0].text`,
    setting the `text` attribute to the JSON-serialized `content_dict`.

    Args:
        content_dict (dict): The dictionary to serialize into the mock response text.
        usage_data (dict, optional): Usage data to include in the response.

    Returns:
        MagicMock: A mock object ready for use with `patch`.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    # Add usage data if provided
    if usage_data:
        mock_usage = MagicMock()
        for key, value in usage_data.items():
            setattr(mock_usage, key, value)
        mock_resp.usage = mock_usage
    else:
        mock_resp.usage = None

    return mock_resp


@pytest.fixture
def agent():
    """
    Provides a fresh instance of `OpenAIAgent` for each test function.
    """
    return OpenAIAgent()


# === Test Initialization ===


def test_initialization_missing_api_key():
    """
    Tests that `OpenAIAgent` raises `ValueError` when API key is missing from config.

    This test covers the missing line 70 in coverage.
    """
    with patch("src.agents.agent.config", {"openai": {"api_key": ""}}):
        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIAgent()


def test_initialization_none_api_key():
    """
    Tests that `OpenAIAgent` raises `ValueError` when API key is None.
    """
    with patch("src.agents.agent.config", {"openai": {"api_key": None}}):
        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIAgent()


def test_initialization_success():
    """
    Tests successful initialization of `OpenAIAgent` with valid config.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "openai_api": {"retry": {"max_attempts": 3, "backoff_factor": 2}},
    }

    with patch("src.agents.agent.config", mock_config):
        agent = OpenAIAgent()
        assert agent.model == "gpt-4"
        assert agent.max_tokens == 1000
        assert agent.temperature == 0.7
        assert agent.retry_attempts == 3
        assert agent.backoff_factor == 2


# === Test Successful API Calls ===


async def test_successful_call(agent):
    """
    Tests `call_model` with a valid, successful mock API response.

    Verifies that the returned dictionary correctly parses the JSON content
    from the mock response and contains the expected keys and values.
    """
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    # Patch the API call on the agent's client to use AsyncMock
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response(response_content)
        response = await agent.call_model("Test prompt")

        # Define the expected keys in the JSON response.
        expected_keys = [
            "function_type",
            "structure_type",
            "purpose",
            "topic_level_1",
            "topic_level_3",
            "overall_keywords",
            "domain_keywords",
        ]
        # Verify that the response is a dictionary with all expected keys and correct values.
        assert isinstance(response, dict)
        for key in expected_keys:
            assert key in response, f"Missing key: {key}"
        assert response["function_type"] == "declarative"
        assert response["structure_type"] == "simple sentence"
        assert response["purpose"] == "to state a fact"
        assert response["topic_level_1"] == "testing"
        assert response["topic_level_3"] == "evaluation"
        assert response["overall_keywords"] == ["test"]
        assert response["domain_keywords"] == ["assessment", "evaluation"]


async def test_successful_call_with_token_usage(agent):
    """
    Tests `call_model` with successful response including token usage data.

    Verifies that token usage is properly tracked when available.
    """
    response_content = {"function_type": "declarative"}
    usage_data = {"total_tokens": 150, "input_tokens": 100, "output_tokens": 50}

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_create.return_value = mock_response(response_content, usage_data)
        response = await agent.call_model("Test prompt")

        # Verify response
        assert response["function_type"] == "declarative"

        # Verify metrics tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_called_once_with(150)


async def test_successful_call_without_token_usage(agent):
    """
    Tests `call_model` with successful response but no token usage data.

    This test covers the missing line 176 in coverage (the else branch).
    """
    response_content = {"function_type": "declarative"}

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_create.return_value = mock_response(response_content)  # No usage data
        response = await agent.call_model("Test prompt")

        # Verify response
        assert response["function_type"] == "declarative"

        # Verify metrics tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_not_called()  # Should not be called without usage data


async def test_successful_call_with_incomplete_token_usage(agent):
    """
    Tests `call_model` with response that has usage object but no total_tokens.

    This covers the case where usage exists but total_tokens is None.
    """
    response_content = {"function_type": "declarative"}

    # Create a mock response with usage but without total_tokens
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_content)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    # Create usage object that doesn't have total_tokens attribute
    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50
    # Remove total_tokens attribute to simulate it not being present
    del mock_usage.total_tokens
    mock_resp.usage = mock_usage

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_create.return_value = mock_resp
        response = await agent.call_model("Test prompt")

        # Verify response
        assert response["function_type"] == "declarative"

        # Verify metrics tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_not_called()  # Should not be called without total_tokens


# === Test Retry Logic ===


async def test_retry_on_rate_limit(agent):
    """
    Tests `call_model` successfully retries after a `RateLimitError`.

    Mocks the API call to raise `RateLimitError` once, then return success.
    Asserts the final returned dictionary matches the successful response content.
    """
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    error_response = RateLimitError(
        "Rate limit exceeded", response=MagicMock(), body=None
    )
    success_response = mock_response(response_content)

    # Use an async function for side_effect
    call_count = 0

    async def side_effect_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error_response
        else:
            return success_response

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func
        response = await agent.call_model("Test prompt")
        assert response["function_type"] == "declarative"
        assert call_count == 2  # Ensure it was called exactly twice (initial + 1 retry)


async def test_retry_on_api_error(agent):
    """
    Tests `call_model` successfully retries after a generic `APIError`.

    Mocks the API call to raise `APIError` once, then return success.
    Asserts the final returned dictionary matches the successful response content.
    """
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    error_response = APIError("API error", request=MagicMock(), body=MagicMock())
    success_response = mock_response(response_content)

    # Use an async function for side_effect
    call_count = 0

    async def side_effect_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error_response
        else:
            return success_response

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func
        response = await agent.call_model("Test prompt")
        assert response["purpose"] == "to state a fact"
        assert call_count == 2  # Ensure it was called exactly twice (initial + 1 retry)


async def test_max_retry_exceeded(agent):
    """
    Tests `call_model` raises the original `APIError` after exhausting all retries.

    Mocks the API call to *always* raise `RateLimitError`. Asserts that
    `pytest.raises(APIError)` correctly catches the exception after the configured
    number of attempts and that the mock was called the expected number of times.
    """
    error_to_raise = RateLimitError(
        "Rate limit exceeded repeatedly", response=MagicMock(), body=None
    )
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        # Simulate the error occurring on every call
        mock_create.side_effect = error_to_raise
        # Expect the specific APIError (or subclass) to be raised after retries
        with pytest.raises(APIError):  # Changed from Exception and removed match
            await agent.call_model("Test prompt for max retries")
        # Verify it was called the expected number of times (initial + retries)
        assert mock_create.call_count == agent.retry_attempts


async def test_retry_on_unexpected_exception(agent):
    """
    Tests `call_model` retries on unexpected exceptions and eventually raises them.
    """
    error_to_raise = ValueError("Unexpected error")

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error_to_raise

        with pytest.raises(ValueError, match="Unexpected error"):
            await agent.call_model("Test prompt")

        # Verify it was called the expected number of times
        assert mock_create.call_count == agent.retry_attempts


# === Test Response Validation ===


async def test_empty_output(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with an empty `output` list.
    """
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        # Create a mock response with an empty 'output' list.
        mock_resp = MagicMock()
        mock_resp.output = []
        mock_create.return_value = mock_resp
        with pytest.raises(ValueError, match="No output received from OpenAI API."):
            await agent.call_model("Test prompt")


async def test_empty_content(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with empty `output[0].content`.
    """
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_output.content = []
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        with pytest.raises(
            ValueError, match="No content received from OpenAI API response."
        ):
            await agent.call_model("Test prompt")


async def test_empty_message(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with empty/whitespace text.
    """
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "   "  # Only whitespace
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        # Expect ValueError with the updated message from agent.py
        with pytest.raises(
            ValueError, match="Received empty response content from OpenAI API."
        ):
            await agent.call_model("Test prompt for empty message")


async def test_malformed_json_response(agent):
    """
    Tests `call_model` returns an empty dict `{}` for a non-JSON API response.

    Verifies that `JSONDecodeError` is caught internally and an empty dict is returned,
    allowing processing to potentially continue for other items.
    """
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Not a JSON string"  # Invalid JSON
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp

        # Expected behavior: returns {} and logs error, does not raise JSONDecodeError
        result = await agent.call_model("Test prompt for malformed JSON")
        assert result == {}

        # Verify error was tracked
        mock_metrics.increment_errors.assert_called_once()


# === Test Logging ===


async def test_retry_log_message(agent, caplog):
    """
    Tests `call_model` logs the correct retry message when retrying API calls.

    Mocks the API call to raise `RateLimitError` once, then succeed.
    Uses `caplog` to capture log records emitted by the agent and verify the
    "Retrying after..." message is present at INFO level.
    """
    # Explicitly set logger level for caplog to capture INFO messages
    caplog.set_level(logging.INFO)

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    error_response = RateLimitError(
        "Rate limit exceeded", response=MagicMock(), body=None
    )
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error_response, mock_response(response_content)]
        await agent.call_model("Test prompt")

    # Assert the message is present in the captured log records
    assert (
        "Retrying after" in caplog.text
    ), f"Expected retry log message not found in caplog.text. Captured: {caplog.text}"


# === Test Edge Cases ===


async def test_zero_retry_attempts(agent):
    """
    Tests `call_model` with zero retry attempts configured.

    This test covers the missing lines 235-244 in coverage (fallback error handling).
    """
    # Set retry attempts to 0 to trigger the fallback logic
    agent.retry_attempts = 0

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        # This should never be called since retry_attempts is 0
        mock_create.side_effect = RateLimitError(
            "Rate limit", response=MagicMock(), body=None
        )

        # Should raise the fallback exception
        with pytest.raises(
            Exception,
            match="call_model failed after retries without specific exception recorded",
        ):
            await agent.call_model("Test prompt")

        # Verify the call was never made due to retry_attempts being 0
        mock_create.assert_not_called()

        # Verify error was tracked
        mock_metrics.increment_errors.assert_called_once()


async def test_negative_retry_attempts(agent):
    """
    Tests `call_model` with negative retry attempts configured.

    This also covers the fallback error handling.
    """
    # Set retry attempts to negative value to trigger the fallback logic
    agent.retry_attempts = -1

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        # Should raise the fallback exception
        with pytest.raises(
            Exception,
            match="call_model failed after retries without specific exception recorded",
        ):
            await agent.call_model("Test prompt")

        # Verify the call was never made due to retry_attempts being negative
        mock_create.assert_not_called()

        # Verify error was tracked
        mock_metrics.increment_errors.assert_called_once()


async def test_fallback_with_last_exception(agent):
    """
    Tests the fallback logic when a last_exception exists but loop exits unexpectedly.
    """
    # Mock an agent with 1 retry attempt
    agent.retry_attempts = 1

    # Create a custom exception that will be stored as last_exception
    test_exception = ValueError("Test exception")

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:

        # Mock the scenario where an exception occurs and is stored as last_exception
        # but somehow the loop exits (this is a contrived scenario for coverage)
        mock_create.side_effect = test_exception

        # Should raise the original exception
        with pytest.raises(ValueError, match="Test exception"):
            await agent.call_model("Test prompt")

        # Verify the call was made once (initial attempt)
        assert mock_create.call_count == 1


# === Test Metrics Integration ===


async def test_metrics_tracking_on_api_error(agent):
    """
    Tests that metrics are properly tracked when API errors occur after retries.
    """
    error_to_raise = APIError(
        "Persistent API error", request=MagicMock(), body=MagicMock()
    )

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_create.side_effect = error_to_raise

        with pytest.raises(APIError):
            await agent.call_model("Test prompt")

        # Verify error was tracked after retries exhausted
        mock_metrics.increment_errors.assert_called_once()


async def test_metrics_tracking_on_unexpected_error(agent):
    """
    Tests that metrics are properly tracked when unexpected errors occur after retries.
    """
    error_to_raise = ValueError("Unexpected error")

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create, patch("src.agents.agent.metrics_tracker") as mock_metrics:

        mock_create.side_effect = error_to_raise

        with pytest.raises(ValueError):
            await agent.call_model("Test prompt")

        # Verify error was tracked after retries exhausted
        mock_metrics.increment_errors.assert_called_once()


# === Test Configuration Edge Cases ===


def test_initialization_with_missing_retry_config():
    """
    Tests initialization when retry configuration is missing from config.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        # Missing openai_api section
    }

    with patch("src.agents.agent.config", mock_config):
        agent = OpenAIAgent()
        # Should use default values
        assert agent.retry_attempts == 5  # Default
        assert agent.backoff_factor == 2  # Default


def test_initialization_with_partial_retry_config():
    """
    Tests initialization when retry configuration is partially present.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "openai_api": {
            "retry": {
                "max_attempts": 3
                # Missing backoff_factor
            }
        },
    }

    with patch("src.agents.agent.config", mock_config):
        agent = OpenAIAgent()
        assert agent.retry_attempts == 3  # From config
        assert agent.backoff_factor == 2  # Default


# === Test API Call Parameters ===


async def test_api_call_parameters(agent):
    """
    Tests that the API call is made with correct parameters.
    """
    response_content = {"function_type": "declarative"}

    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:

        mock_create.return_value = mock_response(response_content)
        await agent.call_model("Test prompt")

        # Verify the API call was made with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        assert call_args[1]["model"] == agent.model
        assert call_args[1]["input"] == "Test prompt"
        assert call_args[1]["max_output_tokens"] == agent.max_tokens
        assert call_args[1]["temperature"] == agent.temperature
        assert call_args[1]["text"] == {"format": {"type": "json_object"}}
        assert "You are a coding assistant" in call_args[1]["instructions"]
