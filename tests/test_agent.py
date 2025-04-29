"""
tests/test_agent.py

Contains unit tests for the `OpenAIAgent` class (`src.agents.agent.py`).

These tests verify:
    - Successful processing of valid API responses.
    - Correct retry logic for `RateLimitError` and generic `APIError`.
    - Correct exception handling for malformed/missing API response data.
    - Correct logging of retry attempts.

Key Testing Techniques Used:
    - Asynchronous testing with `pytest-asyncio` (`@pytest.mark.asyncio`).
    - Fixtures (`agent`) for creating test instances.
    - Patching asynchronous methods (`agent.client.responses.create`) with `AsyncMock`.
    - Testing expected exceptions using `pytest.raises`.
    - Capturing log output using the `caplog` fixture.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent
import logging

# This marks all tests in this file as asynchronous tests.
pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """
    Creates a mock `openai.responses.CreateResponse` object structure.
    
    Mimics the expected nested structure: `response.output[0].content[0].text`,
    setting the `text` attribute to the JSON-serialized `content_dict`.
    
    Args:
        content_dict (dict): The dictionary to serialize into the mock response text.
    
    Returns:
        MagicMock: A mock object ready for use with `patch`.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

@pytest.fixture
def agent():
    """
    Provides a fresh instance of `OpenAIAgent` for each test function.
    """
    return OpenAIAgent()

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
        "domain_keywords": ["assessment", "evaluation"]
    }
    # Patch the API call on the agent's client to use AsyncMock
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
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
            "domain_keywords"
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
        "domain_keywords": ["assessment", "evaluation"]
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
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

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func
        response = await agent.call_model("Test prompt")
        assert response["function_type"] == "declarative"
        assert call_count == 2 # Ensure it was called exactly twice (initial + 1 retry)

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
        "domain_keywords": ["assessment", "evaluation"]
    }
    error_response = APIError("API error", request="mock_request", body="mock_body")
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

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func
        response = await agent.call_model("Test prompt")
        assert response["purpose"] == "to state a fact"
        assert call_count == 2 # Ensure it was called exactly twice (initial + 1 retry)

async def test_max_retry_exceeded(agent):
    """
    Tests `call_model` raises the original `APIError` after exhausting all retries.
    
    Mocks the API call to *always* raise `RateLimitError`. Asserts that
    `pytest.raises(APIError)` correctly catches the exception after the configured
    number of attempts and that the mock was called the expected number of times.
    """
    error_to_raise = RateLimitError("Rate limit exceeded repeatedly", response=MagicMock(), body=None)
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Simulate the error occurring on every call
        mock_create.side_effect = error_to_raise
        # Expect the specific APIError (or subclass) to be raised after retries
        with pytest.raises(APIError): # Changed from Exception and removed match
            await agent.call_model("Test prompt for max retries")
        # Verify it was called the expected number of times (initial + retries)
        assert mock_create.call_count == agent.retry_attempts

async def test_empty_output(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with an empty `output` list.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
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
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_output.content = []
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        with pytest.raises(ValueError, match="No content received from OpenAI API response."):
            await agent.call_model("Test prompt")

async def test_empty_message(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with empty/whitespace text.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "   "  # Only whitespace
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        # Expect ValueError with the updated message from agent.py
        with pytest.raises(ValueError, match="Received empty response content from OpenAI API."):
            await agent.call_model("Test prompt for empty message")

async def test_malformed_json_response(agent):
    """
    Tests `call_model` returns an empty dict `{}` for a non-JSON API response.
    
    Verifies that `JSONDecodeError` is caught internally and an empty dict is returned,
    allowing processing to potentially continue for other items.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Not a JSON string" # Invalid JSON
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        
        # Expected behavior: returns {} and logs error, does not raise JSONDecodeError
        result = await agent.call_model("Test prompt for malformed JSON")
        assert result == {} 

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
        "domain_keywords": ["assessment", "evaluation"]
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [error_response, mock_response(response_content)]
        await agent.call_model("Test prompt")
        
    # Assert the message is present in the captured log records
    assert "Retrying after" in caplog.text, f"Expected retry log message not found in caplog.text. Captured: {caplog.text}"
