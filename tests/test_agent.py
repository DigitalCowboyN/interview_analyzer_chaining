"""
tests/test_agent.py

This file contains unit tests for the OpenAIAgent class found in src/agents/agent.py.
The OpenAIAgent class uses OpenAI's Responses API to analyze sentences and return structured JSON.
These tests verify that:
    - The agent successfully processes a correct API response.
    - The retry logic works for RateLimitError and APIError.
    - The agent correctly raises exceptions for malformed or missing data.
    - The logging of retry messages occurs when an error is encountered.

Key concepts covered in these tests:
    - Asynchronous testing with pytest (using pytest.mark.asyncio)
    - Using fixtures to create test instances of our agent.
    - Patching asynchronous API calls with AsyncMock.
    - Testing for expected exceptions with pytest.raises.
    - Capturing and asserting logging (or stdout output) with capsys.

How to run these tests:
    From the command line in the project root, run:
        pytest tests/test_agent.py

How to modify these tests:
    - If the API response format changes, update the expected keys and values in test_successful_call.
    - Adjust the simulated errors (e.g. RateLimitError, APIError) if you change the retry logic.
    - Add new tests if you introduce more edge cases or functionality in agent.py.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent

# This marks all tests in this file as asynchronous tests.
pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """
    Creates a mock response object that mimics the structure returned by the OpenAI API.
    
    The response structure is:
        - response.output: a list containing one item
            - The item has a 'content' attribute, which is a list containing one object.
            - The object has a 'text' attribute, containing a JSON string.
    
    Args:
        content_dict (dict): A dictionary representing the expected JSON structure,
                             which will be JSON-serialized.
    
    Returns:
        MagicMock: A MagicMock object configured with the nested attributes and
                   serialized content, suitable for mocking `client.responses.create`.
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
    Pytest fixture that creates and returns an instance of OpenAIAgent.
    
    Ensures a fresh agent instance is available for each test function.

    Returns:
        OpenAIAgent: A new instance of the agent class.
    """
    return OpenAIAgent()

async def test_successful_call(agent):
    """
    Test `call_model` successfully processes a valid mocked API response.
    
    Patches `agent.client.responses.create` to return a predefined successful
    response via `mock_response`. Asserts that the dictionary returned by
    `call_model` contains all expected keys and values from the mock content.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If the returned dictionary is not a dict, misses keys,
                      or has incorrect values.
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
    Test `call_model` retries successfully after a simulated RateLimitError.
    
    Mocks `agent.client.responses.create` to first raise `RateLimitError` and then
    return a successful response. Asserts that the final dictionary returned by
    `call_model` matches the content of the successful response.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If the final response does not match the expected successful content.
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
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # First call raises RateLimitError, second call returns a valid response.
        mock_create.side_effect = [error_response, mock_response(response_content)]
        response = await agent.call_model("Test prompt")
        assert response["function_type"] == "declarative"

async def test_retry_on_api_error(agent):
    """
    Test `call_model` retries successfully after a simulated generic APIError.
    
    Mocks `agent.client.responses.create` to first raise `APIError` and then
    return a successful response. Asserts that the final dictionary returned by
    `call_model` matches the content of the successful response.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If the final response does not match the expected successful content.
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
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [error_response, mock_response(response_content)]
        response = await agent.call_model("Test prompt")
        assert response["purpose"] == "to state a fact"

async def test_max_retry_exceeded(agent):
    """
    Test `call_model` raises the original APIError after exhausting retries.
    
    Mocks `agent.client.responses.create` to *always* raise a `RateLimitError`.
    Asserts that `call_model` raises an `APIError` (or subclass like `RateLimitError`)
    after making the configured number of retry attempts (`agent.retry_attempts`).

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If `APIError` is not raised, or if the mock was not called
                      the expected number of times.
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
    Test `call_model` raises ValueError for an API response with empty `output` list.
    
    Mocks `agent.client.responses.create` to return a response where `response.output`
    is an empty list. Asserts that `call_model` raises a `ValueError` with a specific
    message indicating no output was received.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If `ValueError` is not raised or the message doesn't match.
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
    Test `call_model` raises ValueError for an API response with empty `content` list.
    
    Mocks `agent.client.responses.create` to return a response where 
    `response.output[0].content` is an empty list. Asserts that `call_model` raises
    a `ValueError` with a specific message indicating no content was received.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If `ValueError` is not raised or the message doesn't match.
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
    Test `call_model` raises ValueError for an API response with empty text content.
    
    Mocks `agent.client.responses.create` to return a response where 
    `response.output[0].content[0].text` is empty or whitespace. Asserts that
    `call_model` raises a `ValueError` with a specific message indicating empty content.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If `ValueError` is not raised or the message doesn't match.
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
    Test `call_model` returns an empty dict for invalid JSON response content.
    
    Mocks `agent.client.responses.create` to return text that is not valid JSON.
    Asserts that `call_model` catches the `JSONDecodeError`, logs it (implicitly),
    and returns an empty dictionary `{}` instead of raising the error.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.

    Returns:
        None

    Raises:
        AssertionError: If the return value is not an empty dictionary.
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

async def test_retry_log_message(agent, capsys):
    """
    Test that `call_model` logs a retry message via stdout when retrying.
    
    Mocks `agent.client.responses.create` to raise an `APIError` then succeed.
    Uses the `capsys` fixture to capture standard output (where the configured
    logger directs human-readable output).

    Asserts that the captured output contains the expected "Retrying after" message.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.
        capsys: Pytest fixture for capturing stdout/stderr.

    Returns:
        None

    Raises:
        AssertionError: If the expected log message is not found in stdout.
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
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [error_response, mock_response(response_content)]
        response = await agent.call_model("Test prompt")
        captured = capsys.readouterr().out
        # Assert that the retry warning message appears in the captured stdout.
        assert "Retrying after" in captured, f"Expected retry log message not found in stdout. Captured: {captured}"
    assert response["purpose"] == "to state a fact"
