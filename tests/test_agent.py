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
    
    :param content_dict: A dictionary representing the expected JSON structure.
    :return: A MagicMock object with the expected nested attributes.
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
    Fixture that creates and returns an instance of OpenAIAgent.
    
    This allows tests to use the same setup and ensures that if the constructor changes,
    you only need to update it in one place.
    """
    return OpenAIAgent()

async def test_successful_call(agent):
    """
    Test that a successful API call returns the expected structured dictionary.
    
    This test patches the 'create' method on the agent's client (agent.client.responses.create)
    using an AsyncMock, simulating a valid response from the OpenAI API.
    It then checks that the response contains all the expected keys and correct values.
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
    Test that the agent correctly retries after a RateLimitError.
    
    This simulates a RateLimitError on the first call and then a successful response on retry.
    It asserts that the final response is as expected.
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
    Test that the agent retries when an APIError occurs.
    
    Similar to the rate limit test, this simulates an APIError followed by a successful response.
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
    Test that the agent raises an exception after exceeding the maximum number of retries.
    
    This test simulates repeated RateLimitErrors so that the retry loop exceeds its limit,
    and then checks that the correct exception is raised.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
        with pytest.raises(Exception, match="Max retry attempts exceeded"):
            await agent.call_model("Test prompt")

async def test_empty_output(agent):
    """
    Test that the agent raises a ValueError when the API response has an empty 'output' list.
    
    This helps verify that our error handling for missing data (i.e., no output) works correctly.
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
    Test that the agent raises a ValueError when the 'content' list is empty.
    
    This simulates the scenario where the API returns an output structure without any content.
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
    Test that the agent raises a ValueError when the returned text is empty (only whitespace).
    
    This verifies that our code properly handles cases where the API returns no usable text.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "   "  # Only whitespace
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        with pytest.raises(ValueError, match="Received empty response from OpenAI API."):
            await agent.call_model("Test prompt")

async def test_malformed_json_response(agent):
    """
    Test that the agent raises a JSONDecodeError when the API response contains invalid JSON.
    
    This ensures that our JSON parsing logic correctly detects and raises errors for malformed data.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Not a JSON string"
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        with pytest.raises(json.JSONDecodeError):
            await agent.call_model("Test prompt")

async def test_retry_log_message(agent, capsys):
    """
    Test that the agent logs a retry warning when an APIError occurs.
    
    Since our logger's output is captured in stdout, this test uses the 'capsys' fixture to capture stdout,
    and asserts that the expected retry message (containing "Retrying after") appears in the captured output.
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
