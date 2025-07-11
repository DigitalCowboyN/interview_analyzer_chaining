"""
test_openai_agent_response.py

This module contains basic unit tests for the `OpenAIAgent.call_model` method,
specifically verifying the structure and basic content of the dictionary returned
when processing a mocked successful API response.

These tests utilize mocking (`unittest.mock.patch`) to simulate responses
from the `agent.client.responses.create` method.

Note: More comprehensive tests for `OpenAIAgent`, including error handling and
      retry logic, can be found in `tests/test_agent.py`.

Usage Example:
    Run the tests using pytest:
        pytest tests/test_openai_agent_response.py
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent import OpenAIAgent

pytestmark = pytest.mark.asyncio


def mock_response(content_dict):
    """
    Helper function to create a mock Response object mimicking `openai.responses.create`.

    This function creates a mock response object that simulates the structure
    of the response returned by the OpenAI API, allowing for controlled testing
    of the OpenAIAgent's behavior.

    Args:
        content_dict (dict): A dictionary representing the content of the mock response,
                             which will be JSON-serialized.

    Returns:
        MagicMock: A mock response object with the specified content, suitable for patching
                   `client.responses.create`.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp


async def test_openai_response_structure():
    """
    Test `call_model` returns a dict with expected keys from a mocked response.

    Mocks `agent.client.responses.create` to return a predefined successful
    response. Calls `agent.call_model` and verifies that the result is a dictionary
    containing at least one expected key (`topic_level_1`) with the correct value.

    Args:
        None (implicitly uses `OpenAIAgent` initialized within the test).

    Returns:
        None

    Raises:
        AssertionError: If the response is not a dict or the expected key/value
                      is missing or incorrect.
    """
    agent = OpenAIAgent()
    prompt = "What is the capital of France?"
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "geography",
        "topic_level_3": "capitals",
        "overall_keywords": ["France", "Paris"],
        "domain_keywords": ["geography"]
    }
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response(response_content)
        response = await agent.call_model(prompt)
    # Now we expect the response to be a dictionary with the expected keys.
    assert isinstance(response, dict)
    assert response.get("topic_level_1") == "geography"
