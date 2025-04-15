"""
test_api_calls.py

This module contains unit tests for the `OpenAIAgent.call_model` method,
focusing on its ability to process and parse responses from the OpenAI API client.

These tests utilize mocking (`unittest.mock.patch`) to simulate responses
from the `agent.client.responses.create` method, thereby isolating the
`call_model` logic without making actual external API calls.

Usage Example:
    Run the tests using pytest:
        pytest tests/integration/test_api_calls.py
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json
from src.agents.agent import OpenAIAgent

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """
    Helper function to create a mock Response object mimicking `openai.responses.create`.

    Constructs a `MagicMock` object with the nested structure expected from the
    OpenAI client's response (response -> output -> content -> text).

    Args:
        content_dict (dict): A dictionary to be JSON-serialized and set as the
                             `text` attribute of the innermost mock content.

    Returns:
        MagicMock: A mock response object suitable for patching `client.responses.create`.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

async def test_openai_integration():
    """
    Unit test for `OpenAIAgent.call_model` handling a successful mocked response.

    Instantiates `OpenAIAgent`, patches the underlying `client.responses.create`
    method to return a predefined mock response, and calls `agent.call_model`.

    Asserts that the dictionary returned by `call_model` contains the expected
    structure and values extracted from the mocked response content.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the response dictionary does not match the expected content.
    """
    agent = OpenAIAgent()
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "integration testing",
        "topic_level_3": "API evaluation",
        "overall_keywords": ["integration", "API"],
        "domain_keywords": ["integration"]
    }
    # Patch the create method on the agent's client using AsyncMock.
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response(response_content)
        response = await agent.call_model("Integration test prompt")
    
    # Verify the response structure and values.
    assert isinstance(response, dict)
    assert response.get("function_type") == "declarative"
    assert response.get("structure_type") == "simple sentence"
    assert response.get("purpose") == "informational"
    assert response.get("topic_level_1") == "integration testing"
    assert response.get("topic_level_3") == "API evaluation"
    assert response.get("overall_keywords") == ["integration", "API"]
    assert response.get("domain_keywords") == ["integration"]
