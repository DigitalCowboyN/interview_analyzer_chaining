"""
test_openai_agent_response.py

This module contains unit tests for the OpenAIAgent class, specifically testing
the structure of the response returned by the OpenAI API. The tests ensure that
the response adheres to the expected format and contains the necessary attributes.

Usage Example:
    Run the tests using pytest:
        pytest tests/test_openai_agent_response.py
"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from src.agents.agent import OpenAIAgent

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """
    Return a mock Response object mimicking openai.responses.create.

    This function creates a mock response object that simulates the structure
    of the response returned by the OpenAI API, allowing for controlled testing
    of the OpenAIAgent's behavior.

    Parameters:
        content_dict (dict): A dictionary representing the content of the mock response.

    Returns:
        MagicMock: A mock response object with the specified content.
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
    Test the structure of the response returned by the OpenAIAgent's call_model method.

    This test verifies that the response (a dictionary) contains the expected attributes,
    such as "topic_level_1", and that its value matches the expected test value.

    Asserts:
        - The response is a dictionary.
        - The "topic_level_1" attribute in the response matches the expected value.
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
