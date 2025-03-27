"""
test_api_calls.py

This module contains integration tests for the OpenAIAgent class, specifically testing
the API call functionality. The tests ensure that the response from the OpenAI API (simulated
via a mock) adheres to the expected structure and contains the necessary attributes.

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
    """Return a mock Response object mimicking openai.responses.create."""
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

async def test_openai_integration():
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
