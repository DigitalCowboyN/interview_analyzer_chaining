"""
test_api_calls.py

Multi-provider integration tests for agent call_model methods.

These tests verify that both OpenAI and Anthropic agents can process and parse
responses correctly. Tests use mocking to simulate API responses, isolating the
call_model logic without making actual external API calls.

Usage Example:
    Run the tests using pytest:
        pytest tests/integration/test_api_calls.py
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent_factory import AgentFactory

pytestmark = pytest.mark.asyncio


def mock_openai_response(content_dict):
    """
    Helper function to create a mock OpenAI Response object.

    Constructs a MagicMock with the structure expected from OpenAI's
    Responses API: response.output[0].content[0].text

    Args:
        content_dict (dict): Dictionary to be JSON-serialized as response text.

    Returns:
        MagicMock: Mock response for OpenAI client.responses.create
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp


def mock_anthropic_response(content_dict):
    """
    Helper function to create a mock Anthropic Messages API response.

    Constructs a MagicMock with the structure expected from Anthropic's
    Messages API: response.content[0].text

    Args:
        content_dict (dict): Dictionary to be JSON-serialized as response text.

    Returns:
        MagicMock: Mock response for Anthropic client.messages.create
    """
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_resp.content = [mock_content]
    return mock_resp


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_agent_integration(provider):
    """
    Multi-provider test for agent.call_model handling successful mocked responses.

    Tests both OpenAI and Anthropic agents using parameterization. Each provider
    uses its appropriate mock response structure and API endpoint.

    Args:
        provider (str): Provider name ("openai" or "anthropic")

    Raises:
        AssertionError: If response doesn't match expected content.
    """
    # Reset factory to ensure clean state
    AgentFactory.reset()
    agent = AgentFactory.create_agent(provider)

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "integration testing",
        "topic_level_3": "API evaluation",
        "keywords": ["integration", "API"],
        "domain_keywords": ["integration"],
    }

    # Patch appropriate API endpoint based on provider
    if provider == "openai":
        mock_target = agent.client.responses
        mock_response_fn = mock_openai_response
    else:  # anthropic
        mock_target = agent.client.messages
        mock_response_fn = mock_anthropic_response

    with patch.object(mock_target, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response_fn(response_content)
        response = await agent.call_model("Integration test prompt")

    # Verify response structure and values (same for both providers)
    assert isinstance(response, dict)
    assert response.get("function_type") == "declarative"
    assert response.get("structure_type") == "simple sentence"
    assert response.get("purpose") == "informational"
    assert response.get("topic_level_1") == "integration testing"
    assert response.get("topic_level_3") == "API evaluation"
    assert response.get("keywords") == ["integration", "API"]
    assert response.get("domain_keywords") == ["integration"]


async def test_openai_integration():
    """
    Legacy test maintained for backward compatibility.

    Use test_agent_integration with provider="openai" instead.
    This test is kept to avoid breaking existing test suites.
    """
    await test_agent_integration("openai")


async def test_anthropic_integration():
    """
    Anthropic-specific integration test.

    Tests that Anthropic agent correctly processes mocked responses
    with the Messages API structure.
    """
    await test_agent_integration("anthropic")
