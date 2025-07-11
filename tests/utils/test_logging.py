"""
test_logging.py

This module contains tests focused on verifying the logging output generated
by other parts of the application, specifically the retry mechanism within
the `OpenAIAgent`.

It utilizes the standard `caplog` fixture to capture log records
during test execution and assert that expected messages (e.g., retry warnings)
are logged under specific conditions (e.g., simulated API errors).
"""

# tests/utils/test_logging.py

import json
import logging  # Import logging for caplog level
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from src.agents.agent import OpenAIAgent

# Removed: from src.utils.logger import get_logger (not needed for test logic itself)

pytestmark = pytest.mark.asyncio


def mock_response(content_dict):
    """
    Helper function to create a mock Response object mimicking `openai.responses.create`.

    Constructs a `MagicMock` object with the nested structure expected from the
    OpenAI client's response, specifically setting the `text` attribute within
    `response.output[0].content[0].text`.

    Args:
        content_dict (dict): A dictionary to be JSON-serialized and set as the
                             `text` attribute.

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


@pytest.fixture
def agent():
    """
    Fixture to create an instance of OpenAIAgent for testing.
    """
    return OpenAIAgent()


# Removed the Loguru-specific log_sink fixture


async def test_retry_log_message(agent, caplog):  # Use caplog fixture
    """
    Test that the agent's retry logic logs a specific message upon API error.

    Mocks the underlying API call (`client.responses.create`) to first raise a
    `RateLimitError` and then return a successful mock response. Uses the `caplog`
    fixture to capture log output *emitted by the agent* during the
    `agent.call_model` execution.

    Asserts that the captured log text contains the expected
    "Retrying after" substring, confirming the retry mechanism logged correctly.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.
        caplog: Pytest fixture to capture log messages.

    Raises:
        AssertionError: If the expected retry log message is not found in the logs.
    """
    # Set caplog level to capture INFO messages (retry logs at INFO)
    caplog.set_level(logging.INFO)

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "retry test",
        "topic_level_3": "test",
        "overall_keywords": ["test"],
        "domain_keywords": ["test"],
    }
    error_response = RateLimitError(
        "Rate limit exceeded", response=MagicMock(), body=None
    )
    with patch.object(
        agent.client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        # First call raises an error, second call returns a valid response.
        mock_create.side_effect = [error_response, mock_response(response_content)]
        await agent.call_model("Test prompt")

    # Check that the captured log text contains "Retrying after"
    assert (
        "Retrying after" in caplog.text
    ), "Expected retry log message not found in logs"
