"""
test_logging.py

This module contains tests focused on verifying the logging output generated
by other parts of the application, specifically the retry mechanism within
the `OpenAIAgent`.

It utilizes a custom Loguru sink fixture (`log_sink`) to capture log records
during test execution and assert that expected messages (e.g., retry warnings)
are logged under specific conditions (e.g., simulated API errors).
"""

# tests/utils/test_logging.py

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from openai import RateLimitError
from src.agents.agent import OpenAIAgent
from src.utils.logger import get_logger

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

@pytest.fixture
def agent():
    """
    Fixture to create an instance of OpenAIAgent for testing.
    """
    return OpenAIAgent()

@pytest.fixture
def log_sink():
    """
    Fixture to capture loguru log messages.
    
    Returns a tuple (sink, remove_sink) where 'sink' is a list that will be appended
    with log messages, and 'remove_sink' is a function to remove the sink from the logger.
    """
    logs = []

    def sink(message):
        logs.append(message.record["message"])

    logger = get_logger()
    # Add the sink with INFO level to capture retry messages
    sink_id = logger.add(sink, level="INFO")
    
    yield logs

    # Remove the sink after the test completes.
    logger.remove(sink_id)

async def test_retry_log_message(agent, log_sink):
    """
    Test that the agent's retry logic logs a specific message upon API error.

    Mocks the underlying API call (`client.responses.create`) to first raise a
    `RateLimitError` and then return a successful mock response. Uses the `log_sink`
    fixture to capture log output during the `agent.call_model` execution.

    Asserts that at least one of the captured log messages contains the expected
    "Retrying after" substring, confirming the retry mechanism logged correctly.

    Args:
        agent: Fixture providing an `OpenAIAgent` instance.
        log_sink: Fixture providing a list to capture log messages.

    Raises:
        AssertionError: If the expected retry log message is not found.
    """
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "retry test",
        "topic_level_3": "test",
        "overall_keywords": ["test"],
        "domain_keywords": ["test"]
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # First call raises an error, second call returns a valid response.
        mock_create.side_effect = [error_response, mock_response(response_content)]
        await agent.call_model("Test prompt")
    
    # Check that one of the captured log messages contains "Retrying after"
    assert any("Retrying after" in msg for msg in log_sink), "Expected retry log message not found in logs"
