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
    Return a mock Response object mimicking openai.responses.create.
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
    Test that the retry logic logs a warning message when an API error occurs.

    This test simulates a RateLimitError on the first API call, followed by a successful response.
    It uses a custom Loguru sink to capture log messages and asserts that one of the messages
    contains the substring "Retrying after".
    
    Asserts:
        - At least one captured log message contains "Retrying after".
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
