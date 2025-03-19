# tests/test_agent.py
import os
import pytest
from unittest.mock import patch, MagicMock
import asyncio
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent

# Set the environment variable for logging (optional)
os.environ["OPENAI_LOG"] = "debug"

@pytest.fixture
def agent():
    """Pytest fixture to provide an OpenAIAgent instance."""
    return OpenAIAgent()

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_successful_call(mock_create, agent):
    """
    Simulates a single successful response from openai.responses.create.
    Agent code uses response.output_text, so let's return a mocked object
    that has that attribute.
    """
    mock_response = MagicMock()
    mock_response.output_text = "Test response"
    mock_create.return_value = mock_response

    response = await agent.call_model("Test prompt")
    assert response == "Test response"

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_retry_on_rate_limit(mock_create, agent):
    """
    First call => RateLimitError,
    Second call => recovers with a mock object that has .output_text
    """
    mock_response = MagicMock()
    mock_response.output_text = "Recovered response"

    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)

    mock_create.side_effect = [
        error_response,      # Raise on first call
        mock_response        # Return success on second call
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered response"

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_retry_on_api_error(mock_create, agent):
    """
    First call => APIError,
    Second call => returns a mock object with .output_text
    """
    mock_response = MagicMock()
    mock_response.output_text = "Recovered from API error"

    error_response = APIError("API error", request="mock_request", body="mock_body")

    mock_create.side_effect = [
        error_response,      # Raise on first call
        mock_response        # Return success on second call
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered from API error"

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_max_retry_exceeded(mock_create, agent):
    """
    Forces repeated RateLimitError so that the agent exhausts its retries
    and raises "Max retry attempts exceeded".
    """
    # Always raise RateLimitError on every call
    mock_create.side_effect = RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None
    )

    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        await agent.call_model("Test prompt")
