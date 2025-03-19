# tests/test_agent.py
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent

# Set the environment variable for logging
os.environ["OPENAI_LOG"] = "debug"

@pytest.fixture
def agent():
    return OpenAIAgent()

@pytest.mark.asyncio
@patch("openai.responses.create", new_callable=AsyncMock)
async def test_successful_call(mock_create, agent):
    # Simulate a single successful response
    mock_create.return_value = {"output_text": "Test response"}

    response = await agent.call_model("Test prompt")
    assert response == "Test response"

@pytest.mark.asyncio
@patch("openai.responses.create", new_callable=AsyncMock)
async def test_retry_on_rate_limit(mock_create, agent):
    # First call => RateLimitError
    # Second call => a dictionary with "output_text"
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}

    mock_create.side_effect = [
        RateLimitError("Rate limit exceeded", response=mock_response, body=None),
        {"output_text": "Recovered response"},
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered response"

@pytest.mark.asyncio
@patch("openai.responses.create", new_callable=AsyncMock)
async def test_retry_on_api_error(mock_create, agent):
    # First call => APIError
    # Second call => a dictionary with "output_text"
    mock_create.side_effect = [
        APIError("API error", request="mock_request", body="mock_body"),
        {"output_text": "Recovered from API error"},
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered from API error"

@pytest.mark.asyncio
@patch("openai.ChatCompletion.acreate", new_callable=AsyncMock)
async def test_max_retry_exceeded(mock_acreate, agent):
    # Force repeated RateLimitError so we hit max retries
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}

    mock_acreate.side_effect = RateLimitError(
        "Rate limit exceeded",
        response=mock_response,
        body=None
    )

    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        await agent.call_model("Test prompt")
