# tests/test_agent.py
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent

# Set the environment variable for logging
os.environ["OPENAI_LOG"] = "debug"

# Helper async functions to return the desired responses.
async def recovered_response():
    return {"output_text": "Recovered response"}

async def recovered_api_error_response():
    return {"output_text": "Recovered from API error"}

@pytest.fixture
def agent():
    return OpenAIAgent()

@patch("openai.responses.create", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_successful_call(mock_create, agent):
    # Return a dictionary wrapped by AsyncMock automatically.
    mock_create.return_value = {"output_text": "Test response"}

    response = await agent.call_model("Test prompt")
    assert response == "Test response"

@patch("openai.responses.create", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retry_on_rate_limit(mock_create, agent):
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}
    # First call raises a RateLimitError; second call calls recovered_response.
    mock_create.side_effect = [
        RateLimitError("Rate limit exceeded", response=mock_response, body=None),
        recovered_response  # pass the function itself (not invoked)
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered response"

@patch("openai.responses.create", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retry_on_api_error(mock_create, agent):
    # First call raises an APIError; second call calls recovered_api_error_response.
    mock_create.side_effect = [
        APIError("API error", request="mock_request", body="mock_body"),
        recovered_api_error_response  # pass the function itself (not invoked)
    ]

    response = await agent.call_model("Test prompt")
    assert response == "Recovered from API error"

@patch("openai.ChatCompletion.acreate", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_max_retry_exceeded(mock_create, agent):
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}
    # Always raises a RateLimitError.
    mock_create.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        await agent.call_model("Test prompt")
