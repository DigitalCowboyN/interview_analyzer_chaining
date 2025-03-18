# tests/test_agent.py
import pytest
import pytest
from unittest.mock import patch, AsyncMock
import asyncio
from openai import RateLimitError, APIError
from unittest.mock import MagicMock
from src.agents.agent import OpenAIAgent


@pytest.fixture
def agent():
    return OpenAIAgent()


@patch("openai.responses.create", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_successful_call(mock_create, agent):
    mock_create.return_value = {
        "output_text": "Test response"  # Update to match the new response structure
    }

    response = await agent.call_model("Test prompt")  # Use await
    assert response == "Test response"


@patch("openai.responses.create")
async def test_retry_on_rate_limit(mock_create, agent):
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}
    mock_create.side_effect = [
        RateLimitError("Rate limit exceeded", response=mock_response, body=None),
        {
            "output_text": "Recovered response"  # Update to match the new response structure
        }
    ]  # Ensure proper structure for mock return value

    response = await agent.call_model("Test prompt")  # Use await
    assert response == "Recovered response"


@patch("openai.responses.create")
async def test_retry_on_api_error(mock_create, agent):
    mock_create.side_effect = [
        APIError("API error", request="mock_request", body="mock_body"),
        {
            "output_text": "Recovered from API error"  # Update to match the new response structure
        }
    ]  # Ensure proper structure for mock return value

    response = await agent.call_model("Test prompt")  # Use await
    assert response == "Recovered from API error"


@patch("openai.ChatCompletion.acreate")
async def test_max_retry_exceeded(mock_create, agent):
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.headers = {"x-request-id": "mock_request_id"}
    mock_create.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        agent.call_model("Test prompt")
