# tests/test_agent.py
import pytest
from unittest.mock import patch
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent


@pytest.fixture
def agent():
    return OpenAIAgent()


@patch("openai.ChatCompletion.create")
def test_successful_call(mock_create, agent):
    mock_create.return_value = type("obj", (object,), {"choices": [type("obj", (object,), {"message": type("obj", (object,), {"content": "Test response"})})]})

    response = agent.call_model("Test prompt")
    assert response == "Test response"


@patch("openai.ChatCompletion.create")
def test_retry_on_rate_limit(mock_create, agent):
    mock_create.side_effect = [RateLimitError("Rate limit exceeded"), type("obj", (object,), {"choices": [type("obj", (object,), {"message": type("obj", (object,), {"content": "Recovered response"})})]})]

    response = agent.call_model("Test prompt")
    assert response == "Recovered response"


@patch("openai.ChatCompletion.create")
def test_retry_on_api_error(mock_create, agent):
    mock_create.side_effect = [APIError("API error"), type("obj", (object,), {"choices": [type("obj", (object,), {"message": type("obj", (object,), {"content": "Recovered from API error"})})]})]

    response = agent.call_model("Test prompt")
    assert response == "Recovered from API error"


@patch("openai.ChatCompletion.create")
def test_max_retry_exceeded(mock_create, agent):
    mock_create.side_effect = RateLimitError("Rate limit exceeded")

    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        agent.call_model("Test prompt")
