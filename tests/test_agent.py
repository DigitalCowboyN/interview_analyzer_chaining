# tests/test_agent.py
import pytest
import json
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIError
from src.agents.agent import OpenAIAgent
from src.models.analysis_result import AnalysisResult

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

@pytest.fixture
def agent():
    return OpenAIAgent()

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_successful_call(mock_create, agent):
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"]
    }
    mock_create.return_value = mock_response(response_content)
    response = await agent.call_model("Test prompt")
    assert isinstance(response, AnalysisResult)
    assert response.function_type == "declarative"
    assert response.overall_keywords == ["test"]

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_retry_on_rate_limit(mock_create, agent):
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"]
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    mock_create.side_effect = [error_response, mock_response(response_content)]
    response = await agent.call_model("Test prompt")
    assert response.function_type == "declarative"

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_retry_on_api_error(mock_create, agent):
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"]
    }
    error_response = APIError("API error", request="mock_request", body="mock_body")
    mock_create.side_effect = [error_response, mock_response(response_content)]
    response = await agent.call_model("Test prompt")
    assert response.purpose == "to state a fact"

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_max_retry_exceeded(mock_create, agent):
    mock_create.side_effect = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        await agent.call_model("Test prompt")
