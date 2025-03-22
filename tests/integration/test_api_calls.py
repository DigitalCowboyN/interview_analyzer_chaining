# tests/integration/test_api_calls.py
import pytest
from unittest.mock import patch, MagicMock
import json
from src.agents.agent import OpenAIAgent
from src.models.analysis_result import AnalysisResult

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """Return a mock Response object mimicking openai.responses.create."""
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

async def test_openai_integration():
    agent = OpenAIAgent()
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "integration testing",
        "topic_level_3": "API evaluation",
        "overall_keywords": ["integration", "API"],
        "domain_keywords": ["integration"]
    }
    with patch("openai.responses.create", return_value=mock_response(response_content)):
        response = await agent.call_model("Integration test prompt")
    assert isinstance(response, AnalysisResult)
    assert response.function_type == "declarative"
    assert response.structure_type == "simple sentence"
    assert response.overall_keywords == ["integration", "API"]
    assert response.domain_keywords == ["integration"]
