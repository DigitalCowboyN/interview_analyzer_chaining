# tests/test_openai_response.py
import pytest
import json
from unittest.mock import patch, MagicMock
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

async def test_openai_response_structure():
    agent = OpenAIAgent()
    prompt = "What is the capital of France?"
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "geography",
        "topic_level_3": "capitals",
        "overall_keywords": ["France", "Paris"],
        "domain_keywords": ["geography"]
    }
    with patch("openai.responses.create", return_value=mock_response(response_content)):
        response = await agent.call_model(prompt)
    assert isinstance(response, AnalysisResult)
    assert response.topic_level_1 == "geography"
