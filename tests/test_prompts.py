# tests/test_prompts.py
import pytest
import yaml
import json
from unittest.mock import patch, MagicMock
from src.agents.agent import OpenAIAgent
from src.models.analysis_result import AnalysisResult

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """Return a mock Response object mimicking openai.responses.create."""
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    # Convert our dictionary to JSON string.
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

@pytest.fixture
def load_prompts():
    with open("prompts/domain_prompts.yaml") as f:
        domain_prompts = yaml.safe_load(f)
    with open("prompts/task_prompts.yaml") as f:
        task_prompts = yaml.safe_load(f)
    return domain_prompts, task_prompts

async def test_prompt_attributes(load_prompts):
    domain_prompts, task_prompts = load_prompts
    agent = OpenAIAgent()

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "prompt testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test prompt"],
        "domain_keywords": ["prompt domain"]
    }
    with patch("openai.responses.create", return_value=mock_response(response_content)):
        for prompt_key, prompt in task_prompts.items():
            # Supply both 'sentence' and 'context' (and any other placeholders required)
            formatted_prompt = prompt["prompt"].format(
                sentence="This is a test sentence.",
                context="Default context"
            )
            response = await agent.call_model(formatted_prompt)
            assert isinstance(response, AnalysisResult)
            assert response.function_type == "declarative"
        for keyword in domain_prompts['domain_keywords']:
            response = await agent.call_model(f"Identify the keyword: {keyword}")
            assert response.domain_keywords == ["prompt domain"]
