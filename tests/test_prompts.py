import pytest
import yaml
from src.agents.agent import OpenAIAgent

@pytest.fixture
def load_prompts():
    with open("prompts/domain_prompts.yaml") as f:
        domain_prompts = yaml.safe_load(f)
    with open("prompts/task_prompts.yaml") as f:
        task_prompts = yaml.safe_load(f)
    return domain_prompts, task_prompts

@pytest.mark.asyncio
async def test_prompt_attributes(load_prompts):
    domain_prompts, task_prompts = load_prompts
    agent = OpenAIAgent()

    # Test each task prompt
    for prompt_key, prompt in task_prompts.items():
        formatted_prompt = prompt["prompt"].format(sentence="This is a test sentence.")
        response = await agent.call_model(formatted_prompt)
        
        assert "function_type" in response
        assert "structure_type" in response
        assert "purpose" in response
        assert "topic_level_1" in response
        assert "topic_level_3" in response
        assert "overall_keywords" in response
        assert "domain_keywords" in response

    # Test domain prompts
    for keyword in domain_prompts['domain_keywords']:
        response = await agent.call_model(f"Identify the keyword: {keyword}")
        assert "domain_keywords" in response
