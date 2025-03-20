import json
import re
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

    # Define a target sentence and surrounding sentences for context
    target_sentence = "This is a test sentence."
    surrounding_sentences = [
        "This is the first surrounding sentence.",
        "This is the second surrounding sentence.",
        "This is the third surrounding sentence.",
        "This is the fourth surrounding sentence.",
        "This is the fifth surrounding sentence.",
        "This is the sixth surrounding sentence."
    ]

    # Create contexts based on surrounding sentences
    contexts = {
        "immediate": " ".join(surrounding_sentences[:2]),  # Immediate context (2 sentences)
        "broader": " ".join(surrounding_sentences),  # Broader context
        "observer": "This is the observer context."  # Observer context
    }

    # Test each task prompt
    for prompt_key, prompt in task_prompts.items():
        formatted_prompt = prompt["prompt"].format(sentence="This is a test sentence.")
        response = await agent.call_model(formatted_prompt)
        
        # Parse the response to extract the function type
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        function_type = response_json.get("function_type")
        assert function_type == "declarative"  # Replace with expected value
        assert "structure_type" in response_json
        assert "purpose" in response
        assert "topic_level_1" in response
        assert "topic_level_3" in response
        assert "overall_keywords" in response
        assert "domain_keywords" in response

    # Test domain prompts
    for keyword in domain_prompts['domain_keywords']:
        response = await agent.call_model(f"Identify the keyword: {keyword}")
        assert "domain_keywords" in response
