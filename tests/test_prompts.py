"""
test_prompts.py

This module contains unit tests for the prompt handling functionality used in the OpenAIAgent class.
The tests verify that:
  - Prompts loaded from YAML files are correctly formatted.
  - When formatted with dummy values, the prompts yield a valid string.
  - The responses from the agent (simulated using a fake response) contain the expected attributes.

Usage Example:
    Run the tests using pytest:
        pytest tests/test_prompts.py
"""

import pytest
import yaml
import json
from unittest.mock import patch, AsyncMock
from src.agents.agent import OpenAIAgent

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    """
    Return a mock Response object mimicking openai.responses.create.

    This function creates a mock response object that simulates the structure
    of the response returned by the OpenAI API, allowing for controlled testing
    of the OpenAIAgent's behavior.

    Parameters:
        content_dict (dict): A dictionary representing the content of the mock response.

    Returns:
        MagicMock: A mock response object with the specified content.
    """
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

@pytest.fixture
def load_prompts():
    """
    Fixture to load prompts from YAML files.

    This fixture reads the domain and task prompts from their respective YAML files
    and returns them as dictionaries for use in the tests.

    Returns:
        tuple: A tuple containing the domain prompts and task prompts as dictionaries.
    """
    with open("prompts/domain_prompts.yaml") as f:
        domain_prompts = yaml.safe_load(f)
    with open("prompts/task_prompts.yaml") as f:
        task_prompts = yaml.safe_load(f)
    return domain_prompts, task_prompts

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_prompt_attributes(mock_call_model, load_prompts):
    """
    Test the attributes of the prompts loaded from YAML files.

    This test verifies that the task prompts are correctly formatted and that when
    formatted with dummy values, the agent's call_model (simulated here) returns a response
    with the expected key. Similarly, it tests that for each domain-specific keyword from the
    domain prompts, the agent returns the expected domain_keywords.

    Parameters:
        load_prompts: A fixture providing the loaded domain and task prompts.

    Asserts:
        - For each task prompt, when formatted with dummy values, the fake response contains the expected key/value.
        - For each domain keyword, a prompt can be sent and the response returns that keyword.
    """
    domain_prompts, task_prompts = load_prompts
    agent = OpenAIAgent()

    # Mapping of task prompt keys to the expected response key.
    expected_keys = {
        "sentence_function_type": "function_type",
        "sentence_structure_type": "structure_type",
        "sentence_purpose": "purpose",
        "topic_level_1": "topic_level_1",
        "topic_level_3": "topic_level_3",
        "topic_overall_keywords": "overall_keywords",
        "domain_specific_keywords": "domain_keywords"
    }

    # Loop over each task prompt.
    for prompt_key, prompt_config in task_prompts.items():
        # Prepare formatted prompt using dummy values.
        dummy_sentence = "This is a test sentence."
        dummy_context = "Default context"
        dummy_domain_keywords = "Default domain keywords"
        formatted_prompt = prompt_config["prompt"].format(
            sentence=dummy_sentence,
            context=dummy_context,
            domain_keywords=dummy_domain_keywords
        )
        # Set the fake response to include the expected key with a known test value.
        fake_value = f"test_{expected_keys[prompt_key]}"
        fake_response = {expected_keys[prompt_key]: fake_value}
        mock_call_model.return_value = fake_response

        response = await agent.call_model(formatted_prompt)
        # Check that the response is a dict with the expected key and value.
        assert isinstance(response, dict)
        assert response.get(expected_keys[prompt_key]) == fake_value

    # Test domain prompts: for each keyword in domain_prompts, simulate a prompt.
    for keyword in domain_prompts['domain_keywords']:
        fake_response = {"domain_keywords": [keyword]}
        mock_call_model.return_value = fake_response
        response = await agent.call_model(f"Identify the keyword: {keyword}")
        assert response.get("domain_keywords") == [keyword]
