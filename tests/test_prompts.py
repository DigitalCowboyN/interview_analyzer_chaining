"""                                                                                                                                                                                                                               
test_prompts.py                                                                                                                                                                                                                   
                                                                                                                                                                                                                                  
This module contains unit tests for the prompt handling functionality in the OpenAIAgent class.                                                                                                                                   
The tests verify that the prompts loaded from YAML files are correctly formatted and that                                                                                                                                         
the responses from the OpenAI API contain the expected attributes.                                                                                                                                                                
                                                                                                                                                                                                                                  
Usage Example:                                                                                                                                                                                                                    
                                                                                                                                                                                                                                  
1. Run the tests using pytest:                                                                                                                                                                                                    
   pytest tests/test_prompts.py                                                                                                                                                                                                   
"""
import pytest
import yaml
import json
from unittest.mock import patch, MagicMock
from src.agents.agent import OpenAIAgent
from src.models.analysis_result import AnalysisResult

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

async def test_prompt_attributes(load_prompts):
    """                                                                                                                                                                                                                           
    Test the attributes of the prompts loaded from YAML files.                                                                                                                                                                    
                                                                                                                                                                                                                                  
    This test verifies that the prompts are correctly formatted and that the                                                                                                                                                      
    responses from the OpenAIAgent contain the expected attributes. It checks                                                                                                                                                     
    both task prompts and domain keywords.                                                                                                                                                                                        
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        load_prompts: A fixture providing the loaded domain and task prompts.                                                                                                                                                     
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The response is an instance of AnalysisResult.                                                                                                                                                                          
        - The function type of the response matches the expected value.                                                                                                                                                           
        - The domain keywords in the response match the expected values.                                                                                                                                                          
    """
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
            # Supply all placeholders expected in the prompt
            formatted_prompt = prompt["prompt"].format(
                sentence="This is a test sentence.",
                context="Default context",
                domain_keywords="Default domain keywords"
            )
            response = await agent.call_model(formatted_prompt)
            assert isinstance(response, AnalysisResult)
            assert response.function_type == "declarative"
        for keyword in domain_prompts['domain_keywords']:
            response = await agent.call_model(f"Identify the keyword: {keyword}")
            assert response.domain_keywords == ["prompt domain"]
