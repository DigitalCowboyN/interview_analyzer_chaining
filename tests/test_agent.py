"""                                                                                                                                                                                                                               
test_agent.py                                                                                                                                                                                                                     
                                                                                                                                                                                                                                  
This module contains unit tests for the OpenAIAgent class, which interacts with the OpenAI API.                                                                                                                                   
The tests cover various scenarios, including successful API calls, handling rate limits,                                                                                                                                          
and retry logic for API errors.                                                                                                                                                                                                   
                                                                                                                                                                                                                                  
Usage Example:                                                                                                                                                                                                                    
                                                                                                                                                                                                                                  
1. Run the tests using pytest:                                                                                                                                                                                                    
   pytest tests/test_agent.py                                                                                                                                                                                                     
"""
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

"""                                                                                                                                                                                                                           
Fixture to create an instance of the OpenAIAgent for testing.                                                                                                                                                                 
                                                                                                                                                                                                                                
Returns:                                                                                                                                                                                                                      
    OpenAIAgent: An instance of the OpenAIAgent class.                                                                                                                                                                        
"""
@pytest.fixture
def agent():
    return OpenAIAgent()

@pytest.mark.asyncio
@patch("openai.responses.create")
async def test_successful_call(mock_create, agent):
    """                                                                                                                                                                                                                           
    Test the successful call to the OpenAI API.                                                                                                                                                                                   
                                                                                                                                                                                                                                  
    This test mocks the API response to ensure that the OpenAIAgent                                                                                                                                                               
    correctly processes a successful response and returns an AnalysisResult.                                                                                                                                                      
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        mock_create: The mocked OpenAI API response.                                                                                                                                                                              
        agent: The instance of OpenAIAgent.                                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The response is an instance of AnalysisResult.                                                                                                                                                                          
        - The function type and overall keywords match the expected values.                                                                                                                                                       
    """
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
    """                                                                                                                                                                                                                           
    Test the retry logic when a rate limit error occurs.                                                                                                                                                                          
                                                                                                                                                                                                                                  
    This test simulates a RateLimitError followed by a successful response                                                                                                                                                        
    to ensure that the OpenAIAgent retries the API call correctly.                                                                                                                                                                
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        mock_create: The mocked OpenAI API response.                                                                                                                                                                              
        agent: The instance of OpenAIAgent.                                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The function type of the response matches the expected value.                                                                                                                                                           
    """
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
    """                                                                                                                                                                                                                           
    Test the retry logic when an API error occurs.                                                                                                                                                                                
                                                                                                                                                                                                                                  
    This test simulates an APIError followed by a successful response                                                                                                                                                             
    to ensure that the OpenAIAgent retries the API call correctly.                                                                                                                                                                
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        mock_create: The mocked OpenAI API response.                                                                                                                                                                              
        agent: The instance of OpenAIAgent.                                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The purpose of the response matches the expected value.                                                                                                                                                                 
    """
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
    """                                                                                                                                                                                                                           
    Test the behavior when the maximum number of retry attempts is exceeded.                                                                                                                                                      
                                                                                                                                                                                                                                  
    This test simulates a RateLimitError that occurs repeatedly, ensuring                                                                                                                                                         
    that the OpenAIAgent raises an exception when the maximum retry attempts                                                                                                                                                      
    are exceeded.                                                                                                                                                                                                                 
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        mock_create: The mocked OpenAI API response.                                                                                                                                                                              
        agent: The instance of OpenAIAgent.                                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - An exception is raised with the message "Max retry attempts exceeded".                                                                                                                                                  
    """
    mock_create.side_effect = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    with pytest.raises(Exception, match="Max retry attempts exceeded"):
        await agent.call_model("Test prompt")
