"""
test_openai_agent_response.py

This module contains unit tests for the OpenAIAgent class, specifically testing
the structure of the response returned by the OpenAI API. The tests ensure that
the response adheres to the expected format and contains the necessary attributes.

Usage Example:

1. Run the tests using pytest:
   pytest tests/test_openai_agent_response.py
"""
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
    """                                                                                                                                                                                                                           
    Test the structure of the response returned by the OpenAI API.                                                                                                                                                                
                                                                                                                                                                                                                                  
    This test verifies that the response from the OpenAIAgent's call_model method                                                                                                                                                 
    matches the expected structure and contains the correct attributes.                                                                                                                                                           
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The response is an instance of AnalysisResult.                                                                                                                                                                          
        - The topic_level_1 attribute of the response matches the expected value.                                                                                                                                                 
    """
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
