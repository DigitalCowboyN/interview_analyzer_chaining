import pytest
import os
from src.agents.agent import OpenAIAgent

@pytest.mark.asyncio
async def test_openai_integration():
    agent = OpenAIAgent()
    response = await agent.call_model("Test prompt for integration")
    
    assert hasattr(response, 'function_type')
    assert hasattr(response, 'structure_type')
    assert hasattr(response, 'purpose')
    assert hasattr(response, 'topic_level_1')
    assert hasattr(response, 'topic_level_3')
    assert hasattr(response, 'overall_keywords')
    assert hasattr(response, 'domain_keywords')
