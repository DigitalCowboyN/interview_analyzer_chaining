import pytest
from src.agents.agent import OpenAIAgent

@pytest.mark.asyncio
async def test_openai_response_structure():
    agent = OpenAIAgent()
    prompt = "What is the capital of France?"  # A simple prompt to test

    response = await agent.call_model(prompt)

    # Log the entire response to understand its structure
    print(response)  # This will show you the full response object

    # Optionally, you can assert that the response is of the expected type
    assert isinstance(response, AnalysisResult)  # Check for AnalysisResult instance
