"""
test_openai_api_responses.py

This module contains unit tests for the OpenAI API responses, specifically testing
the functionality of the 'responses.create' method in an asynchronous context.
The tests ensure that the API can be called correctly and that the response
contains the expected attributes.

Usage Example:

1. Run the tests using pytest:
   pytest tests/test_openai_api_responses.py
"""
import os
import pytest
from openai import AsyncOpenAI

# Make sure your OPENAI_API_KEY is set in the environment
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(api_key=API_KEY)

@pytest.mark.asyncio
async def test_responses_create_structured_json():
    """                                                                                                                                                                                                                           
    Test the 'responses.create' method of the OpenAI API in an async context.                                                                                                                                                     
                                                                                                                                                                                                                                  
    This test verifies that the synchronous 'responses.create' method can be                                                                                                                                                      
    called within an asynchronous context using a thread executor. It checks                                                                                                                                                      
    that the response object contains the expected attributes.                                                                                                                                                                    
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The response object has an 'output_text' attribute.                                                                                                                                                                     
        - Optionally, you can assert that the output contains expected content.                                                                                                                                                   
                                                                                                                                                                                                                                  
    Raises:                                                                                                                                                                                                                       
        ValueError: If the OPENAI_API_KEY environment variable is not set.                                                                                                                                                        
    """
    try:
        response = await client.responses.create(
            model="gpt-4o",
            instructions="You are a coding assistant. Return JSON with keys: capital, country.",
            input="What is the capital of France? Respond in JSON.",
            text={"format": {"type": "json_object"}}
        )

        assert response.output, "No output in response"
        first_output = response.output[0]
        assert first_output.content, "No content in first output item"
        output_message = first_output.content[0].text.strip()

        print("\nStructured JSON response:", output_message)

        # Explicit JSON validation
        import json
        parsed_json = json.loads(output_message)
        assert "capital" in parsed_json, "Expected key 'capital' in JSON response"
        assert parsed_json["capital"] == "Paris", "Expected 'capital' value to be 'Paris'"

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
