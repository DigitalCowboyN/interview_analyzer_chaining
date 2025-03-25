"""                                                                                                                                                                                                                               
test_openai_responses.py                                                                                                                                                                                                          
                                                                                                                                                                                                                                  
This module contains unit tests for the OpenAI API responses, specifically testing                                                                                                                                                
the functionality of the 'responses.create' method in an asynchronous context.                                                                                                                                                    
The tests ensure that the API can be called correctly and that the response                                                                                                                                                       
contains the expected attributes.                                                                                                                                                                                                 
                                                                                                                                                                                                                                  
Usage Example:                                                                                                                                                                                                                    
                                                                                                                                                                                                                                  
1. Run the tests using pytest:                                                                                                                                                                                                    
   pytest tests/test_openai_responses.py                                                                                                                                                                                          
"""
import os
import pytest
import asyncio
from openai import OpenAI

# Make sure your OPENAI_API_KEY is set in the environment
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=API_KEY)

@pytest.mark.asyncio
async def test_responses_create():
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
        # Synchronous method to call
        def sync_create():
            return client.responses.create(
                model="gpt-4",
                instructions="You are a coding assistant.",
                input="What is the capital of France?"
            )

        # Run sync method inside a thread so we can 'await' the result
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, sync_create)

        # Check that we got a 'response' object with an 'output_text' attribute
        assert hasattr(response, "output_text"), "Expected a 'response' object with 'output_text'"
        
        # Optionally, you can assert something about the answer
        # For example, that it contains "Paris"
        # assert "Paris" in response.output_text, "Expected 'Paris' in the output_text"

        print("\nTest passed. Model answer was:", response.output_text)

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
