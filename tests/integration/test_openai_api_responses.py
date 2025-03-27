"""
test_openai_api_responses.py

This module contains unit tests for the OpenAI API responses, specifically testing
the functionality of the 'responses.create' method in an asynchronous context.
The tests verify that the API can be called correctly and that the response
contains the expected attributes when properly formatted, and that errors are raised
when the response is malformed.

Usage Example:
    Run the tests using pytest:
        pytest tests/test_openai_api_responses.py
"""

import os
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from openai import AsyncOpenAI

# Make sure your OPENAI_API_KEY is set in the environment.
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(api_key=API_KEY)

def mock_response(content_dict):
    """
    Return a mock Response object mimicking openai.responses.create.

    Parameters:
        content_dict (dict): A dictionary representing the mock content.

    Returns:
        MagicMock: A mock response object with the expected nested structure.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

@pytest.mark.asyncio
async def test_responses_create_structured_json():
    """
    Test the 'responses.create' method of the OpenAI API in an async context.

    This integration test verifies that the API call can be made and that the structured
    JSON response contains the expected attributes.

    Instead of asserting an exact value for "capital", this test verifies:
      - The response object has non-empty output.
      - The output JSON can be parsed.
      - The parsed JSON includes a "capital" key with a non-empty value.
    
    If any error occurs, the test fails.
    """
    try:
        response = await client.responses.create(
            model="gpt-4o",
            instructions="You are a coding assistant. Return JSON with keys: capital, country.",
            input="What is the capital of France? Respond in JSON.",
            text={"format": {"type": "json_object"}}
        )
        # Check that there is output.
        assert response.output, "No output in response"
        first_output = response.output[0]
        assert first_output.content, "No content in first output item"
        output_message = first_output.content[0].text.strip()
        print("\nStructured JSON response:", output_message)

        # Parse the JSON.
        parsed_json = json.loads(output_message)
        # Check that expected keys are present and "capital" is non-empty.
        assert "capital" in parsed_json, "Expected key 'capital' not found in JSON response"
        assert isinstance(parsed_json["capital"], str) and parsed_json["capital"].strip(), \
            "The 'capital' value should be a non-empty string"
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")

@pytest.mark.asyncio
async def test_responses_create_malformed_json():
    """
    Test the behavior of the API call when the response contains malformed JSON.

    This test patches the client's responses.create method to return a response with
    invalid JSON content, verifying that json.loads raises a JSONDecodeError.
    """
    malformed_text = "This is not valid JSON"
    # Create a mock response with malformed JSON text.
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = malformed_text
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    with patch.object(client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_resp
        with pytest.raises(json.JSONDecodeError):
            response = await client.responses.create(
                model="gpt-4o",
                instructions="Return JSON.",
                input="Test malformed JSON response.",
                text={"format": {"type": "json_object"}}
            )
            # Attempt to parse the malformed response.
            first_output = response.output[0]
            output_message = first_output.content[0].text.strip()
            json.loads(output_message)
