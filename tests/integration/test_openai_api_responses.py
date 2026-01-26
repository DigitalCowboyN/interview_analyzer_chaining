"""
test_openai_api_responses.py

Contains tests focused on the OpenAI API's `responses.create` endpoint,
particularly its handling of structured JSON responses.

Includes:
- An integration test (`test_responses_create_structured_json`) that makes a live
  API call (requires `OPENAI_API_KEY`) to verify successful structured JSON retrieval.
- A unit test (`test_responses_create_malformed_json`) that mocks the API response
  to ensure correct error handling for malformed JSON.

Usage Example:
    Run the tests using pytest:
        pytest tests/integration/test_openai_api_responses.py

Note: Running the integration test requires a valid OPENAI_API_KEY environment variable.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

# Mark all tests in this module as integration tests (requires real API keys)
pytestmark = pytest.mark.integration

# Make sure your OPENAI_API_KEY is set in the environment.
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    pytest.skip(
        "OPENAI_API_KEY not set - skipping OpenAI API tests",
        allow_module_level=True
    )

client = AsyncOpenAI(api_key=API_KEY)


def mock_response(content_dict):
    """
    Helper function to create a mock Response object mimicking `openai.responses.create`.

    Constructs a `MagicMock` object with the nested structure expected from the
    OpenAI client's response (response -> output -> content -> text).

    Args:
        content_dict (dict): A dictionary to be JSON-serialized and set as the
                             `text` attribute of the innermost mock content.

    Returns:
        MagicMock: A mock response object suitable for patching `client.responses.create`.
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
    Integration test verifying a live call to `client.responses.create` for structured JSON.

    Requires a valid `OPENAI_API_KEY` environment variable.

    Sends a prompt requesting a JSON response and asserts that:
    - The response object contains output.
    - The output contains content.
    - The content text is valid JSON.
    - The parsed JSON contains the expected keys (e.g., "capital") with non-empty string values.

    Args:
        None

    Raises:
        pytest.fail: If any exception occurs during the API call or assertions.
                     (Indicates failure like API errors, network issues, invalid response
                     structure, or failed assertions).
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    try:
        response = await client.responses.create(
            model="gpt-4o",
            instructions="You are a coding assistant. Return JSON with keys: capital, country.",
            input="What is the capital of France? Respond in JSON.",
            text={"format": {"type": "json_object"}},
        )
        # Check that there is output.
        assert response.output, "No output in response"
        first_output = response.output[0]
        # Access content with type assertion - the actual API returns this structure
        content_items = getattr(first_output, "content", None)
        assert content_items, "No content in first output item"
        output_message = content_items[0].text.strip()
        print("\nStructured JSON response:", output_message)

        # Parse the JSON.
        parsed_json = json.loads(output_message)
        # Check that expected keys are present and "capital" is non-empty.
        assert (
            "capital" in parsed_json
        ), "Expected key 'capital' not found in JSON response"
        assert (
            isinstance(parsed_json["capital"], str) and parsed_json["capital"].strip()
        ), "The 'capital' value should be a non-empty string"
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


@pytest.mark.asyncio
async def test_responses_create_malformed_json():
    """
    Unit test verifying error handling for malformed JSON responses.

    Patches `client.responses.create` to return a mock response containing
    text that is not valid JSON. Asserts that attempting to parse this text
    using `json.loads` correctly raises a `json.JSONDecodeError`.

    Args:
        None

    Raises:
        AssertionError: If `json.JSONDecodeError` is not raised when expected.
    """
    malformed_text = "This is not valid JSON"
    # Create a mock response with malformed JSON text.
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = malformed_text
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    with patch.object(
        client.responses, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_resp
        with pytest.raises(json.JSONDecodeError):
            response = await client.responses.create(
                model="gpt-4o",
                instructions="Return JSON.",
                input="Test malformed JSON response.",
                text={"format": {"type": "json_object"}},
            )
            # Attempt to parse the malformed response.
            first_output = response.output[0]
            content_items = getattr(first_output, "content", None)
            assert content_items, "No content in first output item"
            output_message = content_items[0].text.strip()
            json.loads(output_message)
