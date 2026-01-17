"""
test_anthropic_api_messages.py

Live integration tests for Anthropic's Messages API, mirroring the structure
of test_openai_api_responses.py but adapted for Anthropic's API differences.

Key Differences from OpenAI:
- Uses Messages API (client.messages.create) instead of Responses API
- JSON mode via prompt engineering, not native support
- Different response structure: response.content[0].text
- System prompt + messages format vs instructions + input

Requirements:
- ANTHROPIC_API_KEY environment variable must be set
- Internet connection required

Usage:
    pytest tests/integration/test_anthropic_api_messages.py -xvs
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic

# Check for API key
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY not set - skipping Anthropic Messages API tests",
        allow_module_level=True
    )

client = AsyncAnthropic(api_key=API_KEY)


def mock_anthropic_message_response(content_dict):
    """
    Helper function to create a mock Anthropic Messages API response.

    Constructs a MagicMock object with the structure expected from
    client.messages.create: response.content[0].text

    Args:
        content_dict (dict): A dictionary to be JSON-serialized and set as
                            the text attribute of the mock content.

    Returns:
        MagicMock: A mock response object suitable for patching client.messages.create.
    """
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_resp.content = [mock_content]
    return mock_resp


@pytest.mark.asyncio
async def test_messages_create_structured_json():
    """
    Live integration test verifying Anthropic Messages API returns structured JSON.

    Unlike OpenAI, Anthropic doesn't have native JSON mode. This test verifies
    that prompt engineering (via system prompt) successfully produces valid JSON.

    Requires a valid ANTHROPIC_API_KEY environment variable.

    Tests:
    - Messages API call succeeds
    - System prompt + messages structure works correctly
    - Response contains valid JSON via prompt engineering
    - Parsed JSON contains expected keys with valid values

    Raises:
        pytest.fail: If any exception occurs during the API call or assertions.
        ValueError: If ANTHROPIC_API_KEY environment variable is not set.
    """
    try:
        # System prompt instructs Claude to return JSON only
        system_prompt = (
            "You are a helpful assistant that returns only valid JSON. "
            "Do not include any explanatory text, markdown formatting, or code blocks. "
            "Return only the raw JSON object."
        )

        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Return JSON with keys: capital, country."
                }
            ]
        )

        # Check that response has content
        assert response.content, "No content in response"
        assert len(response.content) > 0, "Content list is empty"

        # Extract text from response (different structure from OpenAI)
        output_message = response.content[0].text.strip()
        print("\nStructured JSON response from Anthropic:", output_message)

        # Parse the JSON
        parsed_json = json.loads(output_message)

        # Check that expected keys are present and "capital" is non-empty
        assert (
            "capital" in parsed_json
        ), "Expected key 'capital' not found in JSON response"
        assert (
            isinstance(parsed_json["capital"], str) and parsed_json["capital"].strip()
        ), "The 'capital' value should be a non-empty string"

        # Verify it correctly identified Paris
        assert "paris" in parsed_json["capital"].lower(), (
            f"Expected 'Paris' but got '{parsed_json['capital']}'"
        )

        print(f"  ✓ Successfully parsed JSON: {parsed_json}")
        print(f"  ✓ Capital correctly identified: {parsed_json['capital']}")

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


@pytest.mark.asyncio
async def test_messages_create_malformed_json():
    """
    Unit test verifying error handling for malformed JSON responses.

    Patches client.messages.create to return a mock response containing
    text that is not valid JSON. Asserts that attempting to parse this text
    using json.loads correctly raises a json.JSONDecodeError.

    This mirrors the OpenAI test but uses Anthropic's response structure.

    Args:
        None

    Raises:
        AssertionError: If json.JSONDecodeError is not raised when expected.
    """
    malformed_text = "This is not valid JSON, just plain text"

    # Create a mock response with malformed JSON text
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = malformed_text
    mock_resp.content = [mock_content]

    with patch.object(
        client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_resp

        with pytest.raises(json.JSONDecodeError):
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                system="Return JSON only.",
                messages=[
                    {
                        "role": "user",
                        "content": "Test malformed JSON response."
                    }
                ]
            )

            # Attempt to parse the malformed response
            output_message = response.content[0].text.strip()
            json.loads(output_message)


@pytest.mark.asyncio
async def test_messages_create_with_conversation():
    """
    Live test verifying Anthropic handles multi-turn conversations correctly.

    This tests a feature unique to the Messages API - maintaining conversation
    context across multiple messages.

    Tests:
    - Multi-message conversation structure works
    - Context from previous messages is maintained
    - JSON response still works in conversational context
    """
    try:
        system_prompt = (
            "You are a helpful assistant that returns only valid JSON. "
            "Return only the raw JSON object."
        )

        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": "I'm going to ask you about a country."
                },
                {
                    "role": "assistant",
                    "content": "I'm ready to provide information about a country in JSON format."
                },
                {
                    "role": "user",
                    "content": "What is the capital of Japan? Return JSON with keys: capital, country."
                }
            ]
        )

        # Verify response
        assert response.content, "No content in response"
        output_message = response.content[0].text.strip()
        print("\nConversational JSON response:", output_message)

        parsed_json = json.loads(output_message)
        assert "capital" in parsed_json
        assert "tokyo" in parsed_json["capital"].lower()

        print(f"  ✓ Conversation context maintained")
        print(f"  ✓ JSON parsing works in conversation: {parsed_json}")

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


@pytest.mark.asyncio
async def test_messages_json_prompt_engineering():
    """
    Test that Anthropic's prompt engineering approach to JSON works reliably.

    This is a critical test because unlike OpenAI, Anthropic doesn't have
    native JSON mode. We rely entirely on prompt engineering.

    Tests multiple scenarios to ensure robustness:
    - Simple JSON object
    - Nested JSON structure
    - JSON array
    - JSON with special characters
    """
    system_prompt = (
        "You are a coding assistant that analyzes text. "
        "Always respond with valid JSON only. Do not include any explanatory text. "
        "Return only the JSON object requested in the prompt."
    )

    test_cases = [
        # (prompt, expected_keys)
        (
            "Analyze this sentence: 'The sky is blue.' Return JSON with keys: sentiment, color_mentioned.",
            ["sentiment", "color_mentioned"]
        ),
        (
            "Return JSON with nested structure: {\"analysis\": {\"type\": \"simple\", \"confidence\": \"high\"}}",
            ["analysis"]
        ),
        (
            "Return JSON array of colors: {\"colors\": [\"red\", \"blue\", \"green\"]}",
            ["colors"]
        ),
    ]

    for prompt, expected_keys in test_cases:
        try:
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            output_message = response.content[0].text.strip()
            parsed_json = json.loads(output_message)

            # Verify expected keys are present
            for key in expected_keys:
                assert key in parsed_json, (
                    f"Expected key '{key}' not found in response: {parsed_json}"
                )

            print(f"  ✓ Prompt engineering successful for: {prompt[:50]}...")

        except json.JSONDecodeError as e:
            pytest.fail(
                f"Prompt engineering failed to produce valid JSON.\n"
                f"Prompt: {prompt}\n"
                f"Response: {output_message}\n"
                f"Error: {e}"
            )
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    print("  ✅ All prompt engineering scenarios passed")
