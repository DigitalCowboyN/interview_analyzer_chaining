"""
tests/agents/test_openai_agent.py

Contains unit tests for the `OpenAIAgent` class (`src.agents.openai_agent.py`).

These tests verify:
    - Successful processing of valid API responses.
    - Correct retry logic for `RateLimitError` and generic `APIError`.
    - Correct exception handling for malformed/missing API response data.
    - Correct logging of retry attempts.
    - Initialization error handling.
    - Token usage tracking and metrics.
    - Edge cases and error scenarios.

Key Testing Techniques Used:
    - Asynchronous testing with `pytest-asyncio` (`@pytest.mark.asyncio`).
    - Fixtures (`agent`) for creating test instances.
    - Patching asynchronous methods (`agent.client.responses.create`) with `AsyncMock`.
    - Testing expected exceptions using `pytest.raises`.
    - Capturing log output using the `caplog` fixture.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError, RateLimitError

from src.agents.openai_agent import OpenAIAgent

# Note: Only async tests are marked with @pytest.mark.asyncio individually


def mock_response(content_dict, usage_data=None):
    """
    Creates a mock `openai.responses.CreateResponse` object structure.

    Mimics the expected nested structure: `response.output[0].content[0].text`,
    setting the `text` attribute to the JSON-serialized `content_dict`.

    Args:
        content_dict (dict): The dictionary to serialize into the mock response text.
        usage_data (dict, optional): Usage data to include in the response.

    Returns:
        MagicMock: A mock object ready for use with `patch`.
    """
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    # Add usage data if provided
    if usage_data:
        mock_usage = MagicMock()
        for key, value in usage_data.items():
            setattr(mock_usage, key, value)
        mock_resp.usage = mock_usage
    else:
        mock_resp.usage = None

    return mock_resp


@pytest.fixture
def agent():
    """
    Provides a fresh instance of `OpenAIAgent` for each test function.
    """
    return OpenAIAgent()


# === Test Initialization ===


def test_initialization_missing_api_key():
    """
    Tests that `OpenAIAgent` raises `ValueError` when API key is missing from config.

    This test covers the missing line 70 in coverage.
    """
    with patch("src.agents.openai_agent.config", {"openai": {"api_key": ""}}):
        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIAgent()


def test_initialization_none_api_key():
    """
    Tests that `OpenAIAgent` raises `ValueError` when API key is None.
    """
    with patch("src.agents.openai_agent.config", {"openai": {"api_key": None}}):
        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIAgent()


def test_initialization_success():
    """
    Tests successful initialization of `OpenAIAgent` with valid config.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "openai_api": {"retry": {"max_attempts": 3, "backoff_factor": 2}},
    }

    with patch("src.agents.openai_agent.config", mock_config):
        agent = OpenAIAgent()
        assert agent.model == "gpt-4"
        assert agent.max_tokens == 1000
        assert agent.temperature == 0.7
        assert agent.retry_attempts == 3
        assert agent.backoff_factor == 2


# === Test Successful API Calls ===


@pytest.mark.asyncio
async def test_successful_call_with_realistic_interview_response(agent):
    """
    Tests `call_model` with a realistic technical interview response.

    Uses authentic interview analysis data to verify that the returned dictionary
    correctly parses the JSON content and contains realistic analysis results.
    """
    # Realistic response for: "Can you explain your experience with microservices architecture?"
    response_content = {
        "function_type": "interrogative",
        "structure_type": "complex",
        "purpose": "technical_assessment",
        "topic_level_1": "technical_skills",
        "topic_level_3": "system_architecture",
        "overall_keywords": ["experience", "microservices", "architecture", "explain"],
        "domain_keywords": ["microservices", "architecture", "distributed_systems"],
    }
    # Patch the API call on the agent's client to use AsyncMock
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response(response_content)
        # Use realistic technical interview prompt
        interview_prompt = """
        Analyze this interview question: "Can you explain your experience with microservices architecture?"
        
        Classify the function type, structure, purpose, and identify relevant topics and keywords.
        Provide your response in JSON format.
        """
        response = await agent.call_model(interview_prompt)

        # Verify response structure and realistic content
        assert isinstance(response, dict)

        # Test realistic analysis results
        assert response["function_type"] == "interrogative"  # It's a question
        assert response["structure_type"] == "complex"  # Complex sentence structure
        assert response["purpose"] == "technical_assessment"  # Interview assessment purpose
        assert response["topic_level_1"] == "technical_skills"  # High-level topic
        assert response["topic_level_3"] == "system_architecture"  # Specific topic

        # Test realistic keywords
        overall_keywords = response["overall_keywords"]
        assert "microservices" in overall_keywords
        assert "architecture" in overall_keywords
        assert "experience" in overall_keywords

        domain_keywords = response["domain_keywords"]
        assert "microservices" in domain_keywords
        assert "distributed_systems" in domain_keywords


@pytest.mark.asyncio
async def test_successful_call_with_token_usage_realistic_scenario(agent):
    """
    Tests `call_model` with realistic interview response including token usage data.

    Uses authentic interview analysis to verify that token usage is properly tracked.
    """
    # Realistic response for a candidate's technical explanation
    response_content = {
        "function_type": "declarative",
        "structure_type": "compound",
        "purpose": "experience_sharing",
        "topic_level_1": "technical_experience",
        "topic_level_3": "containerization",
        "overall_keywords": ["working", "Docker", "containers", "years", "production"],
        "domain_keywords": ["Docker", "containerization", "DevOps", "deployment"],
    }
    # Realistic token usage for technical interview analysis
    usage_data = {"total_tokens": 245, "input_tokens": 180, "output_tokens": 65}

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_create.return_value = mock_response(response_content, usage_data)

        # Use realistic technical interview prompt
        candidate_response_prompt = """
        Analyze this candidate response: "I've been working with Docker containers in production for 3 years, 
        managing deployments across multiple environments."
        
        Classify the response and extract relevant technical information.
        """
        response = await agent.call_model(candidate_response_prompt)

        # Verify realistic response content
        assert response["function_type"] == "declarative"  # Statement about experience
        assert response["purpose"] == "experience_sharing"  # Sharing technical background
        assert "Docker" in response["domain_keywords"]  # Technical keyword identified
        assert "containerization" in response["domain_keywords"]  # Related concept

        # Verify realistic token usage tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_called_once_with(245)  # Realistic token count


@pytest.mark.asyncio
async def test_successful_call_without_token_usage(agent):
    """
    Tests `call_model` with successful response but no token usage data.

    This test covers the missing line 176 in coverage (the else branch).
    """
    response_content = {"function_type": "declarative"}

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_create.return_value = mock_response(response_content)  # No usage data
        response = await agent.call_model("Test prompt")

        # Verify response
        assert response["function_type"] == "declarative"

        # Verify metrics tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_not_called()  # Should not be called without usage data


@pytest.mark.asyncio
async def test_successful_call_with_incomplete_token_usage(agent):
    """
    Tests `call_model` with response that has usage object but no total_tokens.

    This covers the case where usage exists but total_tokens is None.
    """
    response_content = {"function_type": "declarative"}

    # Create a mock response with usage but without total_tokens
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_content)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]

    # Create usage object that doesn't have total_tokens attribute
    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50
    # Remove total_tokens attribute to simulate it not being present
    del mock_usage.total_tokens
    mock_resp.usage = mock_usage

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_create.return_value = mock_resp
        response = await agent.call_model("Test prompt")

        # Verify response
        assert response["function_type"] == "declarative"

        # Verify metrics tracking
        mock_metrics.increment_api_calls.assert_called_once()
        mock_metrics.add_tokens.assert_not_called()  # Should not be called without total_tokens


# === Test Retry Logic ===


async def test_retry_on_rate_limit_with_realistic_interview_scenario(agent):
    """
    Tests `call_model` successfully retries after a `RateLimitError` during interview analysis.

    Uses realistic interview content to test retry logic when analyzing
    a challenging technical question that initially hits rate limits.
    """
    # Realistic response for complex system design question
    response_content = {
        "function_type": "interrogative",
        "structure_type": "complex",
        "purpose": "system_design_assessment",
        "topic_level_1": "technical_challenges",
        "topic_level_3": "scalability_design",
        "overall_keywords": ["design", "scalable", "system", "millions", "users"],
        "domain_keywords": ["scalability", "load_balancing", "distributed_systems", "performance"],
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    success_response = mock_response(response_content)

    # Use an async function for side_effect
    call_count = 0

    async def side_effect_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error_response
        else:
            return success_response

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func

        # Use realistic system design interview prompt
        system_design_prompt = """
        Analyze this system design question: "How would you design a scalable system 
        to handle millions of concurrent users for a social media platform?"
        
        Focus on the technical complexity and scalability challenges discussed.
        """
        response = await agent.call_model(system_design_prompt)

        # Verify realistic analysis after retry
        assert response["function_type"] == "interrogative"  # It's a design question
        assert response["purpose"] == "system_design_assessment"  # Complex assessment
        assert "scalability" in response["domain_keywords"]  # Key technical concept
        assert call_count == 2  # Ensure it was called exactly twice (initial + 1 retry)


@pytest.mark.asyncio
async def test_retry_on_api_error(agent):
    """
    Tests `call_model` successfully retries after a generic `APIError`.

    Mocks the API call to raise `APIError` once, then return success.
    Asserts the final returned dictionary matches the successful response content.
    """
    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    error_response = APIError("API error", request=MagicMock(), body=MagicMock())
    success_response = mock_response(response_content)

    # Use an async function for side_effect
    call_count = 0

    async def side_effect_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error_response
        else:
            return success_response

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Assign the async function to side_effect
        mock_create.side_effect = side_effect_func
        response = await agent.call_model("Test prompt")
        assert response["purpose"] == "to state a fact"
        assert call_count == 2  # Ensure it was called exactly twice (initial + 1 retry)


@pytest.mark.asyncio
async def test_max_retry_exceeded(agent):
    """
    Tests `call_model` raises the original `APIError` after exhausting all retries.

    Mocks the API call to *always* raise `RateLimitError`. Asserts that
    `pytest.raises(APIError)` correctly catches the exception after the configured
    number of attempts and that the mock was called the expected number of times.
    """
    error_to_raise = RateLimitError("Rate limit exceeded repeatedly", response=MagicMock(), body=None)
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Simulate the error occurring on every call
        mock_create.side_effect = error_to_raise
        # Expect the specific APIError (or subclass) to be raised after retries
        with pytest.raises(APIError):  # Changed from Exception and removed match
            await agent.call_model("Test prompt for max retries")
        # Verify it was called the expected number of times (initial + retries)
        assert mock_create.call_count == agent.retry_attempts


@pytest.mark.asyncio
async def test_retry_on_unexpected_exception(agent):
    """
    Tests `call_model` retries on unexpected exceptions and eventually raises them.
    """
    error_to_raise = ValueError("Unexpected error")

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = error_to_raise

        with pytest.raises(ValueError, match="Unexpected error"):
            await agent.call_model("Test prompt")

        # Verify it was called the expected number of times
        assert mock_create.call_count == agent.retry_attempts


# === Test Response Validation ===


@pytest.mark.asyncio
async def test_empty_output(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with an empty `output` list.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        # Create a mock response with an empty 'output' list.
        mock_resp = MagicMock()
        mock_resp.output = []
        mock_create.return_value = mock_resp
        with pytest.raises(ValueError, match="No output received from OpenAI API."):
            await agent.call_model("Test prompt")


@pytest.mark.asyncio
async def test_empty_content(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with empty `output[0].content`.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_output.content = []
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        with pytest.raises(ValueError, match="No content received from OpenAI API response."):
            await agent.call_model("Test prompt")


@pytest.mark.asyncio
async def test_empty_message(agent):
    """
    Tests `call_model` raises `ValueError` for an API response with empty/whitespace text.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "   "  # Only whitespace
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp
        # Expect ValueError with the updated message from agent.py
        with pytest.raises(ValueError, match="Received empty response content from OpenAI API."):
            await agent.call_model("Test prompt for empty message")


async def test_malformed_json_response_during_interview_analysis(agent):
    """
    Tests `call_model` returns an empty dict `{}` for a non-JSON API response during interview analysis.

    Uses realistic interview prompt to verify that `JSONDecodeError` is caught internally
    and an empty dict is returned, allowing batch processing to continue.
    """
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_resp = MagicMock()
        mock_output = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Not a JSON string"  # Invalid JSON
        mock_output.content = [mock_content]
        mock_resp.output = [mock_output]
        mock_create.return_value = mock_resp

        # Use realistic interview analysis prompt that might fail parsing
        complex_analysis_prompt = """
        Analyze this complex interview exchange:
        
        Interviewer: "Describe your approach to handling database transactions in a distributed system."
        Candidate: "I use two-phase commit protocols with eventual consistency models..."
        
        Extract technical concepts and classify the discussion complexity.
        """

        # Expected behavior: returns {} and logs error, does not raise JSONDecodeError
        result = await agent.call_model(complex_analysis_prompt)
        assert result == {}

        # Verify error was tracked for this realistic scenario
        mock_metrics.increment_errors.assert_called_once()


# === Test Logging ===


@pytest.mark.asyncio
async def test_retry_log_message(agent, caplog):
    """
    Tests `call_model` logs the correct retry message when retrying API calls.

    Mocks the API call to raise `RateLimitError` once, then succeed.
    Uses `caplog` to capture log records emitted by the agent and verify the
    "Retrying after..." message is present at INFO level.
    """
    # Explicitly set logger level for caplog to capture INFO messages
    caplog.set_level(logging.INFO)

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"],
    }
    error_response = RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [error_response, mock_response(response_content)]
        await agent.call_model("Test prompt")

    # Assert the message is present in the captured log records
    assert (
        "Retrying after" in caplog.text
    ), f"Expected retry log message not found in caplog.text. Captured: {caplog.text}"


# === Test Edge Cases ===


@pytest.mark.asyncio
async def test_zero_retry_attempts(agent):
    """
    Tests `call_model` with zero retry attempts configured.

    This test covers the missing lines 235-244 in coverage (fallback error handling).
    """
    # Set retry attempts to 0 to trigger the fallback logic
    agent.retry_attempts = 0

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        # This should never be called since retry_attempts is 0
        mock_create.side_effect = RateLimitError("Rate limit", response=MagicMock(), body=None)

        # Should raise the fallback exception
        with pytest.raises(
            Exception,
            match="call_model failed after retries without specific exception recorded",
        ):
            await agent.call_model("Test prompt")

        # Verify the call was never made due to retry_attempts being 0
        mock_create.assert_not_called()

        # Verify error was tracked
        mock_metrics.increment_errors.assert_called_once()


@pytest.mark.asyncio
async def test_negative_retry_attempts(agent):
    """
    Tests `call_model` with negative retry attempts configured.

    This also covers the fallback error handling.
    """
    # Set retry attempts to negative value to trigger the fallback logic
    agent.retry_attempts = -1

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        # Should raise the fallback exception
        with pytest.raises(
            Exception,
            match="call_model failed after retries without specific exception recorded",
        ):
            await agent.call_model("Test prompt")

        # Verify the call was never made due to retry_attempts being negative
        mock_create.assert_not_called()

        # Verify error was tracked
        mock_metrics.increment_errors.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_with_last_exception(agent):
    """
    Tests the fallback logic when a last_exception exists but loop exits unexpectedly.
    """
    # Mock an agent with 1 retry attempt
    agent.retry_attempts = 1

    # Create a custom exception that will be stored as last_exception
    test_exception = ValueError("Test exception")

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:

        # Mock the scenario where an exception occurs and is stored as last_exception
        # but somehow the loop exits (this is a contrived scenario for coverage)
        mock_create.side_effect = test_exception

        # Should raise the original exception
        with pytest.raises(ValueError, match="Test exception"):
            await agent.call_model("Test prompt")

        # Verify the call was made once (initial attempt)
        assert mock_create.call_count == 1


# === Test Metrics Integration ===


@pytest.mark.asyncio
async def test_metrics_tracking_on_api_error(agent):
    """
    Tests that metrics are properly tracked when API errors occur after retries.
    """
    error_to_raise = APIError("Persistent API error", request=MagicMock(), body=MagicMock())

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_create.side_effect = error_to_raise

        with pytest.raises(APIError):
            await agent.call_model("Test prompt")

        # Verify error was tracked after retries exhausted
        mock_metrics.increment_errors.assert_called_once()


@pytest.mark.asyncio
async def test_metrics_tracking_on_unexpected_error(agent):
    """
    Tests that metrics are properly tracked when unexpected errors occur after retries.
    """
    error_to_raise = ValueError("Unexpected error")

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create, patch(
        "src.agents.openai_agent.metrics_tracker"
    ) as mock_metrics:

        mock_create.side_effect = error_to_raise

        with pytest.raises(ValueError):
            await agent.call_model("Test prompt")

        # Verify error was tracked after retries exhausted
        mock_metrics.increment_errors.assert_called_once()


# === Test Configuration Edge Cases ===


def test_initialization_with_missing_retry_config():
    """
    Tests initialization when retry configuration is missing from config.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        # Missing openai_api section
    }

    with patch("src.agents.openai_agent.config", mock_config):
        agent = OpenAIAgent()
        # Should use default values
        assert agent.retry_attempts == 5  # Default
        assert agent.backoff_factor == 2  # Default


def test_initialization_with_partial_retry_config():
    """
    Tests initialization when retry configuration is partially present.
    """
    mock_config = {
        "openai": {
            "api_key": "test-api-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "openai_api": {
            "retry": {
                "max_attempts": 3
                # Missing backoff_factor
            }
        },
    }

    with patch("src.agents.openai_agent.config", mock_config):
        agent = OpenAIAgent()
        assert agent.retry_attempts == 3  # From config
        assert agent.backoff_factor == 2  # Default


# === Test API Call Parameters ===


@pytest.mark.asyncio
async def test_api_call_parameters(agent):
    """
    Tests that the API call is made with correct parameters.
    """
    response_content = {"function_type": "declarative"}

    with patch.object(agent.client.responses, "create", new_callable=AsyncMock) as mock_create:

        mock_create.return_value = mock_response(response_content)
        await agent.call_model("Test prompt")

        # Verify the API call was made with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        assert call_args[1]["model"] == agent.model
        assert call_args[1]["input"] == "Test prompt"
        assert call_args[1]["max_output_tokens"] == agent.max_tokens
        assert call_args[1]["temperature"] == agent.temperature
        assert call_args[1]["text"] == {"format": {"type": "json_object"}}
        assert "You are a coding assistant" in call_args[1]["instructions"]
