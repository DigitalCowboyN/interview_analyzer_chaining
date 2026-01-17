"""
anthropic_agent.py

Anthropic Claude implementation of the LLM agent interface for sentence analysis.
Handles interactions with Anthropic's API using the Messages API.

This implementation matches the OpenAI agent's behavior while handling API differences:
- Messages API instead of Responses API
- System/user message structure
- Different response format
- JSON mode via prompt engineering

Key Functionality:
    - Asynchronous API calls using `anthropic` library
    - Same retry logic and error handling as OpenAI agent
    - Prompt engineering for JSON output (no native JSON mode)
    - Token usage tracking (input + output tokens)
    - Graceful degradation on JSON decode errors
    - Support for future prompt caching (Phase 2)

Usage:
    from src.agents.agent_factory import AgentFactory

    # Get Anthropic agent
    agent = AgentFactory.create_agent("anthropic")
    result = await agent.call_model("Prompt instructing JSON output.")
"""

import asyncio
import json
from typing import Any, Dict

from anthropic import AsyncAnthropic, APIError, RateLimitError

from src.config import config
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker
from .base_agent import BaseLLMAgent

logger = get_logger()


class AnthropicAgent(BaseLLMAgent):
    """
    Anthropic Claude implementation for sentence analysis.

    API Differences from OpenAI:
    - Uses Messages API: client.messages.create()
    - Response structure: response.content[0].text (less nesting)
    - System prompt separate from user message
    - No native JSON mode - uses prompt engineering
    - Token usage: input_tokens + output_tokens (not total_tokens)
    - Supports prompt caching (Phase 2 feature)

    Maintains compatibility with:
    - Same call_model() signature
    - Same retry logic and exponential backoff
    - Same error handling patterns
    - Same metrics tracking integration
    """

    def __init__(self):
        """
        Initialize Anthropic client with configuration.

        Reads API key, model name, token limits, temperature, and retry settings
        from configuration. Uses Claude 3 Haiku (cheapest model).

        Raises:
            ValueError: If anthropic.api_key is not found in configuration
        """
        super().__init__()

        api_key = config["anthropic"]["api_key"]
        if not api_key:
            raise ValueError("Anthropic API key is not set.")

        # Initialize async Anthropic client
        self.client = AsyncAnthropic(api_key=api_key)

        # Model and output settings
        self.model = config["anthropic"]["model_name"]
        self.max_tokens = config["anthropic"]["max_tokens"]
        self.temperature = config["anthropic"]["temperature"]

        # Retry configuration
        self.retry_attempts = (
            config.get("anthropic_api", {}).get("retry", {}).get("max_attempts", 5)
        )
        self.backoff_factor = (
            config.get("anthropic_api", {}).get("retry", {}).get("backoff_factor", 2)
        )

        # System prompt for JSON output (replaces OpenAI's native JSON mode)
        self.system_prompt = (
            "You are a coding assistant that analyzes text. "
            "Always respond with valid JSON only. Do not include any explanatory text. "
            "Return only the JSON object requested in the prompt."
        )

    async def call_model(self, function_prompt: str) -> Dict[str, Any]:
        """
        Call Anthropic Claude with a prompt and return parsed JSON.

        Handles API differences:
        - Messages API instead of Responses API
        - System/user message separation
        - Different response structure
        - JSON format enforcement via prompt engineering

        Args:
            function_prompt: User prompt requesting JSON output

        Returns:
            Dict[str, Any]: Parsed JSON response or {} on error (graceful degradation)

        Raises:
            anthropic.APIError: If API call fails after all retries
            ValueError: If response structure is unexpected
            Exception: For other unexpected errors after retries
        """
        attempt = 0
        last_exception = None
        output_message = ""

        while attempt < self.retry_attempts:
            try:
                logger.debug(
                    f"Calling Anthropic Messages API (Attempt {attempt + 1}/{self.retry_attempts})"
                )

                # Make async API call with Messages API
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": function_prompt
                        }
                    ]
                )

                # Extract text from response (less nesting than OpenAI)
                if not response.content:
                    raise ValueError("No content received from Anthropic API.")

                # Anthropic response: response.content[0].text (direct access)
                output_message = response.content[0].text.strip()
                logger.debug(f"Raw output message: {output_message}")

                if not output_message:
                    raise ValueError("Received empty response from Anthropic API.")

                # Parse JSON
                parsed_response = json.loads(output_message)

                # --- Metrics Tracking ---
                metrics_tracker.increment_api_calls()

                # Track token usage (input + output, no total_tokens field)
                usage = getattr(response, "usage", None)
                if usage and hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                    total_tokens = usage.input_tokens + usage.output_tokens
                    metrics_tracker.add_tokens(total_tokens)
                    logger.debug(
                        f"API Call Successful. Total Tokens: {total_tokens} "
                        f"(Input: {usage.input_tokens}, Output: {usage.output_tokens})"
                    )
                else:
                    logger.debug("API Call Successful. Token usage data not available.")
                # --- End Metrics Tracking ---

                return parsed_response

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON decoding failed for prompt: '{function_prompt[:50]}...'. "
                    f"Response: {output_message}. Error: {e}"
                )
                metrics_tracker.increment_errors()
                # Return empty dict for graceful degradation
                return {}

            except (APIError, RateLimitError) as e:
                logger.warning(
                    f"Anthropic API error: {e} (Attempt {attempt + 1}/{self.retry_attempts})"
                )
                last_exception = e
                attempt += 1

                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"API error persisted after {self.retry_attempts} attempts: {e}"
                    )
                    metrics_tracker.increment_errors()
                    raise last_exception

            except Exception as e:
                logger.warning(
                    f"Unexpected error during API call "
                    f"(Attempt {attempt + 1}/{self.retry_attempts}): {type(e).__name__}: {e}"
                )
                last_exception = e
                attempt += 1

                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Unexpected error persisted after {self.retry_attempts} attempts: {e}"
                    )
                    metrics_tracker.increment_errors()
                    raise last_exception

        # Fallback (should never reach here)
        logger.critical(
            "Reached end of call_model loop unexpectedly. "
            "This indicates a logic error or <= 0 retries configured."
        )
        metrics_tracker.increment_errors()
        if last_exception:
            raise last_exception
        else:
            raise Exception(
                "call_model failed after retries without specific exception recorded."
            )

    def get_provider_name(self) -> str:
        """Return the provider name for logging and metrics."""
        return "anthropic"

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self.model

    def supports_prompt_caching(self) -> bool:
        """Anthropic supports prompt caching (Phase 2 feature)."""
        return True

    def supports_batch_api(self) -> bool:
        """Anthropic supports batch API (Phase 2 feature)."""
        return True
