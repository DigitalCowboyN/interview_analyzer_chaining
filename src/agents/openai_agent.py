"""
openai_agent.py

OpenAI implementation of the LLM agent interface for sentence analysis.
Handles interactions with OpenAI's API using the Responses API with JSON mode.

This is a refactored version of the original agent.py, now inheriting from
BaseLLMAgent to support multi-provider architecture.

Key Functionality:
    - Asynchronous API calls using `openai` library and `asyncio`.
    - Configurable retry logic with exponential backoff for transient errors.
    - Strict JSON output enforcement via API parameters.
    - Parsing of JSON responses into Python dictionaries.
    - Integration with centralized configuration (`src/config.py`) for API key, model,
      and retry settings.
    - Integration with centralized logging (`src.utils.logger`) and metrics
      (`src.utils.metrics`).

Usage:
    from src.agents.agent_factory import AgentFactory

    # Get OpenAI agent
    agent = AgentFactory.create_agent("openai")
    result = await agent.call_model("Prompt instructing JSON output.")
"""

import asyncio
import json
from typing import Any, Dict

from openai import APIError, AsyncOpenAI

from src.config import config
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker
from .base_agent import BaseLLMAgent

# Get a configured logger for logging debug, warning, and error messages.
logger = get_logger()


class OpenAIAgent(BaseLLMAgent):
    """
    OpenAI implementation of LLM agent for text analysis tasks.

    Encapsulates API call logic, including prompt formatting for JSON output,
    response handling, error retries, and metrics tracking (API calls, token usage).
    Uses configuration settings loaded via `src.config`.

    Inherits from BaseLLMAgent to support multi-provider architecture.
    """

    def __init__(self):
        """
        Initializes the asynchronous OpenAI client and loads configuration.

        Reads API key, model name, token limits, temperature, and retry settings
        from the globally loaded configuration (`src.config.config`).

        Raises:
            ValueError: If the `openai.api_key` is not found in the configuration.
        """
        super().__init__()

        api_key = config["openai"]["api_key"]
        if not api_key:
            raise ValueError("OpenAI API key is not set.")

        # Initialize the asynchronous OpenAI client.
        self.client = AsyncOpenAI(api_key=api_key)
        # Model and output settings for the API call.
        self.model = config["openai"]["model_name"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.temperature = config["openai"]["temperature"]
        # Retry configuration: max attempts and exponential backoff factor.
        self.retry_attempts = (
            config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        )
        self.backoff_factor = (
            config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)
        )

    async def call_model(self, function_prompt: str) -> Dict[str, Any]:
        """
        Asynchronously calls the configured OpenAI model with a prompt.

        Sends the prompt, requests a JSON object response, handles potential API errors
        with configured retries, parses the JSON content, and tracks metrics.

        Args:
            function_prompt (str): The prompt string to send to the OpenAI model.
                                   This prompt should instruct the model to return JSON.

        Returns:
            Dict[str, Any]: The parsed JSON response from the model as a dictionary.
                            Returns an empty dictionary if JSON decoding fails, allowing
                            the pipeline to continue processing other items.

        Raises:
            openai.APIError: If an API error occurs and persists after all retry attempts.
            ValueError: If the API response structure is unexpected (e.g., missing content).
            Exception: For other unexpected errors during the process or if retries are
                       exhausted for non-APIError exceptions.
        """
        attempt = 0
        last_exception = None  # Store last exception for re-raising
        output_message = ""  # Initialize for use in exception logging

        while attempt < self.retry_attempts:
            try:
                # Make the asynchronous API call.
                logger.debug(
                    f"Calling OpenAI responses API (Attempt {attempt + 1}/{self.retry_attempts})"
                )
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=(
                        "You are a coding assistant. Respond only in JSON format as explicitly instructed."
                    ),
                    input=function_prompt,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    text={"format": {"type": "json_object"}},
                )

                # Extract the 'output' part of the response.
                output_items = response.output
                if not output_items:
                    raise ValueError("No output received from OpenAI API.")

                # Take the first item in the output list.
                first_item = output_items[0]
                # Extract the 'content' from the first item.
                content_items = first_item.content  # type: ignore
                if not content_items:
                    raise ValueError("No content received from OpenAI API response.")

                # Get the text content and remove any leading/trailing whitespace.
                output_message = content_items[0].text.strip()  # type: ignore
                # Log the raw output for debugging purposes.
                logger.debug(f"Raw output message: {output_message}")

                if not output_message:
                    raise ValueError("Received empty response content from OpenAI API.")

                # Parse the JSON text into a Python dictionary.
                parsed_response = json.loads(output_message)

                # --- Metrics Tracking on Success ---
                metrics_tracker.increment_api_calls()

                # Check for usage attribute on the main response object
                usage = getattr(response, "usage", None)

                # Check if usage exists and has the total_tokens attribute (and it's not None)
                if (
                    usage
                    and hasattr(usage, "total_tokens")
                    and usage.total_tokens is not None
                ):
                    total_tokens_used = usage.total_tokens
                    metrics_tracker.add_tokens(total_tokens_used)

                    # Log detailed usage based on provided documentation structure
                    input_tokens = getattr(usage, "input_tokens", "N/A")
                    output_tokens = getattr(usage, "output_tokens", "N/A")
                    logger.debug(
                        f"API Call Successful. Total Tokens: {total_tokens_used} "
                        f"(Input: {input_tokens}, Output: {output_tokens})"
                    )
                else:
                    # If usage or total_tokens is not available, we just can't track it
                    logger.debug(
                        "API Call Successful. Token usage data (total_tokens) not available "
                        "from responses API usage object for this call."
                    )
                # --- End Metrics Tracking ---

                return parsed_response

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON decoding failed for prompt: '{function_prompt[:50]}...'. "
                    f"Response: {output_message}. Error: {e}"
                )
                metrics_tracker.increment_errors()
                # Return empty dict for this error to allow pipeline to continue.
                return {}

            except APIError as e:
                logger.warning(
                    f"OpenAI API error: {e} (Attempt {attempt + 1}/{self.retry_attempts})"
                )
                last_exception = e
                attempt += 1
                # *** Restore retry logic ***
                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (
                        attempt - 1
                    )  # Corrected backoff index
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"API error persisted after {self.retry_attempts} attempts: {e}"
                    )
                    metrics_tracker.increment_errors()  # Track error *after* retries exhausted
                    raise last_exception  # Re-raise the specific error caught

            except Exception as e:
                logger.warning(
                    f"Unexpected error during API call processing "
                    f"(Attempt {attempt + 1}/{self.retry_attempts}): {type(e).__name__}: {e}"
                )
                last_exception = e
                attempt += 1
                # *** Restore retry logic ***
                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (
                        attempt - 1
                    )  # Corrected backoff index
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Unexpected error persisted after {self.retry_attempts} attempts: {e}"
                    )
                    metrics_tracker.increment_errors()  # Track error *after* retries exhausted
                    raise last_exception  # Re-raise the specific error caught

        # Fallback if loop finishes unexpectedly (e.g., retry_attempts <= 0)
        logger.critical(
            "Reached end of call_model loop unexpectedly. "
            "This indicates a logic error or <= 0 retries configured."
        )
        metrics_tracker.increment_errors()
        if last_exception:
            raise last_exception  # Re-raise the last known exception
        else:
            # If no exception was caught somehow, raise a generic one
            raise Exception(
                "call_model failed after retries without specific exception recorded."
            )

    def get_provider_name(self) -> str:
        """Return the provider name for logging and metrics."""
        return "openai"

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self.model
