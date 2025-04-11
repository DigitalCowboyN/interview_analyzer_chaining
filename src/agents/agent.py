"""
agent.py

This module defines the OpenAIAgent class, which is responsible for interacting with
OpenAI's Responses API. The agent sends a prompt (a text input) to the API and expects
a strictly formatted JSON response. The JSON is then parsed into a Python dictionary.

Usage:
    from src.agents.agent import agent
    result = await agent.call_model("Analyze this sentence...")

Key points:
    - Uses asynchronous execution with asyncio.
    - Incorporates retry logic with exponential backoff to handle transient API errors.
    - Logs detailed debug and error messages for traceability.
    - Expects configuration values (like API key, model name, tokens, etc.) from src/config.py.
    - Relies on a specific output format as defined by the OpenAI Responses API.
    
If modifying this file:
    - Changing configuration keys: Update config.yaml and adjust key names accordingly.
    - Modifying retry logic: Be aware that altering the retry_attempts or backoff_factor
      may affect the agent's behavior when facing API errors.
    - Adjusting the expected JSON response: If the API output structure changes,
      update the parsing logic and any related tests.
"""

import asyncio
import json
from typing import Dict, Any
from openai import AsyncOpenAI, APIError
from src.config import config  # Custom configuration loader; see config.yaml for details
from src.utils.logger import get_logger  # Centralized logger for the project
from src.utils.metrics import metrics_tracker  # Correctly import the global instance

# Get a configured logger for logging debug, warning, and error messages.
logger = get_logger()

class OpenAIAgent:
    """
    OpenAIAgent class wraps around OpenAI's Responses API to process and analyze text.
    
    It sends a prompt to the API, expects a strict JSON response, and parses it.
    If errors occur (like API rate limits or malformed responses), it retries the request
    using an exponential backoff strategy.
    """
    
    def __init__(self):
        """
        Initializes the OpenAIAgent instance with necessary configuration values.
        
        Required configuration values (from config.yaml):
            - openai.api_key: Your OpenAI API key.
            - openai.model_name: The model to be used (e.g., "gpt-4o").
            - openai.max_tokens: Maximum number of tokens in the API response.
            - openai.temperature: Temperature setting for response randomness.
            - openai_api.retry.max_attempts: Maximum number of retry attempts.
            - openai_api.retry.backoff_factor: Factor used to calculate delay between retries.
            
        Raises:
            ValueError: If the OpenAI API key is not set.
        """
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
        self.retry_attempts = config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        self.backoff_factor = config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)

    async def call_model(self, function_prompt: str) -> Dict[str, Any]:
        """
        Calls the OpenAI API with the provided prompt and returns the structured JSON response.
        
        The method:
            - Sends a prompt with strict JSON instructions.
            - Waits for the API response.
            - Parses the JSON from the response.
            - Implements retry logic if errors occur.
        
        Parameters:
            function_prompt (str): The text prompt to be analyzed by the API.
        
        Returns:
            Dict[str, Any]: The parsed JSON response from the API.
        
        Raises:
            ValueError: If the API response is missing expected data.
            json.JSONDecodeError: If the response text is not valid JSON.
            Exception: If maximum retry attempts are exceeded.
        """
        attempt = 0
        last_exception = None # Store last exception for re-raising
        output_message = "" # Initialize for use in exception logging

        while attempt < self.retry_attempts:
            try:
                # Make the asynchronous API call.
                logger.debug(f"Calling OpenAI responses API (Attempt {attempt + 1}/{self.retry_attempts})")
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=(
                        "You are a coding assistant. Respond only in JSON format as explicitly instructed."
                    ),
                    input=function_prompt,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    text={"format": {"type": "json_object"}}
                )

                # Extract the 'output' part of the response.
                output_items = response.output
                if not output_items:
                    raise ValueError("No output received from OpenAI API.")

                # Take the first item in the output list.
                first_item = output_items[0]
                # Extract the 'content' from the first item.
                content_items = first_item.content
                if not content_items:
                    raise ValueError("No content received from OpenAI API response.")

                # Get the text content and remove any leading/trailing whitespace.
                output_message = content_items[0].text.strip()
                # Log the raw output for debugging purposes.
                logger.debug(f"Raw output message: {output_message}")

                if not output_message:
                    raise ValueError("Received empty response content from OpenAI API.")

                # Parse the JSON text into a Python dictionary.
                parsed_response = json.loads(output_message)
                
                # --- Metrics Tracking on Success ---
                metrics_tracker.increment_api_calls()
                
                # Check for usage attribute on the main response object
                usage = getattr(response, 'usage', None) 
                
                # Check if usage exists and has the total_tokens attribute (and it's not None)
                if usage and hasattr(usage, 'total_tokens') and usage.total_tokens is not None:
                    total_tokens_used = usage.total_tokens
                    metrics_tracker.add_tokens(total_tokens_used)
                    
                    # Log detailed usage based on provided documentation structure
                    input_tokens = getattr(usage, 'input_tokens', 'N/A')
                    output_tokens = getattr(usage, 'output_tokens', 'N/A')
                    logger.debug(f"API Call Successful. Total Tokens: {total_tokens_used} (Input: {input_tokens}, Output: {output_tokens})")
                else:
                    # If usage or total_tokens is not available, we just can't track it
                    logger.debug("API Call Successful. Token usage data (total_tokens) not available from responses API usage object for this call.")
                # --- End Metrics Tracking ---

                return parsed_response

            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding failed for prompt: '{function_prompt[:50]}...'. Response: {output_message}. Error: {e}")
                metrics_tracker.increment_errors() 
                # Return empty dict for this error to allow pipeline to continue.
                return {} 

            except APIError as e:
                logger.warning(f"OpenAI API error: {e} (Attempt {attempt + 1}/{self.retry_attempts})")
                last_exception = e
                attempt += 1
                # *** Restore retry logic ***
                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (attempt - 1) # Corrected backoff index
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API error persisted after {self.retry_attempts} attempts: {e}")
                    metrics_tracker.increment_errors() # Track error *after* retries exhausted
                    raise last_exception # Re-raise the specific error caught
            
            except Exception as e:
                logger.warning(f"Unexpected error during API call processing (Attempt {attempt + 1}/{self.retry_attempts}): {type(e).__name__}: {e}")
                last_exception = e
                attempt += 1
                # *** Restore retry logic ***
                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor ** (attempt - 1) # Corrected backoff index
                    logger.info(f"Retrying after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error persisted after {self.retry_attempts} attempts: {e}")
                    metrics_tracker.increment_errors() # Track error *after* retries exhausted
                    raise last_exception # Re-raise the specific error caught
        
        # Fallback if loop finishes unexpectedly (e.g., retry_attempts <= 0)
        logger.critical("Reached end of call_model loop unexpectedly. This indicates a logic error or <= 0 retries configured.")
        metrics_tracker.increment_errors()
        if last_exception:
             raise last_exception # Re-raise the last known exception
        else:
             # If no exception was caught somehow, raise a generic one
             raise Exception("call_model failed after retries without specific exception recorded.")

# Create a global instance of OpenAIAgent for ease of use.
agent = OpenAIAgent()
