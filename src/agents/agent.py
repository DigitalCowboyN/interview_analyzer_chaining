"""
agent.py

This module defines the OpenAIAgent class, which interacts with the OpenAI API
to analyze sentences and return structured JSON results. It manages API key
configuration, rate limiting, and retry logic for API calls.

Usage Example:

1. Import the agent instance:
   from src.agents.agent import agent

2. Call the model with a prompt:
   result = await agent.call_model("Your function prompt here")
"""
import openai
import asyncio
import json
from src.config import config
from src.models.analysis_result import AnalysisResult
from src.utils.logger import get_logger

logger = get_logger()

class OpenAIAgent:
    """
    A class to interact with the OpenAI API for sentence analysis.

    This class is responsible for managing the OpenAI API key, model parameters,
    and making calls to the OpenAI API to analyze sentences.

    Attributes:
        api_key (str): The API key for authenticating with OpenAI.
        model (str): The model name to be used for API calls.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): The sampling temperature to use for the API call.
        rate_limit (int): The rate limit for API calls.
        retry_attempts (int): The maximum number of retry attempts for failed API calls.
        backoff_factor (int): The factor by which to increase the wait time between retries.
    """
    def __init__(self):
        self.api_key = config["openai"]["api_key"]
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        logger.debug(f"Using OpenAI API key: {'*' * len(self.api_key)}")
        openai.api_key = self.api_key

        self.model = config["openai"]["model_name"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.temperature = config["openai"]["temperature"]
        self.rate_limit = config.get("openai_api", {}).get("rate_limit", 3000)
        self.retry_attempts = config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        self.backoff_factor = config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)

    async def call_model(self, function_prompt: str) -> AnalysisResult:
        """
        Call the OpenAI API with the provided function prompt and return the analysis result.

        Parameters:
            function_prompt (str): The prompt to send to the OpenAI API for analysis.

        Returns:
            AnalysisResult: The structured analysis result returned by the OpenAI API.

        Raises:
            ValueError: If the OpenAI API key is not set.
            json.JSONDecodeError: If the response from the API cannot be decoded as JSON.
            Exception: If the maximum number of retry attempts is exceeded.
        """
        logger.debug(f"Calling OpenAI API with prompt: {function_prompt}")
        attempt = 0

        def sync_create():
            """
            Synchronously create a response from the OpenAI API.

            This function is run in a thread executor to handle the synchronous nature
            of the OpenAI API call.

            Returns:
                The response object from the OpenAI API.
            """
            return openai.responses.create(
                model=self.model,
                instructions=(
                    "You are a coding assistant. Analyze the sentence and return structured JSON "
                    "with keys: function_type, structure_type, purpose, topic_level_1, topic_level_3, overall_keywords, domain_keywords."
                ),
                input=function_prompt
            )

        # Retry logic for handling API call failures
        while attempt < self.retry_attempts:
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, sync_create)
                output_message = response.output[0].content[0].text.strip()
                logger.debug(f"Raw output message: {output_message}")
                output_data = json.loads(output_message)
                
                # If overall_keywords and domain_keywords are returned as comma-separated strings, split them:
                # Uncomment the following two lines if needed:
                # output_data["overall_keywords"] = [kw.strip() for kw in output_data["overall_keywords"].split(",")]
                # output_data["domain_keywords"] = [kw.strip() for kw in output_data["domain_keywords"].split(",")]

                return AnalysisResult(**output_data)

            except (openai.RateLimitError, openai.APIError) as e:
                # Calculate wait time based on backoff factor
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"{type(e).__name__} encountered: Retrying after {wait_time}s (Attempt {attempt+1})")
                await asyncio.sleep(wait_time)
                attempt += 1
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding failed: {e}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                raise

        raise Exception("Max retry attempts exceeded. Please check your API settings or try again later.")

# Singleton instance
agent = OpenAIAgent()
