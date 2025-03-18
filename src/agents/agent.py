# src/agents/agent.py
# import openai.responses  # Remove this line
import asyncio
import time
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


class OpenAIAgent:
    def __init__(self):
        self.api_key = config["openai"]["api_key"]
        openai.api_key = self.api_key  # This line remains unchanged
        self.model = config["openai"]["model_name"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.temperature = config["openai"]["temperature"]
        self.rate_limit = config.get("openai_api", {}).get("rate_limit", 3000)
        self.retry_attempts = config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        self.backoff_factor = config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)

    async def call_model(self, function_prompt: str) -> str:
        logger.debug(f"Calling OpenAI API with prompt: {function_prompt}")
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                response = await openai.responses.create(  # Use responses.create
                    model=self.model,
                    instructions="You are a coding assistant that talks like a pirate.",
                    input=function_prompt
                )
                logger.debug(f"Received response: {response}")
                return response.output_text  # Change to access output_text

            except openai.RateLimitError as e:
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"Rate limit hit. Retrying after {wait_time} seconds...")
                await asyncio.sleep(wait_time)  # Ensure await is used
                attempt += 1

            except openai.APIError as e:
                wait_time = self.backoff_factor ** attempt
                logger.error(f"OpenAI API error: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)  # Use asyncio.sleep for async context
                attempt += 1

            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                raise

        raise Exception("Max retry attempts exceeded.")        


# Singleton instance for use across pipeline
agent = OpenAIAgent()
