import openai
import asyncio
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()

class OpenAIAgent:
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

    async def call_model(self, function_prompt: str) -> str:
        logger.debug(f"Calling OpenAI API with prompt: {function_prompt}")
        attempt = 0

        # Synchronous method we want to call in a thread.
        def sync_create():
            return openai.responses.create(
                model=self.model,
                instructions="You are a coding assistant that talks like a pirate.",
                input=function_prompt
            )

        while attempt < self.retry_attempts:
            try:
                # Run the sync call in an executor so we can await the result.
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, sync_create)

                logger.debug(f"Received response: {response}")
                # The real openai.responses.create() returns a 'Response' object,
                # which should have an 'output_text' attribute, not be subscriptable.
                return response.output_text

            except openai.RateLimitError as e:
                wait_time = self.backoff_factor ** attempt
                logger.warning(
                    f"Rate limit hit. Retrying after {wait_time} seconds... (Attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)
                attempt += 1

            except openai.APIError as e:
                wait_time = self.backoff_factor ** attempt
                logger.error(
                    f"OpenAI API error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)
                attempt += 1

            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                raise

        raise Exception("Max retry attempts exceeded.")

# Singleton instance (if you need it globally)
agent = OpenAIAgent()
