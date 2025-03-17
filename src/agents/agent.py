# src/agents/agent.py
import openai
import time
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


class OpenAIAgent:
    def __init__(self):
        self.api_key = config["openai"]["api_key"]
        openai.api_key = self.api_key
        self.model = config["openai"]["model_name"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.temperature = config["openai"]["temperature"]
        self.rate_limit = config.get("openai_api", {}).get("rate_limit", 3000)
        self.retry_attempts = config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        self.backoff_factor = config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)

    def call_model(self, prompt: str) -> str:
        """Call the OpenAI API with robust retry logic."""
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                    messages=[{"role": "user", "content": prompt}], 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                result = response['choices'][0]['message']['content'].strip()
                logger.info("Successful OpenAI API call.")
                return result

            except openai.RateLimitError as e:
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"Rate limit hit. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1

            except openai.APIError as e:
                wait_time = self.backoff_factor ** attempt
                logger.error(f"OpenAI API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1

            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                raise

        raise Exception("Max retry attempts exceeded.")


# Singleton instance for use across pipeline
agent = OpenAIAgent()
