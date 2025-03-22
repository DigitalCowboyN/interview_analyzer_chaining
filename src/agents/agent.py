import openai
import asyncio
import json
from src.config import config
from src.models.analysis_result import AnalysisResult
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

    async def call_model(self, function_prompt: str) -> AnalysisResult:
        logger.debug(f"Calling OpenAI API with prompt: {function_prompt}")
        attempt = 0

        def sync_create():
            return openai.responses.create(
                model=self.model,
                instructions=(
                    "You are a coding assistant. Analyze the sentence and return structured JSON "
                    "with keys: function_type, structure_type, purpose, topic_level_1, topic_level_3, overall_keywords, domain_keywords."
                ),
                input=function_prompt
            )

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

        raise Exception("Max retry attempts exceeded.")

# Singleton instance
agent = OpenAIAgent()
