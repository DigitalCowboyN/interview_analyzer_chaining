"""
agent.py

OpenAI agent that interacts with OpenAI's latest Responses API to analyze sentences and return structured JSON results. Includes robust error handling and proper asynchronous execution.

Usage:
    from src.agents.agent import agent
    result = await agent.call_model("Analyze this sentence...")
"""
# agent.py
import asyncio
import json
from typing import Dict, Any
from openai import AsyncOpenAI
from src.config import config
from src.models.analysis_result import AnalysisResult
from src.utils.logger import get_logger

logger = get_logger()

class OpenAIAgent:
    def __init__(self):
        api_key = config["openai"]["api_key"]
        if not api_key:
            raise ValueError("OpenAI API key is not set.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = config["openai"]["model_name"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.temperature = config["openai"]["temperature"]
        self.retry_attempts = config.get("openai_api", {}).get("retry", {}).get("max_attempts", 5)
        self.backoff_factor = config.get("openai_api", {}).get("retry", {}).get("backoff_factor", 2)

    async def call_model(self, function_prompt: str) -> Dict[str, Any]:
        attempt = 0
        while attempt < self.retry_attempts:
            try:
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

                output_items = response.output
                if not output_items:
                    raise ValueError("No output received from OpenAI API.")

                first_item = output_items[0]
                content_items = first_item.content
                if not content_items:
                    raise ValueError("No content received from OpenAI API response.")

                output_message = content_items[0].text.strip()

                logger.debug(f"Raw output message: {output_message}")

                if not output_message:
                    raise ValueError("Received empty response from OpenAI API.")

                return json.loads(output_message)

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Decoding error: {e}")
                raise

            except Exception as e:
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"API error ({type(e).__name__}): Retrying after {wait_time}s (Attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                attempt += 1

        raise Exception("Max retry attempts exceeded.")


agent = OpenAIAgent()
