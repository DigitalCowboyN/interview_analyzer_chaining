"""
base_agent.py

Abstract base class defining the interface for LLM agents.
Enables multi-provider support with consistent API across OpenAI, Anthropic, and future providers.

Key responsibilities:
- Define standard interface for LLM API calls
- Ensure consistent return types and error handling
- Enable provider-agnostic code in sentence analysis pipeline
- Support future features (prompt caching, batch processing)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMAgent(ABC):
    """
    Abstract base class for LLM agents.

    All provider-specific agents must implement this interface to ensure
    compatibility with the sentence analysis pipeline.

    Design Principles:
    - Maintain exact signature of original OpenAIAgent.call_model()
    - Return Dict[str, Any] for compatibility with existing Pydantic validation
    - Providers handle their own retry logic internally
    - Graceful error handling (return {} on JSON decode errors)
    - Integrated metrics tracking
    """

    def __init__(self):
        """Initialize agent with provider-specific configuration."""
        pass

    @abstractmethod
    async def call_model(self, function_prompt: str) -> Dict[str, Any]:
        """
        Call the LLM with a prompt and return parsed JSON response.

        This is the core method that all sentence analysis flows through.
        The pipeline makes 7 concurrent calls per sentence to analyze:
        - Function type (declarative, interrogative, etc.)
        - Structure type (simple, compound, complex)
        - Purpose (statement, query, explanation, etc.)
        - Topic Level 1 (high-level categorization)
        - Topic Level 3 (detailed categorization)
        - Overall keywords
        - Domain keywords

        Args:
            function_prompt: The prompt string instructing the model to return JSON.
                           Includes sentence text and optional context.

        Returns:
            Dict[str, Any]: Parsed JSON response as dictionary.
                          Returns {} on JSON decode errors (graceful degradation).
                          Expected keys depend on analysis dimension (e.g., "function_type",
                          "structure_type", "purpose", "topic_level_1", etc.)

        Raises:
            APIError: If API call fails after all retry attempts.
            ValueError: If response structure is unexpected (e.g., missing content).
            Exception: For other unexpected errors that persist after retries.

        Implementation Requirements:
        - Must handle retries with exponential backoff internally
        - Must track metrics (API calls, tokens, errors) via metrics_tracker
        - Must log debug/warning/error messages appropriately
        - Must return empty dict {} on JSON decode errors (don't raise)
        - Must parse response into dict before returning
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider name for logging and metrics.

        Returns:
            str: Provider identifier (e.g., 'openai', 'anthropic', 'gemini')
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the configured model name.

        Returns:
            str: Model identifier (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
        """
        pass

    # Phase 2 methods: Default implementations for backward compatibility
    def supports_prompt_caching(self) -> bool:
        """
        Whether this provider supports prompt caching.

        Prompt caching can significantly reduce costs for repeated prompts:
        - Anthropic: Cache write 1.25x, cache hit 0.1x (90% savings)
        - Cache TTL: 5 minutes
        - Useful for repeated context in sentence analysis

        Returns:
            bool: True if provider supports caching, False otherwise
        """
        return False

    def supports_batch_api(self) -> bool:
        """
        Whether this provider supports batch processing.

        Batch APIs process non-urgent requests at discounted rates:
        - Anthropic: 50% discount, processes within 24 hours
        - Useful for bulk analysis, historical data reprocessing

        Returns:
            bool: True if provider supports batch API, False otherwise
        """
        return False
