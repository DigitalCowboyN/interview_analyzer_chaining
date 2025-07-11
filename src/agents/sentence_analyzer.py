"""
sentence_analyzer.py

Defines the SentenceAnalyzer class responsible for performing multi-dimensional
analysis on individual sentences using the OpenAI API.

It orchestrates the process of:
    - Loading classification prompts from a YAML configuration file.
    - Receiving sentences and their pre-built textual contexts.
    - Formatting prompts with sentence and context information.
    - Making concurrent API calls via the `agent` singleton for each analysis dimension.
    - Validating the structured JSON responses from the API using Pydantic models.
    - Consolidating results into a dictionary for each sentence.

This analyzer is designed to be used within the main processing pipeline.
"""

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.agents.agent import agent  # Uses the OpenAIAgent instance for API calls.

# Import global config but alias it to avoid direct use unless necessary
from src.config import config as global_config
from src.models.llm_responses import (
    DomainKeywordsResponse,
    OverallKeywordsResponse,
    SentenceFunctionResponse,
    SentencePurposeResponse,
    SentenceStructureResponse,
    TopicLevel1Response,
    TopicLevel3Response,
)
from src.utils.helpers import load_yaml  # Helper to load YAML configuration.

# Removed unused context_builder import here, as it's used in pipeline.py
# Restore global import for deprecated method test -> No longer needed? Removing.
# from src.agents.context_builder import context_builder
from src.utils.logger import get_logger  # Centralized logging.
from src.utils.metrics import metrics_tracker  # Import metrics tracker

# Initialize the logger.
logger = get_logger()


class SentenceAnalyzer:
    """
    Analyzes individual sentences across multiple dimensions using OpenAI.

    Loads classification prompts from configuration and uses the `agent` module
    to make concurrent API calls for dimensions like function, structure, purpose,
    topic, and keywords. Validates API responses using Pydantic models.

    Attributes:
        prompts (Dict[str, Any]): A dictionary containing the loaded classification
                                 prompts from the YAML file specified in the config.
        config (Dict[str, Any]): The configuration dictionary used by the instance.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initializes the SentenceAnalyzer by loading classification prompts."""
        # Use provided config_dict or fall back to global config
        config_to_use = config_dict if config_dict is not None else global_config
        self.config = config_to_use  # Store the config for later use

        # Load the classification prompts using the determined config
        prompts_path = ""  # Initialize prompts_path
        try:
            # Access nested keys carefully
            prompts_path = (
                self.config.get("classification", {})
                .get("local", {})
                .get("prompt_files", {})
                .get("no_context")
            )
            if not prompts_path:
                raise KeyError(
                    "Prompt path 'classification.local.prompt_files.no_context' not found in config."
                )

            self.prompts = load_yaml(prompts_path)
            logger.info(
                f"SentenceAnalyzer initialized with prompts from: {prompts_path}"
            )
        except KeyError as e:
            logger.error(
                f"Failed to get prompt path from config: {e}. Using empty prompts."
            )
            self.prompts = {}
        except FileNotFoundError:
            logger.error(
                f"Prompt file not found at path: {prompts_path}. Using empty prompts."
            )
            self.prompts = {}
        except Exception as e:
            logger.error(
                f"Failed to load prompts yaml from {prompts_path}: {e}", exc_info=True
            )
            self.prompts = {}

    async def classify_sentence(
        self, sentence: str, contexts: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Classifies a single sentence across defined dimensions via concurrent API calls.

        Formats prompts using the sentence and provided contexts, calls the OpenAI API
        concurrently for each classification task (function, structure, purpose, etc.),
        validates the JSON response for each task using Pydantic models defined in
        `src.models.llm_responses`, and aggregates the results.

        If a Pydantic validation error occurs for a specific dimension, a warning is logged,
        an error is tracked in metrics, and a default value (e.g., empty string or list)
        is used for that dimension in the returned dictionary.

        Args:
            sentence (str): The sentence text to classify.
            contexts (Dict[str, str]): A dictionary containing pre-built textual context
                strings for different analysis types (e.g., keys like "immediate_context",
                "observer_context"). These are used in specific prompts.

        Returns:
            Dict[str, Any]: A dictionary containing the classification results for the sentence.
                Keys include analysis dimensions (e.g., "function_type", "topic_level_1",
                "overall_keywords") and the original "sentence".

        Raises:
            Exception: Propagates exceptions raised by `agent.call_model` if API calls
                       fail after retries (e.g., persistent `openai.APIError`).
        """
        results = {}

        # --- Prepare Prompts ---
        function_prompt = self.prompts["sentence_function_type"]["prompt"].format(
            sentence=sentence
        )
        structure_prompt = self.prompts["sentence_structure_type"]["prompt"].format(
            sentence=sentence
        )
        purpose_prompt = self.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=contexts["observer_context"]
        )
        topic_lvl1_prompt = self.prompts["topic_level_1"]["prompt"].format(
            sentence=sentence, context=contexts["immediate_context"]
        )
        topic_lvl3_prompt = self.prompts["topic_level_3"]["prompt"].format(
            sentence=sentence, context=contexts["broader_context"]
        )
        overall_keywords_prompt = self.prompts["topic_overall_keywords"][
            "prompt"
        ].format(context=contexts["observer_context"])
        # Use the instance's config
        domain_keywords_str = ", ".join(self.config.get("domain_keywords", []))
        domain_prompt = self.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=domain_keywords_str
        )

        # --- Execute API Calls Concurrently ---
        tasks: List = [
            agent.call_model(function_prompt),
            agent.call_model(structure_prompt),
            agent.call_model(purpose_prompt),
            agent.call_model(topic_lvl1_prompt),
            agent.call_model(topic_lvl3_prompt),
            agent.call_model(overall_keywords_prompt),
            agent.call_model(domain_prompt),
        ]

        try:
            api_responses = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"Error during concurrent API calls for sentence '{sentence[:50]}...': {e}"
            )
            raise

        # Unpack responses
        (
            function_response,
            structure_response,
            purpose_response,
            topic_lvl1_response,
            topic_lvl3_response,
            overall_keywords_response,
            domain_response,
        ) = api_responses

        # --- Process Results with Pydantic Validation ---

        # Function Type
        try:
            parsed = SentenceFunctionResponse(**function_response)
            results["function_type"] = parsed.function_type
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Function Type response: {e}. Response: {function_response}"
            )
            metrics_tracker.increment_errors("validation_error_function_type")
            results["function_type"] = ""  # Default value

        # Structure Type
        try:
            parsed = SentenceStructureResponse(**structure_response)
            results["structure_type"] = parsed.structure_type
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Structure Type response: {e}. Response: {structure_response}"
            )
            metrics_tracker.increment_errors("validation_error_structure_type")
            results["structure_type"] = ""  # Default value

        # Purpose
        try:
            parsed = SentencePurposeResponse(**purpose_response)
            results["purpose"] = parsed.purpose
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Purpose response: {e}. Response: {purpose_response}"
            )
            metrics_tracker.increment_errors("validation_error_purpose")
            results["purpose"] = ""  # Default value

        # Topic Level 1
        try:
            parsed = TopicLevel1Response(**topic_lvl1_response)
            results["topic_level_1"] = parsed.topic_level_1
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Topic Level 1 response: {e}. Response: {topic_lvl1_response}"
            )
            metrics_tracker.increment_errors("validation_error_topic_level_1")
            results["topic_level_1"] = ""  # Default value

        # Topic Level 3
        try:
            parsed = TopicLevel3Response(**topic_lvl3_response)
            results["topic_level_3"] = parsed.topic_level_3
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Topic Level 3 response: {e}. Response: {topic_lvl3_response}"
            )
            metrics_tracker.increment_errors("validation_error_topic_level_3")
            results["topic_level_3"] = ""  # Default value

        # Overall Keywords
        try:
            parsed = OverallKeywordsResponse(**overall_keywords_response)
            results["overall_keywords"] = parsed.overall_keywords
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Overall Keywords response: {e}. Response: {overall_keywords_response}"
            )
            metrics_tracker.increment_errors("validation_error_overall_keywords")
            results["overall_keywords"] = []  # Default value

        # Domain Keywords
        try:
            parsed = DomainKeywordsResponse(**domain_response)
            # Handle case where LLM might return a string instead of a list (though model expects list)
            # Pydantic handles basic type coercion but explicit check might be safer if needed
            results["domain_keywords"] = parsed.domain_keywords
        except ValidationError as e:
            logger.warning(
                f"Validation failed for Domain Keywords response: {e}. Response: {domain_response}"
            )
            metrics_tracker.increment_errors("validation_error_domain_keywords")
            results["domain_keywords"] = []  # Default value

        # Add the original sentence to the results.
        results["sentence"] = sentence
        logger.debug(f"Completed analysis for sentence: {sentence[:50]}...")
        return results

    # --- Deprecated Method ---
    # async def analyze_sentences(self, sentences: list) -> List[Dict[str, Any]]:
    #     """
    #     DEPRECATED: Analyzes a list of sentences, building contexts internally.
    #
    #     This method is inefficient as it rebuilds context for each call.
    #     Use the pipeline orchestrator (`run_pipeline` or `process_file`) which injects
    #     an `AnalysisService` that manages context building and analysis more effectively.
    #
    #     Args:
    #         sentences (list): A list of sentence strings to analyze.
    #
    #     Returns:
    #         List[Dict[str, Any]]: A list of dictionaries, where each contains the
    #             classification results for a sentence, augmented with "sentence_id"
    #             and the original "sentence".
    #
    #     Raises:
    #         ValueError: If the sentences list is empty.
    #     """
    #     # Build contexts for all sentences. The context builder returns a dictionary for each sentence.
    #     # Note: This internal context building is less efficient than the pipeline's approach.
    #     logger.warning("analyze_sentences is deprecated and less efficient; use the main pipeline.")
    #     if not sentences:
    #          raise ValueError("Input sentence list cannot be empty.")
    #
    #     # !!! INCORRECT USAGE - Causes NameError: context_builder not defined !!!
    #     # contexts = context_builder.build_all_contexts(sentences)
    #     contexts = {} # Placeholder if we were to keep the method structure
    #     logger.error("analyze_sentences method is deprecated and contains non-functional context building logic.")
    #
    #     results = []  # List to accumulate analysis results.
    #
    #     # Process each sentence with its corresponding context.
    #     for idx, sentence in enumerate(sentences):
    #         # Need to handle potential KeyError if context building failed, though unlikely
    #         sentence_context = contexts.get(idx)
    #         if sentence_context is None:
    #              logger.error(f"Context missing for sentence index {idx}. Skipping analysis for this sentence.")
    #              # metrics_tracker.increment_errors() # TODO (Metrics): Reinstate
    #              continue
    #
    #         try:
    #              result = await self.classify_sentence(sentence, sentence_context)
    #              # Augment the result with the sentence ID and the original sentence.
    #              # Ensure keys from classify_sentence are present before updating
    #              result.update({"sentence_id": idx, "sentence": sentence}) # sentence is already in result
    #              result["sentence_id"] = idx # Just add id
    #              logger.debug(f"Completed analysis for sentence ID {idx}")
    #              results.append(result)
    #         except Exception as e:
    #              # Log error from classify_sentence if it propagates
    #              logger.error(f"Error analyzing sentence ID {idx}: {e}", exc_info=True)
    #              # metrics_tracker.increment_errors() # TODO (Metrics): Reinstate
    #              # Skip appending failed analysis
    #
    #     return results


# Note: This module does not export a singleton instance by default.
# Instances are created where needed (e.g., in the pipeline workers).
