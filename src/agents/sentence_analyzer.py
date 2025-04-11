"""
sentence_analyzer.py

This module defines the SentenceAnalyzer class, which is responsible for analyzing sentences 
using the OpenAI API. The analyzer classifies each sentence along several dimensions such as 
function type, structure type, purpose, topic levels, and keywords. It leverages prompts defined 
in a YAML file (configured in config.yaml) and orchestrates multiple API calls to gather structured 
metadata for each sentence.

Usage Example:
    from src.agents.sentence_analyzer import sentence_analyzer
    results = await sentence_analyzer.analyze_sentences(sentences)

Modifications:
    - If the classification prompts change, update the YAML file referenced in the config.
    - If new classification dimensions are added, include additional API calls in classify_sentence.
    - Ensure that any changes in API output format are mirrored in the extraction logic (e.g., safe_extract).
"""

import json
import asyncio
from typing import Dict, Any, List
from pydantic import ValidationError
from src.agents.agent import agent  # Uses the OpenAIAgent instance for API calls.
from src.agents.context_builder import context_builder  # Builds context for each sentence.
from src.utils.logger import get_logger  # Centralized logging.
from src.utils.helpers import load_yaml  # Helper to load YAML configuration.
from src.config import config  # Project configuration settings.
from src.utils.metrics import metrics_tracker # Import metrics tracker
from src.models.llm_responses import (
    SentenceFunctionResponse,
    SentenceStructureResponse,
    SentencePurposeResponse,
    TopicLevel1Response,
    TopicLevel3Response,
    OverallKeywordsResponse,
    DomainKeywordsResponse,
)

# Initialize the logger.
logger = get_logger()


class SentenceAnalyzer:
    """
    A class to analyze sentences using the OpenAI API.

    This class uses prompts (loaded from a YAML file) to classify sentences into various dimensions:
      - Function type (e.g., declarative, interrogative).
      - Structure type (e.g., simple, compound).
      - Purpose (e.g., informational, persuasive).
      - Topic levels and keywords (overall and domain-specific).

    The analysis is performed by sending formatted prompts to the OpenAI API via a singleton agent,
    and then processing the API responses into a structured dictionary.

    Attributes:
        prompts (dict): The dictionary of prompts loaded from the specified YAML file. These prompts 
                        define how the classification is performed.
    """
    
    def __init__(self):
        # Load the classification prompts from the YAML file specified in the configuration.
        prompts_path = config["classification"]["local"]["prompt_files"]["no_context"]
        self.prompts = load_yaml(prompts_path)

    async def classify_sentence(self, sentence: str, contexts: Dict[str, str]) -> Dict[str, Any]:
        """
        Classify a single sentence across all required dimensions using concurrent API calls
        and validate responses using Pydantic models.

        The method prepares prompts for seven classification dimensions, executes API calls
        concurrently using asyncio.gather, validates each response against its corresponding
        Pydantic model, and combines the results into a single dictionary. If validation fails
        for a response, a warning is logged, and default values are used.

        Parameters:
            sentence (str): The sentence to classify.
            contexts (Dict[str, str]): A dictionary of context strings for classification.

        Returns:
            Dict[str, Any]: A dictionary containing the classification results. Keys include:
                - "function_type"
                - "structure_type"
                - "purpose"
                - "topic_level_1"
                - "topic_level_3"
                - "overall_keywords"
                - "domain_keywords"
                - "sentence" (the original sentence)

        Raises:
            Exception: If any of the underlying API calls fail after retries (propagated from asyncio.gather).
        """
        results = {}

        # --- Prepare Prompts ---
        function_prompt = self.prompts["sentence_function_type"]["prompt"].format(sentence=sentence)
        structure_prompt = self.prompts["sentence_structure_type"]["prompt"].format(sentence=sentence)
        purpose_prompt = self.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=contexts["observer"]
        )
        topic_lvl1_prompt = self.prompts["topic_level_1"]["prompt"].format(
            sentence=sentence, context=contexts["immediate"]
        )
        topic_lvl3_prompt = self.prompts["topic_level_3"]["prompt"].format(
            sentence=sentence, context=contexts["broader"]
        )
        overall_keywords_prompt = self.prompts["topic_overall_keywords"]["prompt"].format(
            context=contexts["observer"]
        )
        domain_keywords_str = ", ".join(config.get("domain_keywords", []))
        domain_prompt = self.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=domain_keywords_str
        )

        # --- Execute API Calls Concurrently ---
        tasks = [
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
            logger.error(f"Error during concurrent API calls for sentence '{sentence[:50]}...': {e}")
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
            logger.warning(f"Validation failed for Function Type response: {e}. Response: {function_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["function_type"] = "" # Default value

        # Structure Type
        try:
            parsed = SentenceStructureResponse(**structure_response)
            results["structure_type"] = parsed.structure_type
        except ValidationError as e:
            logger.warning(f"Validation failed for Structure Type response: {e}. Response: {structure_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["structure_type"] = "" # Default value

        # Purpose
        try:
            parsed = SentencePurposeResponse(**purpose_response)
            results["purpose"] = parsed.purpose
        except ValidationError as e:
            logger.warning(f"Validation failed for Purpose response: {e}. Response: {purpose_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["purpose"] = "" # Default value

        # Topic Level 1
        try:
            parsed = TopicLevel1Response(**topic_lvl1_response)
            results["topic_level_1"] = parsed.topic_level_1
        except ValidationError as e:
            logger.warning(f"Validation failed for Topic Level 1 response: {e}. Response: {topic_lvl1_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["topic_level_1"] = "" # Default value

        # Topic Level 3
        try:
            parsed = TopicLevel3Response(**topic_lvl3_response)
            results["topic_level_3"] = parsed.topic_level_3
        except ValidationError as e:
            logger.warning(f"Validation failed for Topic Level 3 response: {e}. Response: {topic_lvl3_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["topic_level_3"] = "" # Default value

        # Overall Keywords
        try:
            parsed = OverallKeywordsResponse(**overall_keywords_response)
            results["overall_keywords"] = parsed.overall_keywords
        except ValidationError as e:
            logger.warning(f"Validation failed for Overall Keywords response: {e}. Response: {overall_keywords_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["overall_keywords"] = [] # Default value

        # Domain Keywords
        try:
            parsed = DomainKeywordsResponse(**domain_response)
            # Handle case where LLM might return a string instead of a list (though model expects list)
            # Pydantic handles basic type coercion but explicit check might be safer if needed
            results["domain_keywords"] = parsed.domain_keywords
        except ValidationError as e:
            logger.warning(f"Validation failed for Domain Keywords response: {e}. Response: {domain_response}")
            metrics_tracker.increment_errors() # Track validation error
            results["domain_keywords"] = [] # Default value

        # Add the original sentence to the results.
        results["sentence"] = sentence
        logger.debug(f"Completed analysis for sentence: {sentence[:50]}...")
        return results

    async def analyze_sentences(self, sentences: list) -> List[Dict[str, Any]]:
        """
        Analyze a list of sentences and classify each one.
        
        This method builds context for the entire list of sentences using the context builder.
        It then iterates over each sentence, calling classify_sentence to perform individual 
        classification, and augments each result with a sentence_id and the original sentence text.
        
        Parameters:
            sentences (list): A list of sentences to analyze.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the classification results 
                                  for a sentence. Each dictionary will have keys such as "function_type",
                                  "structure_type", etc., and additional keys "sentence_id" and "sentence".
        
        Raises:
            ValueError: If the sentences list is empty or if contexts cannot be built.
        """
        # Build contexts for all sentences. The context builder returns a dictionary for each sentence.
        contexts = context_builder.build_all_contexts(sentences)
        results = []  # List to accumulate analysis results.

        # Process each sentence with its corresponding context.
        for idx, sentence in enumerate(sentences):
            result = await self.classify_sentence(sentence, contexts[idx])
            # Augment the result with the sentence ID and the original sentence.
            result.update({"sentence_id": idx, "sentence": sentence})
            logger.debug(f"Completed analysis for sentence ID {idx}")
            results.append(result)

        return results


# Singleton instance for pipeline use.
sentence_analyzer = SentenceAnalyzer()
