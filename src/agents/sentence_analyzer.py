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
from src.agents.agent import agent  # Uses the OpenAIAgent instance for API calls.
from src.agents.context_builder import context_builder  # Builds context for each sentence.
from src.utils.logger import get_logger  # Centralized logging.
from src.utils.helpers import load_yaml  # Helper to load YAML configuration.
from src.config import config  # Project configuration settings.

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
        Classify a single sentence across all required dimensions using concurrent API calls.

        The method prepares prompts for seven classification dimensions (function type, 
        structure type, purpose, topic level 1, topic level 3, overall keywords, and 
        domain-specific keywords). It then executes these API calls concurrently using 
        asyncio.gather. Finally, it extracts relevant information from each API response 
        using a helper function and combines them into a single dictionary.

        Parameters:
            sentence (str): The sentence to classify.
            contexts (Dict[str, str]): A dictionary of context strings (e.g., observer, immediate, broader)
                                       that provide additional context for classification.

        Returns:
            Dict[str, Any]: A dictionary containing the classification results. Expected keys include:
                - "function_type"
                - "structure_type"
                - "purpose"
                - "topic_level_1"
                - "topic_level_3"
                - "overall_keywords"
                - "domain_keywords"
                - "sentence" (the original sentence)

        Raises:
            AssertionError: If an expected attribute is missing in the API response (this error may be raised 
                            indirectly through safe_extract if a key is not found).
            Exception: If any of the underlying API calls fail after retries.
        """
        results = {}

        def safe_extract(response: Dict[str, Any], key: str) -> Any:
            """
            Helper function to extract a value from a response dictionary in a case-insensitive manner.

            Parameters:
                response (Dict[str, Any]): The API response dictionary.
                key (str): The key to extract (case-insensitive).

            Returns:
                Any: The value associated with the key, or an empty string if the key is not found.
            """
            if not isinstance(response, dict): # Add check if response is a dict
                logger.error(f"Expected dict but got {type(response)} for key '{key}': {response}")
                return ""
            key = key.lower()
            # Convert all keys to lowercase to perform a case-insensitive lookup.
            return {k.lower(): v for k, v in response.items()}.get(key, "")

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
            # Gather results from all concurrent API calls
            api_responses = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during concurrent API calls for sentence '{sentence[:50]}...': {e}")
            # Depending on desired behavior, you might return partial results,
            # a specific error structure, or re-raise the exception.
            # For now, let's re-raise to signal failure.
            raise

        # Unpack responses (assuming gather preserves order)
        (
            function_response,
            structure_response,
            purpose_response,
            topic_lvl1_response,
            topic_lvl3_response,
            overall_keywords_response,
            domain_response,
        ) = api_responses

        # --- Process Results ---
        results["function_type"] = safe_extract(function_response, "function_type")
        results["structure_type"] = safe_extract(structure_response, "structure_type")
        results["purpose"] = safe_extract(purpose_response, "purpose")
        results["topic_level_1"] = safe_extract(topic_lvl1_response, "topic_level_1")
        results["topic_level_3"] = safe_extract(topic_lvl3_response, "topic_level_3")
        
        # Handle keyword extraction (checking response type before .get)
        if isinstance(overall_keywords_response, dict):
             results["overall_keywords"] = overall_keywords_response.get("overall_keywords", [])
        else:
            logger.error(f"Expected dict for overall_keywords, got {type(overall_keywords_response)}: {overall_keywords_response}")
            results["overall_keywords"] = []

        if isinstance(domain_response, dict):
            domain_keywords = domain_response.get("domain_keywords", [])
            # If domain_keywords is returned as a string, split it into a list.
            if isinstance(domain_keywords, str):
                domain_keywords = [kw.strip() for kw in domain_keywords.split(",") if kw.strip()]
            results["domain_keywords"] = domain_keywords
        else:
            logger.error(f"Expected dict for domain_keywords, got {type(domain_response)}: {domain_response}")
            results["domain_keywords"] = []
            
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
            logger.debug(f"Finalized result: {result}")
            results.append(result)

        return results


# Singleton instance for pipeline use.
sentence_analyzer = SentenceAnalyzer()
