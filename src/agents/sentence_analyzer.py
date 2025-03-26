"""                                                                                                                                                                                                                                    
sentence_analyzer.py                                                                                                                                                                                                                   
                                                                                                                                                                                                                                       
This module defines the SentenceAnalyzer class, which is responsible for analyzing                                                                                                                                                     
sentences across various dimensions using the OpenAI API. It classifies sentences                                                                                                                                                      
based on function type, structure type, purpose, topic levels, and keywords.                                                                                                                                                           
                                                                                                                                                                                                                                       
Usage Example:                                                                                                                                                                                                                         
                                                                                                                                                                                                                                       
1. Import the sentence analyzer instance:                                                                                                                                                                                              
   from src.agents.sentence_analyzer import sentence_analyzer                                                                                                                                                                          
                                                                                                                                                                                                                                       
2. Analyze a list of sentences:                                                                                                                                                                                                        
   results = await sentence_analyzer.analyze_sentences(sentences)                                                                                                                                                                      
"""
import json
from typing import Dict, Any, List, AsyncGenerator
from typing import Generator
from src.agents.agent import agent
from src.agents.context_builder import context_builder
from src.utils.logger import get_logger
from src.utils.helpers import load_yaml
from src.config import config

logger = get_logger()


class SentenceAnalyzer:
    # This class analyzes sentences and may classify them using domain-specific keywords.
    """                                                                                                                                                                                                                                
    A class to analyze sentences using the OpenAI API.                                                                                                                                                                                 
                                                                                                                                                                                                                                       
    This class classifies sentences across various dimensions, including function type,                                                                                                                                                
    structure type, purpose, topic levels, and keywords. It utilizes prompts defined in                                                                                                                                                
    a YAML file and manages the interaction with the OpenAI API.                                                                                                                                                                       
                                                                                                                                                                                                                                       
    Attributes:                                                                                                                                                                                                                        
        prompts (dict): The prompts used for classification, loaded from a YAML file.                                                                                                                                                  
    """
    def __init__(self):
        prompts_path = config["classification"]["local"]["prompt_files"]["no_context"]
        self.prompts = load_yaml(prompts_path)

    async def classify_sentence(self, sentence: str, contexts: Dict[str, str]) -> Dict[str, Any]:
        """                                                                                                                                                                                                                            
        Classify a single sentence across all required dimensions.                                                                                                                                                                     
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            sentence (str): The sentence to classify.                                                                                                                                                                                  
            contexts (Dict[str, str]): A dictionary of contexts related to the sentence.                                                                                                                                               
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            Dict[str, Any]: A dictionary containing classification results for the sentence.                                                                                                                                           
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            AssertionError: If the response does not contain the expected attributes.                                                                                                                                                  
        """
        results = {}

        def safe_extract(response: Dict[str, Any], key: str) -> Any:
            """Helper to get a value by key, case-insensitive."""
            key = key.lower()
            return {k.lower(): v for k, v in response.items()}.get(key, "")

        # Function type
        function_prompt = self.prompts["sentence_function_type"]["prompt"].format(sentence=sentence)
        function_response = await agent.call_model(function_prompt)
        results["function_type"] = safe_extract(function_response, "function_type")

        # Structure type
        structure_prompt = self.prompts["sentence_structure_type"]["prompt"].format(sentence=sentence)
        structure_response = await agent.call_model(structure_prompt)
        results["structure_type"] = safe_extract(structure_response, "structure_type")

        # Purpose
        purpose_prompt = self.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=contexts["observer"]
        )
        purpose_response = await agent.call_model(purpose_prompt)
        results["purpose"] = safe_extract(purpose_response, "purpose")

        # Topic level 1
        topic_lvl1_prompt = self.prompts["topic_level_1"]["prompt"].format(
            sentence=sentence, context=contexts["immediate"]
        )
        topic_lvl1_response = await agent.call_model(topic_lvl1_prompt)
        results["topic_level_1"] = safe_extract(topic_lvl1_response, "topic_level_1")

        # Topic level 3
        topic_lvl3_prompt = self.prompts["topic_level_3"]["prompt"].format(
            sentence=sentence, context=contexts["broader"]
        )
        topic_lvl3_response = await agent.call_model(topic_lvl3_prompt)
        results["topic_level_3"] = safe_extract(topic_lvl3_response, "topic_level_3")

        # Overall keywords
        overall_keywords_prompt = self.prompts["topic_overall_keywords"]["prompt"].format(
            context=contexts["observer"]
        )
        overall_keywords_response = await agent.call_model(overall_keywords_prompt)
        results["overall_keywords"] = overall_keywords_response.get("overall_keywords", [])

        # Domain-specific keywords
        domain_keywords_str = ", ".join(config.get("domain_keywords", []))
        domain_prompt = self.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=domain_keywords_str
        )
        domain_response = await agent.call_model(domain_prompt)
        domain_keywords = domain_response.get("domain_keywords", [])
        if isinstance(domain_keywords, str):
            domain_keywords = [kw.strip() for kw in domain_keywords.split(",") if kw.strip()]
        results["domain_keywords"] = domain_keywords

        results["sentence"] = sentence
        logger.debug(f"Completed analysis for sentence: {sentence[:50]}...")
        return results

    async def analyze_sentences(self, sentences: list) -> List[Dict[str, Any]]:
        """                                                                                                                                                                                                                            
        Analyze a list of sentences and classify each one.                                                                                                                                                                             
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            sentences (list): A list of sentences to analyze.                                                                                                                                                                          
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            List[Dict[str, Any]]: A list of dictionaries containing classification results for each sentence.                                                                                                                          
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            ValueError: If the sentences list is empty or contexts cannot be built.                                                                                                                                                    
        """
        contexts = context_builder.build_all_contexts(sentences)
        results = []  # Initialize results list

        for idx, sentence in enumerate(sentences):
            result = await self.classify_sentence(sentence, contexts[idx])
            result.update({"sentence_id": idx, "sentence": sentence})
            logger.debug(f"Completed analysis for sentence ID {idx}")

            # Removed redundant assignments and updates
            logger.debug(f"Finalized result: {result}")
            results.append(result)

        return results  # Return the list of results


# Singleton instance for pipeline use
sentence_analyzer = SentenceAnalyzer()
