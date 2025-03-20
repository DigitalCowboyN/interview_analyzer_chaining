# src/agents/sentence_analyzer.py
import json
from typing import Dict, Any, List, AsyncGenerator
from typing import Generator
from src.agents.agent import agent
from src.agents.context_builder import context_builder
from src.utils.logger import get_logger
from src.utils.helpers import load_yaml
from src.config import config
from typing import Generator

logger = get_logger()


class SentenceAnalyzer:
    def __init__(self):
        prompts_path = config["classification"]["local"]["prompt_files"]["no_context"]
        self.prompts = load_yaml(prompts_path)

    async def classify_sentence(self, sentence: str, contexts: Dict[str, str]) -> Dict[str, Any]:
        """Classify a single sentence across all required dimensions."""
        results = {}

        # Function type classification (no context)
        function_prompt = self.prompts["sentence_function_type"]["prompt"].format(sentence=sentence)
        response = await agent.call_model(function_prompt)
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["function_type"] = response_json.get("function_type")

        # Structure type classification (no context)
        structure_prompt = self.prompts["sentence_structure_type"]["prompt"].format(sentence=sentence)
        response = await agent.call_model(structure_prompt)
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["structure_type"] = response_json.get("structure_type")

        # Purpose classification (observer context)
        purpose_prompt = self.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=contexts["observer"]
        )
        response = await agent.call_model(purpose_prompt)  # Add await
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["purpose"] = response_json.get("purpose")

        # Topic level 1 (immediate context)
        topic_lvl1_prompt = self.prompts["topic_level_1"]["prompt"].format(
            sentence=sentence, context=contexts["immediate"]
        )
        response = await agent.call_model(topic_lvl1_prompt)  # Add await
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["topic_level_1"] = response_json.get("topic_level_1")

        # Topic level 3 (broader context)
        topic_lvl3_prompt = self.prompts["topic_level_3"]["prompt"].format(
            sentence=sentence, context=contexts["broader"]
        )
        response = await agent.call_model(topic_lvl3_prompt)  # Add await
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["topic_level_3"] = response_json.get("topic_level_3")

        # Overall keywords (overall context)
        overall_keywords_prompt = self.prompts["topic_overall_keywords"]["prompt"].format(
            context=contexts["observer"]
        )
        response = await agent.call_model(overall_keywords_prompt)  # Add await
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["overall_keywords"] = response_json.get("overall_keywords")

        # Domain-specific keywords
        domain_keywords = ", ".join(config.get("domain_keywords", []))
        domain_prompt = self.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=domain_keywords
        )
        response = await agent.call_model(domain_prompt)  # Add await
        response_json = json.loads(response.output[0].content[0].text)  # Parse JSON response
        results["domain_keywords"] = response_json.get("domain_keywords")

        logger.info(f"Sentence analyzed: {sentence[:50]}...")

        return results

    async def analyze_sentences(self, sentences: list) -> List[Dict[str, Any]]:
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
