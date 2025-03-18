# src/agents/sentence_analyzer.py
from typing import Dict, Any, Generator
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
        results["function_type"] = agent.call_model(function_prompt)  # Remove await

        # Structure type classification (no context)
        structure_prompt = self.prompts["sentence_structure_type"]["prompt"].format(sentence=sentence)
        results["structure_type"] = agent.call_model(structure_prompt)

        # Purpose classification (observer context)
        purpose_prompt = self.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=contexts["observer"]
        )
        results["purpose"] = agent.call_model(purpose_prompt)

        # Topic level 1 (immediate context)
        topic_lvl1_prompt = self.prompts["topic_level_1"]["prompt"].format(
            sentence=sentence, context=contexts["immediate"]
        )
        results["topic_level_1"] = agent.call_model(topic_lvl1_prompt)

        # Topic level 3 (broader context)
        topic_lvl3_prompt = self.prompts["topic_level_3"]["prompt"].format(
            sentence=sentence, context=contexts["broader"]
        )
        results["topic_level_3"] = agent.call_model(topic_lvl3_prompt)  # Corrected variable

        # Overall keywords (overall context)
        overall_keywords_prompt = self.prompts["topic_overall_keywords"]["prompt"].format(
            context=contexts["observer"]
        )
        results["overall_keywords"] = agent.call_model(overall_keywords_prompt)

        # Domain-specific keywords
        domain_keywords = ", ".join(config.get("domain_keywords", []))
        domain_prompt = self.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=domain_keywords
        )
        results["domain_keywords"] = agent.call_model(domain_prompt)

        logger.info(f"Sentence analyzed: {sentence[:50]}...")

        return results

    async def analyze_sentences(self, sentences: list) -> Generator[Dict[str, Any], None, None]:
        contexts = context_builder.build_all_contexts(sentences)

        for idx, sentence in enumerate(sentences):
            result = await self.classify_sentence(sentence, contexts[idx])
            result.update({"sentence_id": idx, "sentence": sentence})
            logger.debug(f"Completed analysis for sentence ID {idx}")

            # Removed redundant assignments and updates
            logger.debug(f"Finalized result: {result}")
            yield result


# Singleton instance for pipeline use
sentence_analyzer = SentenceAnalyzer()
