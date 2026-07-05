"""Runs registered extractors over fragments and utterances.

One focused LLM call per dimension per unit (never merged), schema-enforced,
Pydantic-validated, concurrency-bounded. Invalid responses degrade to an
omitted dimension plus a review flag; they never fail the run.
"""

import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from src.agents.failover_agent import FailoverAgent
from src.enrichment.graph_context import FragmentView
from src.enrichment.models import ExtractorSpec
from src.enrichment.syntax_check import syntax_flags
from src.utils.logger import get_logger

logger = get_logger()

# Extractors whose result is a single classification value + confidence.
_CLASSIFICATION_KEYS = {
    "function_type": "function_type",
    "structure_type": "structure_type",
    "purpose": "purpose",
    "topic_level_1": "topic_level_1",
    "topic_level_3": "topic_level_3",
}


class FragmentEnrichment(BaseModel):
    index: int
    classification: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    domain_keywords: List[str] = Field(default_factory=list)
    dimension_confidences: Dict[str, float] = Field(default_factory=dict)
    flags: Dict[str, str] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str = ""
    model: str = ""


class UtteranceEnrichment(BaseModel):
    utterance_id: str
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str = ""
    model: str = ""


class EnrichmentExecutor:
    def __init__(
        self,
        agent: FailoverAgent,
        specs: List[ExtractorSpec],
        prompts: Dict[str, Any],
        domain_keywords: List[str],
        concurrency: int = 10,
    ):
        self.agent = agent
        self.fragment_specs = [s for s in specs if s.scope == "fragment"]
        self.utterance_specs = [s for s in specs if s.scope == "utterance"]
        self.prompts = prompts
        self.domain_keywords = domain_keywords
        self.semaphore = asyncio.Semaphore(concurrency)

    def _format_prompt(self, spec: ExtractorSpec, text: str, context: Dict[str, str]) -> str:
        template = self.prompts[spec.prompt_key]["prompt"]
        kwargs: Dict[str, str] = {"sentence": text}
        if spec.context_needs:
            kwargs["context"] = context.get(spec.context_needs[0], "")
        if spec.name == "domain_keywords":
            kwargs["domain_keywords"] = ", ".join(self.domain_keywords)
        return template.format(**kwargs)

    async def _run_spec(self, spec: ExtractorSpec, text: str, context: Dict[str, str]):
        async with self.semaphore:
            prompt = self._format_prompt(spec, text, context)
            schema = spec.resolve_model().model_json_schema()
            return await self.agent.call(prompt, schema=schema)

    async def enrich_fragments(
        self, fragments: List[FragmentView], contexts: List[Dict[str, str]]
    ) -> List[FragmentEnrichment]:
        context_by_index = {f.index: contexts[i] for i, f in enumerate(fragments)}
        return list(
            await asyncio.gather(
                *(self._enrich_fragment(f, context_by_index[f.index]) for f in fragments)
            )
        )

    async def _enrich_fragment(
        self, frag: FragmentView, context: Dict[str, str]
    ) -> FragmentEnrichment:
        out = FragmentEnrichment(index=frag.index)
        results = await asyncio.gather(
            *(self._run_spec(spec, frag.text, context) for spec in self.fragment_specs)
        )
        for spec, call_result in zip(self.fragment_specs, results):
            out.provider, out.model = call_result.provider, call_result.model
            try:
                parsed = spec.resolve_model().model_validate(call_result.data)
            except ValidationError as e:
                logger.warning(f"{spec.name}: invalid response ({e.error_count()} errors)")
                out.flags[f"{spec.name}_invalid_response"] = "validation failed"
                continue
            data = parsed.model_dump()
            if spec.name in _CLASSIFICATION_KEYS:
                out.classification[spec.name] = data[_CLASSIFICATION_KEYS[spec.name]]
                out.dimension_confidences[spec.name] = data["confidence"]
            elif spec.name == "overall_keywords":
                out.keywords = data["overall_keywords"]
            elif spec.name == "domain_keywords":
                out.domain_keywords = data["domain_keywords"]
            elif spec.name == "entity_mentions":
                out.entities = data["entities"]
        out.flags.update(
            syntax_flags(
                frag.text,
                out.classification.get("function_type", ""),
                out.classification.get("structure_type", ""),
            )
        )
        return out

    async def enrich_utterances(
        self, utterance_texts: Dict[str, str]
    ) -> List[UtteranceEnrichment]:
        async def one(uid: str, text: str) -> UtteranceEnrichment:
            out = UtteranceEnrichment(utterance_id=uid)
            for spec in self.utterance_specs:
                call_result = await self._run_spec(spec, text, {})
                out.provider, out.model = call_result.provider, call_result.model
                try:
                    parsed = spec.resolve_model().model_validate(call_result.data)
                except ValidationError:
                    logger.warning(f"{spec.name}: invalid response for utterance {uid}")
                    continue
                out.claims = list(parsed.model_dump()["claims"])
            return out

        return list(
            await asyncio.gather(*(one(uid, text) for uid, text in utterance_texts.items()))
        )
