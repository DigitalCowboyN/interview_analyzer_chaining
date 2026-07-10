"""Runs registered extractors over fragments and utterances.

One focused LLM call per dimension per unit (never merged), schema-enforced,
Pydantic-validated, concurrency-bounded. Invalid responses degrade to an
omitted dimension plus a review flag; they never fail the run.
"""

import asyncio
from typing import Any, Dict, List, Optional

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


class SpecOutcome(BaseModel):
    """Result of one focused spec call: validated data or a flag, never a raise."""

    data: Optional[Dict[str, Any]] = None
    flags: Dict[str, str] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""


class UtteranceEnrichment(BaseModel):
    utterance_id: str
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    flags: Dict[str, str] = Field(default_factory=dict)
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
        self.document_specs = [s for s in specs if s.scope == "document"]
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

    async def run_spec_on_text(
        self, spec: ExtractorSpec, text: str, context: Optional[Dict[str, str]] = None
    ) -> SpecOutcome:
        """One focused, schema-enforced call for one spec over one text unit.

        Call errors and invalid responses degrade to flags on the outcome
        (`<name>_call_error` / `<name>_invalid_response`); this never raises.
        """
        outcome = SpecOutcome()
        try:
            call_result = await self._run_spec(spec, text, context or {})
        except Exception as exc:
            logger.warning(f"{spec.name}: call failed ({type(exc).__name__})")
            outcome.flags[f"{spec.name}_call_error"] = type(exc).__name__
            return outcome
        outcome.provider, outcome.model = call_result.provider, call_result.model
        try:
            parsed = spec.resolve_model().model_validate(call_result.data)
        except ValidationError as e:
            logger.warning(f"{spec.name}: invalid response ({e.error_count()} errors)")
            outcome.flags[f"{spec.name}_invalid_response"] = "validation failed"
            return outcome
        outcome.data = parsed.model_dump()
        return outcome

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
        # out.provider reflects the LAST successful call's provider; when calls
        # span multiple providers, flags["mixed_providers"] lists them all.
        out = FragmentEnrichment(index=frag.index)
        # run_spec_on_text degrades errors to flags, so one provider failure
        # loses that dimension, never the whole fragment.
        outcomes = await asyncio.gather(
            *(self.run_spec_on_text(spec, frag.text, context) for spec in self.fragment_specs)
        )
        providers_seen: set = set()
        for spec, outcome in zip(self.fragment_specs, outcomes):
            out.flags.update(outcome.flags)
            if outcome.provider:
                out.provider, out.model = outcome.provider, outcome.model
                providers_seen.add(outcome.provider)
            if outcome.data is None:
                continue
            data = outcome.data
            if spec.name in _CLASSIFICATION_KEYS:
                out.classification[spec.name] = data[_CLASSIFICATION_KEYS[spec.name]]
                out.dimension_confidences[spec.name] = data["confidence"]
            elif spec.name == "overall_keywords":
                out.keywords = data["overall_keywords"]
            elif spec.name == "domain_keywords":
                out.domain_keywords = data["domain_keywords"]
            elif spec.name == "entity_mentions":
                out.entities = data["entities"]
        if len(providers_seen) > 1:
            out.flags["mixed_providers"] = ",".join(sorted(providers_seen))
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
                # Same isolation policy as the fragment path: one exhausted
                # chain flags this utterance's dimension, never aborts the run.
                outcome = await self.run_spec_on_text(spec, text, {})
                out.flags.update(outcome.flags)
                if outcome.provider:
                    out.provider, out.model = outcome.provider, outcome.model
                if outcome.data is not None:
                    out.claims.extend(outcome.data["claims"])
            return out

        return list(
            await asyncio.gather(*(one(uid, text) for uid, text in utterance_texts.items()))
        )
