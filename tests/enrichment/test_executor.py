from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.failover_agent import CallResult
from src.enrichment.executor import EnrichmentExecutor
from src.enrichment.graph_context import FragmentView
from src.enrichment.registry import ExtractorRegistry
from src.utils.helpers import load_yaml

RESPONSES = {
    "function_type": {"function_type": "interrogative", "confidence": 0.9},
    "structure_type": {"structure_type": "simple", "confidence": 0.8},
    "purpose": {"purpose": "Query", "confidence": 0.85},
    "topic_level_1": {"topic_level_1": "tools", "confidence": 0.7},
    "topic_level_3": {"topic_level_3": "processes", "confidence": 0.6},
    "overall_keywords": {"overall_keywords": ["audio"]},
    "domain_keywords": {"domain_keywords": []},
    "entity_mentions": {"entities": []},
    "claims": {"claims": [{"text": "We will ship", "kind": "commitment", "confidence": 0.8}]},
}

# Each prompt contains a distinguishing phrase; the fake agent routes on it
# without depending on call order.
MARKERS = {
    "function_type": "function type of the sentence",
    "structure_type": "structure type of the given sentence",
    "purpose": "sentence's purpose",
    "topic_level_1": "based on its immediate surrounding",
    "topic_level_3": "considering a broader context",
    "overall_keywords": "main topics of conversation",
    "domain_keywords": "domain-specific list",
    "entity_mentions": "entity mentions",
    "claims": "complete utterance",
}


def make_agent(responses=None):
    responses = responses or RESPONSES
    agent = MagicMock()

    async def call(prompt, schema=None):
        for key, marker in MARKERS.items():
            if marker in prompt:
                return CallResult(data=responses[key], provider="anthropic", model="haiku")
        raise AssertionError(f"No canned response for prompt: {prompt[:80]}")

    agent.call = AsyncMock(side_effect=call)
    return agent


def make_executor(agent):
    specs = ExtractorRegistry.load("config/extractors.yaml")
    prompts = load_yaml("prompts/core_extractors.yaml")
    return EnrichmentExecutor(agent, specs, prompts, domain_keywords=["ECU"], concurrency=4)


CONTEXT = {
    "immediate_context": "c",
    "observer_context": "c",
    "broader_context": "c",
    "overall_context": "c",
    "utterance_context": "Can you hear me?",
}


@pytest.mark.asyncio
async def test_enrich_fragment_collects_all_dimensions():
    executor = make_executor(make_agent())
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    r = results[0]
    assert r.classification["purpose"] == "Query"
    assert r.classification["function_type"] == "interrogative"
    assert r.dimension_confidences["purpose"] == 0.85
    assert r.keywords == ["audio"]
    assert r.provider == "anthropic"
    # spaCy agrees interrogative/simple here -> no syntax flags
    assert "function_type_disagreement" not in r.flags


@pytest.mark.asyncio
async def test_invalid_response_flagged_not_fatal():
    bad = dict(RESPONSES)
    bad["purpose"] = {"purpose": "Query", "confidence": 5.0}  # out of range
    executor = make_executor(make_agent(bad))
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    assert "purpose" not in results[0].classification
    assert "purpose_invalid_response" in results[0].flags


@pytest.mark.asyncio
async def test_entity_extractor_populates_entities():
    resp = dict(RESPONSES)
    resp["entity_mentions"] = {
        "entities": [{"text": "ECU", "entity_type": "product", "start": 0, "end": 3, "confidence": 0.9}]
    }
    executor = make_executor(make_agent(resp))
    fragments = [FragmentView(index=0, text="ECU owners build files.", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    assert results[0].entities[0]["entity_type"] == "product"


@pytest.mark.asyncio
async def test_enrich_utterances_extracts_claims():
    executor = make_executor(make_agent())
    results = await executor.enrich_utterances(
        {"u-1": "We will ship Friday. That is the complete utterance."}
    )
    assert results[0].utterance_id == "u-1"
    assert results[0].claims[0]["kind"] == "commitment"
    assert results[0].provider == "anthropic"


@pytest.mark.asyncio
async def test_domain_keywords_receives_configured_list():
    executor = make_executor(make_agent())
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    await executor.enrich_fragments(fragments, [CONTEXT])
    # The domain_keywords prompt must have been formatted with the configured list.
    calls = [c.args[0] for c in executor.agent.call.call_args_list]
    domain_prompt = next(p for p in calls if "domain-specific list" in p)
    assert "ECU" in domain_prompt


@pytest.mark.asyncio
async def test_invalid_utterance_response_flagged():
    bad = dict(RESPONSES)
    bad["claims"] = {"claims": [{"text": "x", "kind": "not_a_kind", "confidence": 0.5}]}  # invalid kind
    executor = make_executor(make_agent(bad))
    results = await executor.enrich_utterances({"u-1": "The complete utterance here."})
    assert results[0].claims == []
    assert "claims_invalid_response" in results[0].flags


@pytest.mark.asyncio
async def test_mixed_providers_flagged():
    calls = {"n": 0}

    agent = MagicMock()

    async def call(prompt, schema=None):
        for key, marker in MARKERS.items():
            if marker in prompt:
                calls["n"] += 1
                provider = "anthropic" if calls["n"] % 2 else "openai"
                return CallResult(data=RESPONSES[key], provider=provider, model="m")
        raise AssertionError

    agent.call = AsyncMock(side_effect=call)
    executor = make_executor(agent)
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    assert "mixed_providers" in results[0].flags


@pytest.mark.asyncio
async def test_provider_call_error_flags_one_dimension_not_fatal():
    from src.agents.failover_agent import CallResult

    agent = MagicMock()

    async def call(prompt, schema=None):
        if "sentence's purpose" in prompt:
            raise RuntimeError("provider exhausted")
        for key, marker in MARKERS.items():
            if marker in prompt:
                return CallResult(data=RESPONSES[key], provider="anthropic", model="haiku")
        raise AssertionError

    agent.call = AsyncMock(side_effect=call)
    executor = make_executor(agent)
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    assert "purpose" not in results[0].classification
    assert "purpose_call_error" in results[0].flags
    assert results[0].classification["function_type"] == "interrogative"  # others survived


@pytest.mark.asyncio
async def test_run_spec_on_text_returns_validated_outcome():
    from src.enrichment.models import ExtractorSpec

    executor = make_executor(make_agent())
    spec = ExtractorSpec(
        name="purpose",
        prompt_key="purpose",
        response_model="PurposeResult",
        context_needs=["observer_context"],
        scope="fragment",
    )
    outcome = await executor.run_spec_on_text(spec, "Can you hear me?", {"observer_context": "c"})
    assert outcome.data == {"purpose": "Query", "confidence": 0.85}
    assert outcome.provider == "anthropic"
    assert outcome.flags == {}


@pytest.mark.asyncio
async def test_run_spec_on_text_flags_call_error():
    from src.enrichment.models import ExtractorSpec

    agent = MagicMock()
    agent.call = AsyncMock(side_effect=RuntimeError("down"))
    executor = make_executor(agent)
    spec = ExtractorSpec(
        name="purpose",
        prompt_key="purpose",
        response_model="PurposeResult",
        context_needs=["observer_context"],
        scope="fragment",
    )
    outcome = await executor.run_spec_on_text(spec, "text")
    assert outcome.data is None
    assert outcome.flags == {"purpose_call_error": "RuntimeError"}


def test_document_scope_specs_split():
    from src.enrichment.models import ExtractorSpec

    specs = ExtractorRegistry.load("config/extractors.yaml")
    specs.append(
        ExtractorSpec(
            name="objectives", prompt_key="purpose", response_model="PurposeResult", scope="document"
        )
    )
    executor = EnrichmentExecutor(
        MagicMock(), specs, load_yaml("prompts/core_extractors.yaml"), domain_keywords=[], concurrency=2
    )
    assert [s.name for s in executor.document_specs] == ["objectives"]
