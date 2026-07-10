"""Regression tests for the M4.2 final whole-branch review findings (C1-C3, I1-I3)."""

import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import FragmentEnrichment, UtteranceEnrichment
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.aggregates import Interview, Sentence

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"


# --- C3: OpenAI strict mode requires additionalProperties: false everywhere ---

def _assert_strict(schema: dict, path: str = "$"):
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, f"{path} not strict"
        # strict mode also requires every property listed in required
        props = set(schema.get("properties", {}))
        assert props <= set(schema.get("required", [])), f"{path} has optional props"
    for name, sub in schema.get("$defs", {}).items():
        _assert_strict(sub, f"{path}.$defs.{name}")
    for name, sub in schema.get("properties", {}).items():
        _assert_strict(sub, f"{path}.properties.{name}")
    if "items" in schema:
        _assert_strict(schema["items"], f"{path}.items")


def test_every_registered_schema_is_openai_strict_compliant():
    from src.enrichment.registry import ExtractorRegistry

    for spec in ExtractorRegistry.load("config/extractors.yaml"):
        _assert_strict(spec.resolve_model().model_json_schema(), spec.name)


# --- C1: every Sentence-stream event must carry the lane-routing key ---

def test_all_sentence_stream_events_carry_interview_id():
    """LaneManager drops Sentence events without data['interview_id']; every
    command method on the Sentence aggregate must include it."""
    from src.events.sentence_events import EditorType

    s = Sentence(str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:0")))
    s.create(interview_id=IID, index=0, text="Original text.")
    s.attribute_speaker(SP1, 0.9, "inference")
    s.reattribute_speaker(SP1)
    s.edit("Edited text.", EditorType.HUMAN)
    s.generate_analysis(model="haiku", model_version="m4.2", classification={"purpose": "Q"})
    s.record_entities(
        [{"text": "x", "entity_type": "tool", "start": 0, "end": 1, "confidence": 0.9}],
        model="haiku", provider="anthropic",
    )
    s.record_embedding(model="m", dim=3, vector_b64="AAAA")
    s.override_analysis({"purpose": "Statement"})

    for event in s.get_uncommitted_events():
        assert event.data.get("interview_id") == IID, (
            f"{event.event_type} lacks interview_id — LaneManager would drop it"
        )


# --- C2 / I1 / I2 world builder ---

def build_world(analyzed_indices=(), with_claims=False):
    interview = Interview(IID)
    interview.create(title="t.txt", source="s", metadata={"fragment_count": 3})
    interview.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(3)]
    sentences = []
    for i, fid in enumerate(f_ids):
        s = Sentence(fid)
        s.create(interview_id=IID, index=i, text=f"Fragment {i}.")
        s.attribute_speaker(SP1, 0.9, "inference")
        if i in analyzed_indices:
            s.generate_analysis(model="haiku", model_version="m4.2", classification={"purpose": "Q"})
        sentences.append(s)
    interview.identify_utterance(U1, SP1, f_ids, 0.9)
    if with_claims:
        claim_id = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:claim:{U1}:0"))
        interview.record_claim(claim_id, U1, "We ship Friday", "commitment", 0.8, "haiku", "anthropic")
        interview.record_utterance_embedding(U1, model="m", dim=3, vector_b64="AAAA")
    interview.mark_events_as_committed()
    for s in sentences:
        s.mark_events_as_committed()
    return interview, {s.aggregate_id: s for s in sentences}


def make_repos(interview, sentences):
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    sentence_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    return interview_repo, sentence_repo


def make_executor(claims=True):
    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(side_effect=lambda views, ctxs: [
        FragmentEnrichment(index=v.index, classification={"purpose": "Statement"},
                           dimension_confidences={"purpose": 0.8},
                           provider="anthropic", model="haiku")
        for v in views
    ])
    executor.enrich_utterances = AsyncMock(return_value=(
        [UtteranceEnrichment(utterance_id=U1,
                             claims=[{"text": "We ship Friday", "kind": "commitment", "confidence": 0.8}],
                             provider="anthropic", model="haiku")]
        if claims else []
    ))
    return executor


# --- C2: double-run must not crash and must not duplicate claims ---

@pytest.mark.asyncio
async def test_second_run_skips_claimed_utterances_no_crash():
    interview, sentences = build_world(analyzed_indices=(0, 1, 2), with_claims=True)
    interview_repo, sentence_repo = make_repos(interview, sentences)
    executor = make_executor()
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        result = await EnrichmentOrchestrator().enrich_interview(IID)  # second run

    assert result.fragments_skipped == 3
    assert result.claims_extracted == 0  # nothing re-extracted
    executor.enrich_utterances.assert_not_awaited()  # no LLM burn
    assert len(interview.claims) == 1  # no duplicates


@pytest.mark.asyncio
async def test_forced_rerun_is_idempotent_on_existing_claim_ids():
    interview, sentences = build_world(analyzed_indices=(0, 1, 2), with_claims=True)
    interview_repo, sentence_repo = make_repos(interview, sentences)
    executor = make_executor()  # returns the same claim -> same deterministic id
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        result = await EnrichmentOrchestrator().enrich_interview(IID, force=True)  # no crash

    assert len(interview.claims) == 1  # idempotent skip, not duplicate or crash
    assert result.claims_extracted == 0


# --- I1: resume-mode contexts must window over ALL fragments ---

@pytest.mark.asyncio
async def test_resume_contexts_include_skipped_neighbors():
    interview, sentences = build_world(analyzed_indices=(0, 2))  # only middle needs work
    interview_repo, sentence_repo = make_repos(interview, sentences)
    executor = make_executor(claims=False)
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        await EnrichmentOrchestrator().enrich_interview(IID)

    views, contexts = executor.enrich_fragments.call_args.args
    assert [v.index for v in views] == [1]
    # The skipped neighbors (fragments 0 and 2) must appear in the window, and
    # the target marker must be present.
    ctx = contexts[0]["immediate_context"]
    assert "Fragment 0." in ctx and "Fragment 2." in ctx
    assert ">>> [S1]: Fragment 1. <<<" in ctx


# --- I2: total provider outage must not seal fragments as analyzed ---

@pytest.mark.asyncio
async def test_total_call_failure_leaves_fragment_unanalyzed():
    interview, sentences = build_world()
    interview_repo, sentence_repo = make_repos(interview, sentences)
    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(side_effect=lambda views, ctxs: [
        FragmentEnrichment(index=v.index, flags={"purpose_call_error": "RuntimeError"})
        for v in views  # model="" -> every call failed
    ])
    executor.enrich_utterances = AsyncMock(return_value=[])
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        await EnrichmentOrchestrator().enrich_interview(IID)

    for s in sentences.values():
        assert s.analysis_model is None  # retried next run, not sealed


# --- I3: utterance-path call errors are isolated, not fatal ---

@pytest.mark.asyncio
async def test_utterance_call_error_flags_not_fatal():
    from src.enrichment.registry import ExtractorRegistry
    from src.enrichment.executor import EnrichmentExecutor
    from src.utils.helpers import load_yaml

    agent = MagicMock()
    agent.call = AsyncMock(side_effect=RuntimeError("chain exhausted"))
    executor = EnrichmentExecutor(
        agent, ExtractorRegistry.load("config/extractors.yaml"),
        load_yaml("prompts/core_extractors.yaml"), domain_keywords=[], concurrency=2,
    )
    results = await executor.enrich_utterances({"u-1": "Some utterance."})
    assert results[0].claims == []
    assert "claims_call_error" in results[0].flags
