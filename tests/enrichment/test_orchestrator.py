import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import FragmentEnrichment, UtteranceEnrichment
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.aggregates import Interview, Fragment

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"


def build_world(analyzed_indices=()):
    interview = Interview(IID)
    interview.create(title="t.txt", source="s", metadata={"fragment_count": 2})
    interview.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(2)]
    sentences = []
    for i, fid in enumerate(f_ids):
        s = Fragment(fid)
        s.create(interview_id=IID, index=i, text=f"Fragment {i}.")
        s.attribute_speaker(SP1, 0.9, "inference")
        if i in analyzed_indices:
            s.generate_analysis(model="haiku", model_version="m4.2", classification={"purpose": "Q"})
        sentences.append(s)
    interview.identify_utterance(U1, SP1, f_ids, 0.9)
    interview.mark_events_as_committed()
    for s in sentences:
        s.mark_events_as_committed()
    return interview, {s.aggregate_id: s for s in sentences}


def make_repos(interview, sentences):
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    fragment_repo = MagicMock()
    fragment_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    fragment_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    return interview_repo, fragment_repo


@pytest.mark.asyncio
async def test_enrich_interview_emits_analysis_entities_claims_embeddings():
    interview, sentences = build_world()
    interview_repo, fragment_repo = make_repos(interview, sentences)

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(index=0, classification={"purpose": "Query"},
                           dimension_confidences={"purpose": 0.9},
                           entities=[{"text": "ECU", "entity_type": "product", "start": 0, "end": 3, "confidence": 0.9}],
                           provider="anthropic", model="haiku"),
        FragmentEnrichment(index=1, classification={"purpose": "Statement"},
                           dimension_confidences={"purpose": 0.8}, provider="anthropic", model="haiku"),
    ])
    executor.enrich_utterances = AsyncMock(return_value=[
        UtteranceEnrichment(utterance_id=U1,
                            claims=[{"text": "We ship Friday", "kind": "commitment", "confidence": 0.8}],
                            provider="anthropic", model="haiku"),
    ])
    embedder = MagicMock(model_name="text-embedding-3-small", dim=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_fragment_repository", return_value=fragment_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID)

    assert result.fragments_enriched == 2
    assert result.entities_extracted == 1
    assert result.claims_extracted == 1
    assert result.embeddings_generated == 3  # 2 fragments + 1 utterance
    first = sentences[list(sentences)[0]]
    assert first.classification["purpose"] == "Query"
    assert first.embedding_model == "text-embedding-3-small"
    assert len(interview.claims) == 1


@pytest.mark.asyncio
async def test_resume_skips_already_analyzed():
    interview, sentences = build_world(analyzed_indices=(0,))
    interview_repo, fragment_repo = make_repos(interview, sentences)

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(index=1, classification={"purpose": "Statement"},
                           dimension_confidences={"purpose": 0.8}, provider="anthropic", model="haiku"),
    ])
    executor.enrich_utterances = AsyncMock(return_value=[])
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_fragment_repository", return_value=fragment_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID)

    assert result.fragments_skipped == 1
    assert result.fragments_enriched == 1
    passed = executor.enrich_fragments.call_args.args[0]
    assert [f.index for f in passed] == [1]


@pytest.mark.asyncio
async def test_force_reenriches_all():
    interview, sentences = build_world(analyzed_indices=(0, 1))
    interview_repo, fragment_repo = make_repos(interview, sentences)

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(index=0, classification={"purpose": "Query"}, provider="a", model="m"),
        FragmentEnrichment(index=1, classification={"purpose": "Statement"}, provider="a", model="m"),
    ])
    executor.enrich_utterances = AsyncMock(return_value=[])
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_fragment_repository", return_value=fragment_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID, force=True)

    assert result.fragments_skipped == 0
    assert result.fragments_enriched == 2


@pytest.mark.asyncio
async def test_missing_interview_raises():
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=None)
    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo):
        orchestrator = EnrichmentOrchestrator()
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.enrich_interview(IID)
