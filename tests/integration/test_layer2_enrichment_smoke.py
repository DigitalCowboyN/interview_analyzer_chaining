"""Layer 2 enrichment smoke test (integration).

Ingests a small labeled transcript, runs enrichment with the executor and
embedder mocked (deterministic, no live LLM), replays every event through the
real handler registry in commit order against real Neo4j, and asserts the
Entity/Claim/embedding subgraph materializes. Requires `make test-infra-up`.
"""

import uuid as uuid_mod

import pytest

from src.enrichment.executor import FragmentEnrichment, UtteranceEnrichment
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.repository import get_repository_factory
from src.projections.bootstrap import create_handler_registry
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

LABELED = """Alice: We will ship the ECU firmware on Friday.
Bob: Sounds good to me.
"""


@pytest.mark.asyncio
async def test_enriched_interview_projects_entity_claim_embedding_subgraph(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.ingestion.orchestrator import IngestionOrchestrator

    input_file = tmp_path / "smoke_enrich.txt"
    input_file.write_text(LABELED)

    project_id = f"smoke-{uuid_mod.uuid4()}"
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    # Mock the executor (no live LLM) and embedder (fixed 3-dim vectors).
    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(
            index=i,
            classification={"purpose": "Statement"},
            dimension_confidences={"purpose": 0.9},
            entities=(
                [{"text": "ECU", "entity_type": "product", "start": 0, "end": 3, "confidence": 0.9}]
                if i == 0 else []
            ),
            provider="anthropic", model="haiku",
        )
        for i in range(ingest_result.fragment_count)
    ])
    executor.enrich_utterances = AsyncMock(return_value=[
        UtteranceEnrichment(
            utterance_id="placeholder",  # replaced below to the real utterance id
            claims=[{"text": "We will ship Friday", "kind": "commitment", "confidence": 0.9}],
            provider="anthropic", model="haiku",
        )
    ])
    embedder = MagicMock(model_name="smoke-embed", dim=3)
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    # Resolve the real utterance id from the interview aggregate.
    factory = get_repository_factory()
    interview = await factory.create_interview_repository().load(interview_id)
    real_uid = next(uid for uid, u in interview.utterances.items() if not u.get("removed"))
    executor.enrich_utterances.return_value[0].utterance_id = real_uid

    monkeypatch.setattr(EnrichmentOrchestrator, "_build_executor", lambda self: executor)
    monkeypatch.setattr("src.enrichment.orchestrator.get_embedder", lambda cfg=None: embedder)

    enrich_result = await EnrichmentOrchestrator().enrich_interview(interview_id)
    assert enrich_result.entities_extracted == 1
    assert enrich_result.claims_extracted == 1

    # Replay all events in commit order through the real registry.
    interview_repo = factory.create_interview_repository()
    sentence_repo = factory.create_sentence_repository()
    registry = create_handler_registry()
    events = list(await interview_repo.event_store.read_stream(f"Interview-{interview_id}"))
    for index in range(ingest_result.fragment_count):
        sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:{index}"))
        events.extend(await sentence_repo.event_store.read_stream(f"Sentence-{sid}"))
    events.sort(key=lambda e: e.occurred_at)
    for event in events:
        handler = registry.get_handler(event.event_type)
        if handler:
            await handler.handle(event)

    async with await Neo4jConnectionManager.get_session() as session:
        res = await session.run(
            """
            MATCH (s:Sentence {aggregate_id: $f0})-[:MENTIONS]->(e:Entity)
            WITH count(e) AS entity_count
            MATCH (c:Claim)-[:MADE_BY]->(:Speaker)
            WITH entity_count, count(DISTINCT c) AS claim_count
            MATCH (i:Interview {interview_id: $iid})-[:HAS_SENTENCE]->(s2:Sentence)
            WHERE s2.embedding IS NOT NULL AND s2.embedding_model = 'smoke-embed'
            RETURN entity_count, claim_count, count(DISTINCT s2) AS embedded
            """,
            f0=str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:0")),
            iid=interview_id,
        )
        record = await res.single()
        assert record["entity_count"] >= 1
        assert record["claim_count"] >= 1
        assert record["embedded"] == ingest_result.fragment_count
