"""Layer 4 resolution smoke test (integration).

Ingests two small transcripts sharing a front-matter participant, emits
overlapping entity surfaces through the Fragment aggregate (Layer 2's canned
pattern), replays everything through the real handler registry, runs
ResolutionEngine with a fake embedder, replays the resulting Project-stream
events through the same registry, and asserts the canonical-entity / person
overlay materializes in Neo4j. Also closes the M4.5a "dual-label invariant"
backlog item. Requires `make test-infra-up`.
"""

import uuid as uuid_mod

import pytest

from src.events.project_events import project_aggregate_id
from src.events.repository import get_repository_factory
from src.events.store import StreamNotFoundError
from src.projections.bootstrap import create_handler_registry
from src.resolution.engine import ResolutionEngine
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

TRANSCRIPT_1 = """---
participants: [Jane Doe]
---
Jane: We work closely with Acme Corp on the rollout.
Bob: Good, the Acme Corp team has been responsive.
"""

TRANSCRIPT_2 = """---
participants: [Jane Doe]
---
Jane: Let's follow up with the Acme Corp folks again next week.
Alice: Sounds good to me.
"""


class _FakeEmbedder:
    """Deterministic orthogonal vectors — never auto-unions across surfaces.

    The smoke's canonical merge comes entirely from the exact-normalization
    group ("acme corp" / "the acme corp"), so embedding numerics must stay
    inert: every distinct normalized surface gets its own orthogonal unit
    vector (cosine 0 against every other key).
    """

    async def embed(self, texts):
        vectors = []
        for i, _ in enumerate(texts):
            vec = [0.0] * len(texts)
            vec[i] = 1.0
            vectors.append(vec)
        return vectors


async def _replay_all(project_id, interview_ids, registry):
    """Replay Interview-, Sentence-, and Project-stream events in commit
    order through the real handler registry (mirrors tests/integration/
    test_layer3_lens_smoke.py's convention — no live subscription consumer
    in this environment)."""
    factory = get_repository_factory()
    interview_repo = factory.create_interview_repository()
    sentence_repo = factory.create_sentence_repository()
    project_repo = factory.create_project_repository()

    events = []
    for interview_id, fragment_count in interview_ids:
        events.extend(await interview_repo.event_store.read_stream(f"Interview-{interview_id}"))
        for index in range(fragment_count):
            sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:{index}"))
            events.extend(await sentence_repo.event_store.read_stream(f"Sentence-{sid}"))

    project_stream = f"Project-{project_aggregate_id(project_id)}"
    try:
        events.extend(await project_repo.event_store.read_stream(project_stream))
    except StreamNotFoundError:
        pass  # ResolutionEngine hasn't run yet on this replay pass

    events.sort(key=lambda e: e.occurred_at)
    for event in events:
        handler = registry.get_handler(event.event_type)
        if handler:
            await handler.handle(event)


async def _emit_entities(interview_id, fragment_index, entities):
    """Emit EntitiesExtracted for one fragment via the Fragment aggregate +
    repository (Layer 2 smoke's canned pattern)."""
    from src.events.repository import get_fragment_repository

    fragment_repo = get_fragment_repository()
    sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:{fragment_index}"))
    fragment = await fragment_repo.load(sid)
    fragment.record_entities(entities, model="smoke-model", provider="smoke")
    await fragment_repo.save(fragment)


@pytest.mark.asyncio
async def test_resolution_engine_canonicalizes_entities_and_links_persons(tmp_path, monkeypatch):
    from src.ingestion.orchestrator import IngestionOrchestrator

    project_id = f"smoke-proj-{uuid_mod.uuid4()}"

    file1 = tmp_path / "smoke_resolution_1.txt"
    file1.write_text(TRANSCRIPT_1)
    file2 = tmp_path / "smoke_resolution_2.txt"
    file2.write_text(TRANSCRIPT_2)

    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    result1 = await ingest.ingest_file(file1)
    result2 = await ingest.ingest_file(file2)

    # Overlapping surfaces: "Acme Corp" / "the Acme Corp" normalize to the
    # same key ("acme corp") -> exact-group auto-merge, no embedding needed.
    await _emit_entities(
        result1.interview_id, 0,
        [{"text": "Acme Corp", "entity_type": "ORG", "start": 20, "end": 29, "confidence": 0.9}],
    )
    await _emit_entities(
        result1.interview_id, 1,
        [{"text": "Acme Corp", "entity_type": "ORG", "start": 4, "end": 13, "confidence": 0.9}],
    )
    await _emit_entities(
        result2.interview_id, 0,
        [{"text": "the Acme Corp", "entity_type": "ORG", "start": 15, "end": 28, "confidence": 0.9}],
    )

    registry = create_handler_registry()
    interview_ids = [
        (result1.interview_id, result1.fragment_count),
        (result2.interview_id, result2.fragment_count),
    ]
    await _replay_all(project_id, interview_ids, registry)

    monkeypatch.setattr(ResolutionEngine, "_build_embedder", lambda self: _FakeEmbedder())
    engine_result = await ResolutionEngine().apply(project_id)
    assert engine_result.entities_canonicalized == 1
    assert engine_result.persons_identified == 1
    assert engine_result.speakers_linked == 2

    await _replay_all(project_id, interview_ids, registry)

    async with await Neo4jConnectionManager.get_session() as session:
        res = await session.run(
            """
            MATCH (c:CanonicalEntity {entity_type: 'ORG'})<-[:ALIAS_OF]-(e:Entity)
            WHERE c.project_id = $project_id
            WITH c, count(DISTINCT e) AS aliases
            MATCH (p:Person {display_name: 'Jane Doe'})<-[r:IDENTIFIED_AS]-(sp:Speaker)
            WHERE p.project_id = $project_id
            RETURN c.canonical_id AS canonical_id, aliases,
                   count(DISTINCT sp) AS speakers, collect(DISTINCT r.method) AS methods
            """,
            project_id=project_id,
        )
        record = await res.single()
        assert record is not None, "resolution overlay did not materialize"
        assert record["aliases"] == 2
        assert record["speakers"] == 2
        assert record["methods"] == ["front_matter"]

        # Dual-label invariant (M4.5a backlog item — closed by this smoke).
        dual_label = await session.run(
            """
            MATCH (n) WHERE (n:Sentence AND NOT n:Fragment) OR (n:Fragment AND NOT n:Sentence)
            RETURN count(n) AS mismatched
            """
        )
        dual_label_record = await dual_label.single()
        assert dual_label_record["mismatched"] == 0

    # Idempotent re-run: second pass over the same (already-resolved) project
    # state emits no new events and writes nothing new to the graph.
    second_result = await ResolutionEngine().apply(project_id)
    assert second_result.entities_canonicalized == 0
    assert second_result.aliases_added == 0
    assert second_result.persons_identified == 0
    assert second_result.speakers_linked == 0

    await _replay_all(project_id, interview_ids, registry)

    async with await Neo4jConnectionManager.get_session() as session:
        res = await session.run(
            """
            MATCH (c:CanonicalEntity {entity_type: 'ORG'})
            WHERE c.project_id = $project_id
            WITH count(c) AS canonicals
            MATCH (e:Entity)-[a:ALIAS_OF]->(:CanonicalEntity {project_id: $project_id})
            WITH canonicals, count(a) AS aliases
            MATCH (p:Person {project_id: $project_id})
            WITH canonicals, aliases, count(p) AS persons
            MATCH (:Speaker)-[r:IDENTIFIED_AS]->(:Person {project_id: $project_id})
            RETURN canonicals, aliases, persons, count(r) AS links
            """,
            project_id=project_id,
        )
        record = await res.single()
        assert record["canonicals"] == 1
        assert record["aliases"] == 2
        assert record["persons"] == 1
        assert record["links"] == 2
