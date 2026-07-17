"""End-to-end smoke (integration): ingest → enrich+segments → lens → resolve → export.

Canned LLM outcomes and a fake embedder throughout; events replayed in commit
order through the real handler registry against real Neo4j (there is no live
subscription consumer in the test environment). Requires `make test-infra-up`.
"""

import uuid as uuid_mod

import pytest

from src.enrichment.executor import (
    FragmentEnrichment,
    SpecOutcome,
    UtteranceEnrichment,
)
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.enrichment.registry import ExtractorRegistry
from src.events.project_events import project_aggregate_id
from src.events.repository import get_repository_factory
from src.events.store import StreamNotFoundError
from src.export.bundler import OkfExporter
from src.lens.engine import LensEngine
from src.projections.bootstrap import create_handler_registry
from src.resolution.engine import ResolutionEngine
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

LABELED = """---
title: Vendor Kickoff
project: telemetry
date: 2026-07-01
participants: [Jane Doe]
---
Jane: We will go with Acme Corp and I'll draft the doc by Friday.
Bob: Sounds good to me.
Jane: The timeline is six weeks starting Monday.
Bob: That works for the schedule.
"""

SEGMENTS_PROPOSAL = {
    "segments": [
        {"topic": "Vendor choice", "start_index": 0, "end_index": 1, "confidence": 0.9},
        {"topic": "Timeline", "start_index": 2, "end_index": 3, "confidence": 0.8},
    ]
}

LENS_CANNED = {
    "objectives": {"objectives": [{"text": "Choose a vendor", "confidence": 0.9}]},
    "decisions": {
        "decisions": [{"text": "Go with Acme Corp", "made_by": "Jane", "confidence": 0.9}]
    },
    "action_items": {
        "action_items": [
            {"text": "Draft the doc", "owner": "SELF", "due": "Friday", "confidence": 0.8}
        ]
    },
    "followups": {"followups": []},
}

LENS_EMPTY = {
    "objectives": {"objectives": []},
    "decisions": {"decisions": []},
    "action_items": {"action_items": []},
    "followups": {"followups": []},
}


def lens_outcome(spec, text):
    """Only Jane's first utterance (and the document) yields lens items."""
    source = LENS_CANNED if "Acme Corp" in text else LENS_EMPTY
    return SpecOutcome(data=source[spec.name], provider="anthropic", model="haiku")


class _FakeEmbedder:
    """Deterministic orthogonal vectors — never auto-unions across surfaces.

    The canned entities use exactly ONE surface ("Acme Corp"), so resolution
    canonicalizes a single-surface cluster without any embedding pass; the
    fake keeps numerics inert if one ever happens (cosine 0 between keys).
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
    test_layer4_resolution_smoke.py's convention — no live subscription
    consumer in this environment)."""
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


async def _segment_shape(session, interview_id):
    """Exact segment topology: per-topic CONTAINS orders + total edge count."""
    res = await session.run(
        """
        MATCH (seg:Segment {interview_id: $iid})
        OPTIONAL MATCH (seg)-[:CONTAINS]->(f:Fragment)
        RETURN seg.topic AS topic, count(f) AS contained,
               collect(f.sequence_order) AS orders
        """,
        iid=interview_id,
    )
    records = [r async for r in res]
    return {r["topic"]: (r["contained"], sorted(r["orders"])) for r in records}


@pytest.mark.asyncio
async def test_full_pipeline_ingest_enrich_segments_lens_resolve_export(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.ingestion.orchestrator import IngestionOrchestrator

    # --- 1. Ingest one labeled, front-mattered transcript -------------------
    input_file = tmp_path / "smoke_e2e.txt"
    input_file.write_text(LABELED)

    project_id = f"smoke-e2e-{uuid_mod.uuid4()}"
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id
    assert ingest_result.fragment_count == 4  # segment ranges below rely on 0..3

    # --- 2. Enrichment: canned fragments/utterances/segments, fake embedder -
    factory = get_repository_factory()
    interview_repo = factory.create_interview_repository()
    interview = await interview_repo.load(interview_id)
    f0 = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:0"))
    real_uid = next(
        uid for uid, u in interview.utterances.items()
        if not u.get("removed") and f0 in u["fragment_ids"]
    )

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(
            index=i,
            classification={"purpose": "Statement"},
            dimension_confidences={"purpose": 0.9},
            entities=(
                # ONE surface only: resolution canonicalizes a single-surface cluster.
                [{"text": "Acme Corp", "entity_type": "organization",
                  "start": 16, "end": 25, "confidence": 0.9}]
                if i == 0 else []
            ),
            provider="anthropic", model="haiku",
        )
        for i in range(ingest_result.fragment_count)
    ])
    executor.enrich_utterances = AsyncMock(return_value=[
        UtteranceEnrichment(
            utterance_id=real_uid,
            claims=[{"text": "Draft the doc by Friday", "kind": "commitment", "confidence": 0.9}],
            provider="anthropic", model="haiku",
        )
    ])
    executor.document_specs = [
        s for s in ExtractorRegistry.load("config/extractors.yaml") if s.scope == "document"
    ]
    executor.run_spec_on_text = AsyncMock(
        return_value=SpecOutcome(data=SEGMENTS_PROPOSAL, provider="anthropic", model="haiku")
    )
    embedder = MagicMock(model_name="smoke-embed", dim=3)
    # values unused by assertions — only counts/labels are asserted
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    monkeypatch.setattr(EnrichmentOrchestrator, "_build_executor", lambda self: executor)
    monkeypatch.setattr("src.enrichment.orchestrator.get_embedder", lambda cfg=None: embedder)

    enrich_result = await EnrichmentOrchestrator().enrich_interview(interview_id)
    assert enrich_result.fragments_enriched == 4
    assert enrich_result.entities_extracted == 1
    assert enrich_result.claims_extracted == 1
    assert enrich_result.embeddings_generated == 8  # 4 fragments + 4 utterances
    assert enrich_result.segments_extracted == 2
    assert enrich_result.flags == {}

    # --- 3. Lens: canned executor, meeting_minutes --------------------------
    lens_executor = MagicMock()
    lens_executor.run_spec_on_text = AsyncMock(
        side_effect=lambda spec, text, ctx=None: lens_outcome(spec, text)
    )
    monkeypatch.setattr(LensEngine, "_build_executor", lambda self, lens: lens_executor)

    lens_result = await LensEngine().apply(interview_id, "meeting_minutes")
    assert lens_result.items_extracted == 3  # objective + decision + action item

    # --- 4/5. Replay, resolve with a fake embedder, replay again ------------
    registry = create_handler_registry()
    interview_ids = [(interview_id, ingest_result.fragment_count)]
    await _replay_all(project_id, interview_ids, registry)

    monkeypatch.setattr(ResolutionEngine, "_build_embedder", lambda self: _FakeEmbedder())
    resolution_result = await ResolutionEngine().apply(project_id)
    assert resolution_result.entities_canonicalized == 1
    assert resolution_result.persons_identified == 1
    assert resolution_result.speakers_linked == 1  # Jane only; Bob isn't a participant

    await _replay_all(project_id, interview_ids, registry)

    # --- 6. Assert the full Layer 1-4 subgraph ------------------------------
    async with await Neo4jConnectionManager.get_session() as session:
        shape = await _segment_shape(session, interview_id)
        assert shape == {"Vendor choice": (2, [0, 1]), "Timeline": (2, [2, 3])}

        res = await session.run(
            """
            MATCH (:Fragment {aggregate_id: $f0})-[:MENTIONS]->(e:Entity)
            WITH count(e) AS entities
            MATCH (li:LensItem {interview_id: $iid})
            WITH entities, count(DISTINCT li) AS lens_items
            MATCH (ce:CanonicalEntity {project_id: $project_id})
            WITH entities, lens_items, count(ce) AS canonicals,
                 collect(ce.entity_type) AS canonical_types
            MATCH (p:Person {display_name: 'Jane Doe', project_id: $project_id})
                  <-[r:IDENTIFIED_AS]-(sp:Speaker)
            RETURN entities, lens_items, canonicals, canonical_types,
                   count(DISTINCT sp) AS linked_speakers,
                   collect(DISTINCT r.method) AS methods
            """,
            f0=f0, iid=interview_id, project_id=project_id,
        )
        record = await res.single()
        assert record is not None, "Layer 2/3/4 overlay did not materialize"
        assert record["entities"] == 1
        assert record["lens_items"] == 3
        assert record["canonicals"] == 1
        assert record["canonical_types"] == ["organization"]
        assert record["linked_speakers"] == 1
        assert record["methods"] == ["front_matter"]

        # Dual-label invariant (the layer4 smoke's exact query).
        dual_label = await session.run(
            """
            MATCH (n) WHERE (n:Sentence AND NOT n:Fragment) OR (n:Fragment AND NOT n:Sentence)
            RETURN count(n) AS mismatched
            """
        )
        dual_label_record = await dual_label.single()
        assert dual_label_record["mismatched"] == 0

    # --- 7. Idempotent segment re-run (unforced) -----------------------------
    second_result = await EnrichmentOrchestrator().enrich_interview(interview_id)
    assert second_result.segments_extracted == 0  # resume gate: live segments skip
    assert second_result.fragments_enriched == 0
    assert second_result.claims_extracted == 0
    assert second_result.flags == {}
    assert executor.run_spec_on_text.await_count == 1  # no second document pass

    await _replay_all(project_id, interview_ids, registry)

    async with await Neo4jConnectionManager.get_session() as session:
        shape = await _segment_shape(session, interview_id)
        assert shape == {"Vendor choice": (2, [0, 1]), "Timeline": (2, [2, 3])}

    # --- 8. Export: segment headings in transcript.md, in order -------------
    await OkfExporter().export(
        interview_id, "meeting_minutes", out_dir=str(tmp_path / "exports")
    )
    bundle = tmp_path / "exports" / f"{interview_id}-meeting_minutes"
    transcript_md = (bundle / "transcript.md").read_text()
    vendor_heading = transcript_md.index("## Vendor choice")
    vendor_text = transcript_md.index("We will go with Acme Corp")
    timeline_heading = transcript_md.index("## Timeline")
    timeline_text = transcript_md.index("The timeline is six weeks")
    assert vendor_heading < vendor_text < timeline_heading < timeline_text

    # --- 9. Correction round-trip: remove the Timeline segment --------------
    interview = await interview_repo.load(interview_id)
    timeline_id = next(
        sid for sid, seg in interview.segments.items()
        if seg["topic"] == "Timeline" and not seg["removed"]
    )
    interview.remove_segment(timeline_id, reason="wrong boundary")
    await interview_repo.save(interview)

    await _replay_all(project_id, interview_ids, registry)

    async with await Neo4jConnectionManager.get_session() as session:
        shape = await _segment_shape(session, interview_id)
        assert shape == {"Vendor choice": (2, [0, 1])}

    await OkfExporter().export(
        interview_id, "meeting_minutes", out_dir=str(tmp_path / "exports_after_removal")
    )
    bundle = tmp_path / "exports_after_removal" / f"{interview_id}-meeting_minutes"
    transcript_md = (bundle / "transcript.md").read_text()
    assert "## Vendor choice" in transcript_md
    assert "## Timeline" not in transcript_md
