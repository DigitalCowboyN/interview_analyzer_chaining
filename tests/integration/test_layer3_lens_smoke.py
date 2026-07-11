"""Layer 3 lens smoke test (integration).

Ingests a small labeled transcript, applies the meeting_minutes lens with the
executor mocked (canned SpecOutcomes, no live LLM), replays every event through
the real handler registry in commit order against real Neo4j, and asserts the
lens subgraph materializes (dual-label nodes, speaker links, fragment
grounding). Requires `make test-infra-up`.
"""

import uuid as uuid_mod

import pytest

from src.enrichment.executor import SpecOutcome
from src.events.repository import get_repository_factory
from src.lens.engine import LensEngine
from src.projections.bootstrap import create_handler_registry
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

LABELED = """Alice: We will go with vendor X and I'll draft the doc by Friday.
Bob: Sounds good to me.
"""

CANNED = {
    "objectives": {"objectives": [{"text": "Choose a vendor", "confidence": 0.9}]},
    "decisions": {
        "decisions": [{"text": "Go with vendor X", "made_by": "Alice", "confidence": 0.9}]
    },
    "action_items": {
        "action_items": [
            {"text": "Draft the doc", "owner": "SELF", "due": "Friday", "confidence": 0.8}
        ]
    },
    "followups": {"followups": []},
}

EMPTY = {
    "objectives": {"objectives": []},
    "decisions": {"decisions": []},
    "action_items": {"action_items": []},
    "followups": {"followups": []},
}


def canned_outcome(spec, text):
    """Only Alice's utterance (and the document) yields items."""
    source = CANNED if "vendor X" in text else EMPTY
    return SpecOutcome(data=source[spec.name], provider="anthropic", model="haiku")


@pytest.mark.asyncio
async def test_lens_projects_dual_label_nodes_with_links_and_grounding(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.ingestion.orchestrator import IngestionOrchestrator

    input_file = tmp_path / "smoke_lens.txt"
    input_file.write_text(LABELED)

    project_id = f"smoke-{uuid_mod.uuid4()}"
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    # Mock the executor: canned outcome per lens extractor, no live LLM.
    executor = MagicMock()
    executor.run_spec_on_text = AsyncMock(
        side_effect=lambda spec, text, ctx=None: canned_outcome(spec, text)
    )
    monkeypatch.setattr(LensEngine, "_build_executor", lambda self, lens: executor)

    result = await LensEngine().apply(interview_id, "meeting_minutes")
    assert result.items_extracted == 3  # objective + decision + action item

    # Replay all events in commit order through the real registry.
    factory = get_repository_factory()
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
            MATCH (d:LensItem:Decision {interview_id: $iid})-[:DECIDED_BY]->(dsp:Speaker)
            WITH count(DISTINCT d) AS decisions, collect(DISTINCT dsp.display_name) AS deciders
            MATCH (a:LensItem:ActionItem {interview_id: $iid})-[:SUPPORTED_BY]->(:Fragment)
            MATCH (a)-[:OWNED_BY]->(:Speaker)
            WITH decisions, deciders, count(DISTINCT a) AS actions
            MATCH (o:LensItem:Objective {interview_id: $iid})
            RETURN decisions, deciders, actions, count(DISTINCT o) AS objectives
            """,
            iid=interview_id,
        )
        record = await res.single()
        assert record is not None, "lens subgraph did not materialize"
        assert record["decisions"] == 1
        assert record["deciders"] == ["Alice"]
        assert record["actions"] == 1
        assert record["objectives"] == 1
