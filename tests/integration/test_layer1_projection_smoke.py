"""Layer 1 projection smoke test (integration).

Automates the plan's manual verification step: ingest a small labeled
transcript (events into ESDB), replay the resulting events through the real
handler registry against Neo4j, and assert the Speaker/Utterance subgraph
materializes. Requires the test infrastructure (`make test-infra-up`).
"""

import uuid as uuid_mod

import pytest

from src.events.repository import get_repository_factory
from src.ingestion.orchestrator import IngestionOrchestrator
from src.projections.bootstrap import create_handler_registry
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

LABELED = """Alice: Hi, thanks for joining today.
Bob: Happy to be here.
Alice: Let's get started.
"""


@pytest.mark.asyncio
async def test_ingested_interview_projects_speaker_utterance_subgraph(tmp_path):
    input_file = tmp_path / "smoke_meeting.txt"
    input_file.write_text(LABELED)

    orchestrator = IngestionOrchestrator(
        project_id=f"smoke-{uuid_mod.uuid4()}", map_dir=tmp_path / "maps"
    )
    result = await orchestrator.ingest_file(input_file)
    assert result.speaker_count == 2

    # Replay this interview's events through the real handlers (simulating the
    # projection service without needing persistent subscriptions running).
    factory = get_repository_factory()
    interview_repo = factory.create_interview_repository()
    sentence_repo = factory.create_sentence_repository()
    registry = create_handler_registry()

    events = []
    interview = await interview_repo.load(result.interview_id)
    assert interview is not None
    events.extend(await interview_repo.event_store.read_stream(f"Interview-{result.interview_id}"))
    for index in range(result.fragment_count):
        sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{result.interview_id}:{index}"))
        events.extend(await sentence_repo.event_store.read_stream(f"Sentence-{sid}"))

    for event in events:
        handler = registry.get_handler(event.event_type)
        if handler:
            await handler.handle(event)

    async with await Neo4jConnectionManager.get_session() as session:
        res = await session.run(
            """
            MATCH (i:Interview {interview_id: $iid})-[:HAS_PARTICIPANT]->(sp:Speaker)
            OPTIONAL MATCH (sp)-[:SPOKE]->(u:Utterance)<-[:PART_OF_UTTERANCE]-(s:Sentence)
            RETURN count(DISTINCT sp) AS speakers,
                   count(DISTINCT u) AS utterances,
                   count(DISTINCT s) AS fragments
            """,
            iid=result.interview_id,
        )
        record = await res.single()
        assert record["speakers"] == 2
        assert record["utterances"] == 3
        assert record["fragments"] == 3
