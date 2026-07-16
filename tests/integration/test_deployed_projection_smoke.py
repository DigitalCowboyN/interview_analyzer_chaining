"""Deployed-path smoke (integration): first end-to-end proof that the
dockerized projection service delivers events to Neo4j.

Unlike test_end_to_end_smoke.py, this test does NOT replay events in-process
through the handler registry — delivery is the dockerized `projection-service`
container's job here. The test only seeds genesis events through the real
command path (ingestion orchestrator) against the shared ESDB, then polls the
DEV Neo4j (not the test instance) for the projected graph.

Requires: `docker compose up -d --build neo4j eventstore projection-service`
(the `make deployed-smoke` target does this). Manages real containers' data,
so it is gated behind DEPLOYED_SMOKE=1 and MUST NOT run in default suites.
"""

import asyncio
import os
import uuid as uuid_mod

import pytest
from neo4j import AsyncGraphDatabase

from src.ingestion.orchestrator import IngestionOrchestrator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("DEPLOYED_SMOKE") != "1",
        reason="deployed-path smoke: run via `make deployed-smoke`",
    ),
]

DEV_NEO4J_URI = "bolt://localhost:7687"
DEV_NEO4J_USER = "neo4j"
DEV_NEO4J_PASSWORD = "aB3cD4eF5gH6iJ7kL8m"

POLL_TIMEOUT_S = 90
POLL_INTERVAL_S = 2

LABELED = """---
title: Deployed Smoke Interview
project: deployed-smoke
date: 2026-07-16
participants: [Jane Doe]
---
Jane: We will go with Acme Corp and I'll draft the doc by Friday.
Bob: Sounds good to me.
Jane: The timeline is six weeks starting Monday.
Bob: That works for the schedule.
"""


async def _poll_fragment_count(session, interview_id: str, expected: int):
    """Poll dev Neo4j (up to POLL_TIMEOUT_S) for the projected Fragment count."""
    deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT_S
    last_count = -1
    while asyncio.get_event_loop().time() < deadline:
        result = await session.run(
            """
            MATCH (i:Interview {interview_id: $iid})-[:HAS_SENTENCE]->(f:Fragment)
            RETURN count(f) AS n
            """,
            iid=interview_id,
        )
        record = await result.single()
        last_count = record["n"] if record else 0
        if last_count == expected:
            return last_count
        await asyncio.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"Projection service did not deliver within {POLL_TIMEOUT_S}s: "
        f"expected {expected} Fragment nodes under Interview {interview_id}, "
        f"last observed {last_count}. Check `docker logs "
        f"interview_analyzer_projection_service`."
    )


@pytest.mark.asyncio
async def test_dockerized_projection_service_delivers_sentences_to_neo4j(tmp_path):
    """Seed genesis events through the real command path; the dockerized
    projection service (not this test) must deliver them to dev Neo4j."""
    driver = AsyncGraphDatabase.driver(
        DEV_NEO4J_URI, auth=(DEV_NEO4J_USER, DEV_NEO4J_PASSWORD)
    )

    project_id = f"deployed-smoke-{uuid_mod.uuid4()}"
    input_file = tmp_path / "deployed_smoke.txt"
    input_file.write_text(LABELED)

    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    try:
        async with driver.session() as session:
            fragment_count = await _poll_fragment_count(
                session, interview_id, ingest_result.fragment_count
            )
            assert fragment_count == ingest_result.fragment_count

            # Dual-label invariant on this interview's projected nodes.
            dual_label = await session.run(
                """
                MATCH (i:Interview {interview_id: $iid})-[:HAS_SENTENCE]->(f)
                WHERE NOT (f:Sentence AND f:Fragment)
                RETURN count(f) AS mismatched
                """,
                iid=interview_id,
            )
            dual_label_record = await dual_label.single()
            assert dual_label_record["mismatched"] == 0

            # Schema proof: the service ran ensure_schema on its own target.
            indexes = await session.run("SHOW INDEXES")
            index_names = {rec["name"] async for rec in indexes}
            assert "segment_segment_id" in index_names
    finally:
        async with driver.session() as session:
            await session.run(
                """
                MATCH (i:Interview {interview_id: $iid})
                OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
                DETACH DELETE i, f
                """,
                iid=interview_id,
            )
        await driver.close()
