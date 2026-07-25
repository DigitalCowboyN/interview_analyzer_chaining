"""Projection-ordering smoke (integration, M4.9): proves the per-lane
commit_position reorder buffer fixes the cross-lane ordering race that made
projection flaky/lossy.

Before M4.9, an interview's events arrived via three independent ESDB
subscriptions and were processed in ARRIVAL order, so dependent handlers
(SentenceCreated's HAS_SENTENCE edge, SpeakerAttributed's SPOKEN_BY) often
ran before their referents projected -> orphaned fragments and null speakers,
non-deterministically. M4.9 releases each lane's events in commit_position
order, so referents are always present.

This smoke seeds SEVERAL interviews through the real command path
(IngestionOrchestrator) against the dockerized dev stack, then polls the DEV
Neo4j until each interview projects EVERY fragment WITH its HAS_SENTENCE edge
AND a non-null speaker (SPOKEN_BY) -- exactly the completeness the race used
to break. Running the loop repeatedly is the point: a single pass could pass
by luck; N consecutive complete projections is the evidence the flakiness is
gone.

Requires: `docker compose up -d --build neo4j eventstore projection-service`
(the `make projection-smoke` target does this, mirroring `deployed-smoke`).
Needs ESDB_CONNECTION_STRING=esdb://localhost:2113?tls=false for this
host-run process (the committed .env points ESDB at the docker-internal
"eventstore" hostname); `make projection-smoke` sets it before pytest.

Gated behind PROJECTION_SMOKE=1 (mirrors DEPLOYED_SMOKE=1); MUST NOT run in
default suites.
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
        os.environ.get("PROJECTION_SMOKE") != "1",
        reason="projection-ordering smoke: run via `make projection-smoke`",
    ),
]

DEV_NEO4J_URI = "bolt://localhost:7687"
DEV_NEO4J_USER = "neo4j"
DEV_NEO4J_PASSWORD = "aB3cD4eF5gH6iJ7kL8m"

POLL_TIMEOUT_S = 90
POLL_INTERVAL_S = 2

# Repeated seed+assert passes: one pass could project completely by luck; N
# consecutive complete projections is the evidence the cross-lane race is gone.
INTERVIEW_COUNT = 5

# Two labeled speakers so every fragment has an attributable speaker; the
# assertion requires BOTH fragments to carry a non-null SPOKEN_BY.
LABELED = """---
title: Projection Ordering Smoke
participants: [Jane Doe]
---
Jane: We will go with Acme Corp and I'll draft the doc by Friday.
Bob: Sounds good to me, that works for the schedule.
"""
EXPECTED_FRAGMENTS = 2


async def _poll_complete_projection(session, interview_id: str) -> list:
    """Poll dev Neo4j (up to POLL_TIMEOUT_S) until the interview projects all
    EXPECTED_FRAGMENTS fragments AND every fragment has a non-null speaker.

    Returns the final rows on success; fails with diagnostics on timeout. The
    query mirrors the /ui reader's fragment->interview linkage
    ((Interview)-[:HAS_SENTENCE]->(Fragment)) and speaker attribution
    ((Fragment)-[:SPOKEN_BY]->(Speaker)) so "complete" here means the same
    thing the workbench transcript needs to render a full line with a speaker.
    """
    query = """
    MATCH (i:Interview {interview_id: $iid})-[:HAS_SENTENCE]->(f:Fragment)
    OPTIONAL MATCH (f)-[:SPOKEN_BY]->(sp:Speaker)
    RETURN f.sentence_id AS fragment_id, sp.display_name AS speaker
    ORDER BY f.sequence_order
    """
    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT_S
    rows: list = []
    while asyncio.get_running_loop().time() < deadline:
        result = await session.run(query, iid=interview_id)
        rows = [dict(r) async for r in result]
        if len(rows) == EXPECTED_FRAGMENTS and all(r["speaker"] for r in rows):
            return rows
        await asyncio.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"Interview {interview_id} did not fully project within {POLL_TIMEOUT_S}s: "
        f"got {len(rows)}/{EXPECTED_FRAGMENTS} fragments, speakers="
        f"{[r['speaker'] for r in rows]}. A missing fragment or null speaker means "
        f"the cross-lane ordering race is NOT fixed. Check `docker logs "
        f"interview_analyzer_projection_service`."
    )


@pytest.mark.asyncio
async def test_projection_is_complete_and_reliable_across_repeated_interviews(tmp_path):
    """Seed INTERVIEW_COUNT interviews; each must project every fragment with a
    non-null speaker. N consecutive complete projections proves the reorder
    buffer eliminated the flaky/lossy cross-lane race."""
    driver = AsyncGraphDatabase.driver(
        DEV_NEO4J_URI, auth=(DEV_NEO4J_USER, DEV_NEO4J_PASSWORD)
    )
    seeded: list = []
    try:
        async with driver.session() as session:
            for n in range(INTERVIEW_COUNT):
                project_id = f"projection-smoke-{uuid_mod.uuid4()}"
                input_file = tmp_path / f"projection_smoke_{n}.txt"
                input_file.write_text(LABELED)
                orchestrator = IngestionOrchestrator(
                    project_id=project_id, map_dir=tmp_path / "maps"
                )
                result = await orchestrator.ingest_file(input_file)
                assert result.fragment_count == EXPECTED_FRAGMENTS
                seeded.append((project_id, result.interview_id))

                rows = await _poll_complete_projection(session, result.interview_id)
                assert len(rows) == EXPECTED_FRAGMENTS
                assert all(r["speaker"] for r in rows), (
                    f"interview {result.interview_id} projected a null speaker: {rows}"
                )

            # All interviews projected completely, in sequence.
            assert len(seeded) == INTERVIEW_COUNT
        # --- cleanup: remove the seeded graph (interview-scoped DETACH DELETE,
        # mirrors test_deployed_projection_smoke.py's teardown) ---
        async with driver.session() as session:
            for _project_id, interview_id in seeded:
                await session.run(
                    """
                    MATCH (i:Interview {interview_id: $iid})
                    OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
                    OPTIONAL MATCH (i)-[:HAS_PARTICIPANT]->(sp:Speaker)
                    OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(u:Utterance)
                    DETACH DELETE i, f, sp, u
                    """,
                    iid=interview_id,
                )
    finally:
        await driver.close()
