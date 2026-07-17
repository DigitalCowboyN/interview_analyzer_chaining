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
import requests
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

ESDB_HTTP_BASE = "http://localhost:2113"
ESDB_HTTP_AUTH = ("admin", "changeit")
SENTENCE_GROUP_INFO_URL = f"{ESDB_HTTP_BASE}/subscriptions/%24ce-Sentence/neo4j-projection-sentence-v1/info"

CHECKPOINT_POLL_TIMEOUT_S = 15
CHECKPOINT_POLL_INTERVAL_S = 1


def _get_sentence_group_info() -> dict:
    """
    Query the ESDB HTTP API for the Sentence category stream's consumer
    group info. Returns a dict with at least `parkedMessageCount`,
    `lastProcessedEventNumber`, and `totalItemsProcessed`; if the group
    doesn't exist yet (e.g. before any event has ever been ingested),
    returns a synthetic baseline so callers can treat "no group yet" the
    same as "group exists but hasn't processed anything".
    """
    response = requests.get(SENTENCE_GROUP_INFO_URL, auth=ESDB_HTTP_AUTH, timeout=10)
    if response.status_code == 404:
        return {
            "lastProcessedEventNumber": -1,
            "parkedMessageCount": 0,
            "totalItemsProcessed": 0,
            "totalInFlightMessages": 0,
        }
    response.raise_for_status()
    return response.json()


async def _poll_consumer_group_progress(baseline_total_processed: int):
    """
    Poll the Sentence consumer group's info (up to CHECKPOINT_POLL_TIMEOUT_S)
    until `totalItemsProcessed` advances past its pre-ingest baseline.

    `totalItemsProcessed` counts DELIVERIES, not acks -- a redelivery
    (e.g. from a failed or wrong-id ack) also advances it, so this alone
    cannot prove acks are succeeding. It's still useful as a liveness
    signal (the group is receiving/processing events at all), unlike
    `lastProcessedEventNumber`/`lastCheckpointedEventPosition`, which are
    only written when esdbclient's persistent-subscription checkpoint
    logic fires (batched: `minCheckPointCount: 10` acks OR
    `checkPointAfterMilliseconds: 2000` -- see `_ensure_subscription_exists`
    config), so on a low-volume smoke run those can legitimately lag behind
    real, successful acks by one run's worth of events. Polling here
    (rather than a single point-in-time check) absorbs that checkpoint
    latency while still proving forward progress. The deterministic
    ack-id regression guard lives in
    tests/projections/test_subscription_manager_unit.py, not here.
    """
    deadline = asyncio.get_running_loop().time() + CHECKPOINT_POLL_TIMEOUT_S
    info = _get_sentence_group_info()
    while asyncio.get_running_loop().time() < deadline:
        info = _get_sentence_group_info()
        if (
            info["totalItemsProcessed"] > baseline_total_processed
            and info["totalInFlightMessages"] == 0
        ):
            return info
        await asyncio.sleep(CHECKPOINT_POLL_INTERVAL_S)
    pytest.fail(
        f"Sentence consumer group did not settle within "
        f"{CHECKPOINT_POLL_TIMEOUT_S}s: totalItemsProcessed="
        f"{info['totalItemsProcessed']} (baseline {baseline_total_processed}), "
        f"totalInFlightMessages={info['totalInFlightMessages']}. Events may be "
        f"stuck in-flight (never acked) -- check `docker logs "
        f"interview_analyzer_projection_service`."
    )


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
    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT_S
    last_count = -1
    while asyncio.get_running_loop().time() < deadline:
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

    # Capture the Sentence consumer group's baseline BEFORE seeding, so we
    # can assert real forward progress afterward rather than just "some
    # positive number". The group may not exist yet on a fresh ESDB
    # instance (never ingested before) -- _get_sentence_group_info()
    # treats a 404 as a zeroed-out baseline.
    baseline_info = _get_sentence_group_info()
    baseline_total_processed = baseline_info["totalItemsProcessed"]

    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    try:
        async with driver.session() as session:
            fragment_count = await _poll_fragment_count(
                session, interview_id, ingest_result.fragment_count
            )
            assert fragment_count == ingest_result.fragment_count

            # Single-label invariant on this interview's projected nodes
            # (M4.8 shim drop): :Fragment only -- no :Sentence label on new
            # nodes written by the deployed (dockerized) projection service.
            fragment_labels = await session.run(
                """
                MATCH (i:Interview {interview_id: $iid})-[:HAS_SENTENCE]->(f:Fragment)
                RETURN count(f) AS total,
                       count(CASE WHEN f:Sentence THEN 1 END) AS mislabeled
                """,
                iid=interview_id,
            )
            fragment_labels_record = await fragment_labels.single()
            assert fragment_labels_record["total"] > 0
            assert fragment_labels_record["mislabeled"] == 0

            # Consumer-group settlement proof: now that Neo4j shows the
            # expected Fragment count, the Sentence subscription's delivery
            # counter must have advanced past baseline, nothing left
            # in-flight, and nothing parked. Note totalItemsProcessed counts
            # DELIVERIES (a redelivery advances it too), so this does not by
            # itself distinguish "acked once" from "acked never, delivered
            # repeatedly" -- it's a liveness/settlement check, not an ack-id
            # regression guard. The deterministic ack-id regression guard
            # (acking `event.ack_id` vs. `event.id` on resolved link events)
            # lives in tests/projections/test_subscription_manager_unit.py.
            # Polled (rather than a single point-in-time read) because these
            # counters can trail the actual acks by a beat under load.
            post_ingest_info = await _poll_consumer_group_progress(baseline_total_processed)
            assert post_ingest_info["parkedMessageCount"] == 0, (
                f"Sentence consumer group has parked messages: "
                f"{post_ingest_info['parkedMessageCount']}. Acks are likely "
                f"failing server-side (e.g. wrong ack id) -- check `docker "
                f"logs interview_analyzer_projection_service`."
            )

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
                OPTIONAL MATCH (i)-[:HAS_PARTICIPANT]->(sp:Speaker)
                OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(u:Utterance)
                MATCH (p:Project {project_id: $pid})
                DETACH DELETE i, f, sp, u, p
                """,
                iid=interview_id,
                pid=project_id,
            )
        await driver.close()
