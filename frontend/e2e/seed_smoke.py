#!/usr/bin/env python3
"""Seed/cleanup helper for the UI Playwright smoke (frontend/e2e/smoke.spec.ts).

Seeds one interview through the REAL command path (`IngestionOrchestrator`,
same idiom as `tests/integration/test_deployed_projection_smoke.py`) against
the shared dev EventStoreDB, so the dockerized `projection-service` delivers
it to dev Neo4j exactly like a real ingest would. A LABELED transcript is
used deliberately: speaker assignment is parsed from labels, not inferred, so
no LLM call happens during seeding (matches the brief: "seeding does NOT need
enrichment/LLM calls").

Three subcommands, all print one JSON line to stdout (spec parses it):

  seed         ingest a small fixed transcript; prints
               {"project_id", "interview_id", "title", "first_line_text",
               "fragment_count"}
  append-line  append ONE new line to an ALREADY-seeded interview via the
               real command path (CreateSentenceCommand/SentenceCommandHandler
               -- the same genesis event ingestion itself emits per fragment,
               just issued directly instead of through IngestionOrchestrator
               since that only ever creates a brand-new interview_id). This is
               M5.1's live-append smoke leg: the transcript page must show
               this line with NO user action, driven by the SSE bridge.
               Takes --interview-id, --index (use the seed's fragment_count
               so it doesn't collide with the seeded indices), --text; prints
               {"sentence_id", "interview_id", "index", "text"}
  cleanup      delete the seeded Interview/Fragment/Speaker/Utterance/Project
               nodes from dev Neo4j (mirrors test_deployed_projection_smoke.py's
               teardown query); takes --project-id and --interview-id (also
               cleans up anything append-line added, via the same
               interview_id-scoped DETACH DELETE)

Self-contained: loads the repo-root .env (like tests/conftest.py's
`_load_env_file`, so ANTHROPIC_API_KEY etc. are present for the agent-factory
import-time singleton) and then unconditionally overrides
ESDB_CONNECTION_STRING to esdb://localhost:2113?tls=false -- the committed
.env points ESDB at the docker-internal "eventstore" hostname (correct for
the dockerized app/worker/projection-service containers, but unresolvable
from this host-run script). Callers (the Makefile, the Playwright spec)
don't need to set up any env themselves. See smoke.spec.ts's header for the
full required-services list and how this script fits into the smoke.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_env_file() -> None:
    """Load repo-root .env into os.environ (existing keys win), then force
    ESDB_CONNECTION_STRING to the host-reachable dev EventStoreDB address."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key not in os.environ:
                os.environ[key] = value
    os.environ["ESDB_CONNECTION_STRING"] = "esdb://localhost:2113?tls=false"


_load_env_file()

from neo4j import AsyncGraphDatabase  # noqa: E402

# Dev Neo4j (docker-compose's `neo4j` service, host-exposed) — same
# credentials as tests/integration/test_deployed_projection_smoke.py.
DEV_NEO4J_URI = "bolt://localhost:7687"
DEV_NEO4J_USER = "neo4j"
DEV_NEO4J_PASSWORD = "aB3cD4eF5gH6iJ7kL8m"

TITLE = "UI Smoke Interview"

TRANSCRIPT = f"""---
title: {TITLE}
participants: [Jane Doe, Bob Smith]
---
Jane: We will go with Acme Corp and I'll draft the doc by Friday.
Bob: Sounds good to me, that works for the schedule.
"""


async def _seed() -> dict:
    # Deferred import: this pulls in src.agents (eager agent-factory
    # singleton at import time, needs ANTHROPIC_API_KEY) — only the `seed`
    # subcommand needs it, `cleanup` shouldn't have to satisfy it.
    from src.ingestion.orchestrator import IngestionOrchestrator

    project_id = f"ui-smoke-{uuid.uuid4()}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "ui_smoke.txt"
        input_file.write_text(TRANSCRIPT)

        orchestrator = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
        result = await orchestrator.ingest_file(input_file)

    return {
        "project_id": project_id,
        "interview_id": result.interview_id,
        "title": TITLE,
        "first_line_text": "We will go with Acme Corp and I'll draft the doc by Friday.",
        "fragment_count": result.fragment_count,
    }


async def _append_line(interview_id: str, index: int, text: str) -> dict:
    # Deferred import: same rationale as _seed()'s IngestionOrchestrator
    # import -- only needed for this subcommand, not `cleanup`.
    import uuid

    from src.commands.handlers import get_sentence_command_handler
    from src.commands.sentence_commands import CreateSentenceCommand
    from src.events.envelope import Actor, ActorType

    sentence_id = str(uuid.uuid4())
    handler = get_sentence_command_handler()
    command = CreateSentenceCommand(
        sentence_id=sentence_id,
        interview_id=interview_id,
        index=index,
        text=text,
        actor=Actor(actor_type=ActorType.SYSTEM, user_id="ui-smoke-append"),
    )
    await handler.handle(command)

    return {
        "sentence_id": sentence_id,
        "interview_id": interview_id,
        "index": index,
        "text": text,
    }


async def _cleanup(project_id: str, interview_id: str) -> dict:
    driver = AsyncGraphDatabase.driver(DEV_NEO4J_URI, auth=(DEV_NEO4J_USER, DEV_NEO4J_PASSWORD))
    try:
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
    finally:
        await driver.close()
    return {"cleaned_up": True, "project_id": project_id, "interview_id": interview_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed")
    append_parser = sub.add_parser("append-line")
    append_parser.add_argument("--interview-id", required=True)
    append_parser.add_argument("--index", required=True, type=int)
    append_parser.add_argument("--text", required=True)
    cleanup_parser = sub.add_parser("cleanup")
    cleanup_parser.add_argument("--project-id", required=True)
    cleanup_parser.add_argument("--interview-id", required=True)

    args = parser.parse_args()

    if args.command == "seed":
        output = asyncio.run(_seed())
    elif args.command == "append-line":
        output = asyncio.run(_append_line(args.interview_id, args.index, args.text))
    else:
        output = asyncio.run(_cleanup(args.project_id, args.interview_id))

    print(json.dumps(output))


if __name__ == "__main__":
    main()
