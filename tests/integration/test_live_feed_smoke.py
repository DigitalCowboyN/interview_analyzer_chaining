"""Live-feed smoke (integration, M5.1 Task 6): first end-to-end proof that
the SSE bridge (`src/ui/notifications.py`'s `EsdbWatcher` + `NotificationHub`,
fed to `src/api/routers/ui.py::stream_events`) delivers a REAL ESDB event to
a live SSE subscriber -- not a stubbed watcher (that's what
tests/api/test_ui_router.py's unit tests already cover), the genuine
catch-up-subscription-to-browser-frame path.

In-process, but NOT via an HTTP client: httpx's `ASGITransport` (like
starlette's `TestClient` -- see tests/api/test_ui_router.py's module
docstring) awaits the WHOLE ASGI app call to completion before returning any
Response. `stream_events`'s generator loops until disconnect, so driving it
through either of those would just hang forever waiting for a Response that
never comes. Instead this calls the route function directly and iterates
its returned `StreamingResponse.body_iterator` by hand -- the same technique
test_ui_router.py's own "generator-level" tests use (e.g.
`test_stream_events_generator_aclosed_mid_stream_releases_subscription`),
just against the REAL `get_live_feed()` singleton (real NotificationHub +
real EsdbWatcher talking to real ESDB) instead of a stubbed watcher.

Requires: `docker compose up -d --build neo4j eventstore projection-service`
(the `make live-feed-smoke` target does this, mirroring `deployed-smoke`'s
dev-stack bring-up for one consistent recipe across all three smokes). Only
EventStoreDB is actually exercised by this test's assertions -- notifications
come straight off ESDB's `$ce-*` catch-up subscriptions, never through
Neo4j/the projection service -- but the shared dev stack is brought up
uniformly regardless. Needs ESDB_CONNECTION_STRING overridden to
esdb://localhost:2113?tls=false for this host-run process: the committed
.env points ESDB at the docker-internal "eventstore" hostname, unresolvable
from here (same trap documented in frontend/e2e/seed_smoke.py's header and
tests/integration/test_deployed_projection_smoke.py's Makefile comment) --
`make live-feed-smoke` sets this in the invoking shell before pytest starts,
so no in-test override is needed (get_event_store_client() reads the env var
lazily, on first use, not at import time).

Gated behind LIVE_FEED_SMOKE=1 (mirrors DEPLOYED_SMOKE=1's skip idiom
exactly); MUST NOT run in default suites.
"""
# verifies: code:ui

import asyncio
import json
import os
import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request as StarletteRequest

from src.enrichment.executor import SpecOutcome
from src.ingestion.orchestrator import IngestionOrchestrator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("LIVE_FEED_SMOKE") != "1",
        reason="live-feed smoke: run via `make live-feed-smoke`",
    ),
]

# Generous, deployed-smoke-spirited timeout for the actual notification: real
# ESDB round trip (ingest -> $ce-Interview catch-up subscription -> hub
# publish), not a fixed/short bound.
NOTIFICATION_TIMEOUT_S = 60

# Sped up from the route's 15s production default so this test doesn't spend
# most of its runtime idling -- still slow enough to be an unambiguous
# "nothing happened yet" signal rather than noise.
HEARTBEAT_TEST_SECONDS = 2.0

# Buffer between the SSE stream opening (subscribe() + ensure_started()) and
# ingesting. ensure_started() only SPAWNS the EsdbWatcher's three per-category
# background tasks -- it returns before their subscribe_to_stream() calls
# actually establish a live position against ESDB. Without this buffer,
# ingesting immediately risks a genuine race: our events committing before
# the watcher's `from_end=True` subscriptions are actually live, which would
# make them invisible to it (from_end means "only events from here forward")
# and this test would then hang until NOTIFICATION_TIMEOUT_S and fail.
SUBSCRIPTION_SETTLE_S = 3.0

LABELED = """---
title: Live Feed Smoke
participants: [Jane Doe]
---
Jane: We will go with Acme Corp and I'll draft the doc by Friday.
"""


def _bare_request() -> StarletteRequest:
    """Minimal starlette Request for driving stream_events directly -- same
    idiom as tests/api/test_ui_router.py's `_bare_request()`. `is_disconnected`
    is patched separately (see the test) so the generator's loop never exits
    on its own; this test controls termination itself via `aclose()`."""
    return StarletteRequest(
        {"type": "http", "method": "GET", "path": "/ui/streams/events", "headers": [], "query_string": b""}
    )


class _FixedFirstUUID4:
    """Callable uuid4() replacement: returns `fixed` on the FIRST call,
    delegates to the real uuid4() for every call after.

    IngestionOrchestrator.ingest_file generates its own random interview_id
    (src/ingestion/orchestrator.py:93, `str(uuid.uuid4())`) -- there is no
    parameter to override it. But NotificationHub only delivers a
    "transcript" notification to a subscriber whose `interview_id` matches
    the event's exactly (see NotificationHub._matches) -- so to open the SSE
    subscription for the SAME interview_id the ingest is about to produce
    (required by the ordering invariant: subscribe before publish), that one
    generation needs to be deterministic and known in advance.

    This is genuinely the FIRST uuid4() call once ingestion starts: passing
    an explicit `correlation_id` to `ingest_file` (see the test) skips the
    only uuid4() call that would otherwise happen earlier
    (`generate_correlation_id()`), and nothing between that and line 93
    (Actor construction, Path/text handling, `normalize()`) calls uuid4.
    Every event-id generated afterward (EventEnvelope's default_factory,
    once real events start getting constructed) correctly falls through to
    the real uuid4() -- only the interview_id itself is pinned.
    """

    def __init__(self, fixed: uuid_mod.UUID):
        self._fixed = fixed
        self._real = uuid_mod.uuid4
        self._used = False

    def __call__(self) -> uuid_mod.UUID:
        if not self._used:
            self._used = True
            return self._fixed
        return self._real()


@pytest.mark.asyncio
async def test_live_feed_delivers_transcript_notification_on_ingest(tmp_path, monkeypatch):
    from src.api.routers.ui import stream_events

    fresh_interview_uuid = uuid_mod.uuid4()
    fresh_interview_id = str(fresh_interview_uuid)
    project_id = f"live-feed-smoke-{uuid_mod.uuid4()}"

    monkeypatch.setattr(uuid_mod, "uuid4", _FixedFirstUUID4(fresh_interview_uuid))
    monkeypatch.setattr("src.api.routers.ui.HEARTBEAT_SECONDS", HEARTBEAT_TEST_SECONDS)
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", AsyncMock(return_value=False)
    )

    response = await stream_events(
        _bare_request(), interview_id=fresh_interview_id, project_id=project_id
    )
    assert response.media_type == "text/event-stream"
    generator = response.body_iterator

    try:
        # ORDERING INVARIANT: pulling this FIRST chunk is what actually runs
        # `hub.subscribe(...)` + `watcher.ensure_started()` -- both live
        # inside the generator body, before its first yield (see
        # stream_events' docstring in src/api/routers/ui.py). This MUST
        # happen before ingesting: the EsdbWatcher's catch-up subscriptions
        # are `from_end=True` (only see events from the moment they connect
        # onward), so an ingest that runs first would produce events the
        # watcher can never see, and this test would then hang until
        # NOTIFICATION_TIMEOUT_S. The first chunk is always the ": connected"
        # prelude (yielded immediately, before the heartbeat wait-loop, so a
        # buffering proxy can't delay EventSource.onopen); heartbeat framing
        # is exercised separately by the ": keep-alive" skip in the loop below
        # (a heartbeat fires during the settle + notification wait).
        first_chunk = await asyncio.wait_for(
            generator.__anext__(), timeout=HEARTBEAT_TEST_SECONDS + 10
        )
        assert first_chunk == ": connected\n\n"

        await asyncio.sleep(SUBSCRIPTION_SETTLE_S)

        input_file = tmp_path / "live_feed_smoke.txt"
        input_file.write_text(LABELED)
        orchestrator = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
        result = await orchestrator.ingest_file(input_file, correlation_id="live-feed-smoke-corr")
        assert result.interview_id == fresh_interview_id

        transcript_notification = None
        deadline = asyncio.get_running_loop().time() + NOTIFICATION_TIMEOUT_S
        while transcript_notification is None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                pytest.fail(
                    f"No transcript notification for interview {fresh_interview_id} "
                    f"within {NOTIFICATION_TIMEOUT_S}s of ingest."
                )
            chunk = await asyncio.wait_for(generator.__anext__(), timeout=remaining)
            if chunk == ": keep-alive\n\n":
                continue
            assert chunk.startswith("data: ") and chunk.endswith("\n\n")
            payload = json.loads(chunk[len("data: "):].strip())
            if payload.get("surface") == "transcript" and payload.get("interview_id") == fresh_interview_id:
                transcript_notification = payload

        assert transcript_notification == {"surface": "transcript", "interview_id": fresh_interview_id}
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_live_feed_delivers_project_notifications_for_resolution_and_lens(
    tmp_path, monkeypatch
):
    """Gallery liveness (M5.1b): a project-scoped SSE subscriber receives a
    `project` notification for (a) a Project-stream resolution event and
    (b) stamped Interview-stream persona-lens events. Leg (b) proves the
    aggregate-stamp mechanism end-to-end: the persona lens run emits events
    whose project_id lives only in ESDB metadata, and the watcher recovers it
    there and routes them to the gallery.

    Emits the resolution event through the Project aggregate + repository (the
    same calls the resolution router makes) and runs the persona lens with a
    mocked executor (LLM-free, mirroring test_layer3_lens_smoke.py) -- both
    produce real, stamped ESDB events, which is exactly the path under test.
    The stream is project-scoped (no interview_id), the gallery's own scope.
    """
    from src.api.routers.ui import stream_events
    from src.events.aggregates import Project
    from src.events.project_events import project_aggregate_id
    from src.events.repository import get_project_repository
    from src.lens.engine import LensEngine

    project_id = f"live-gallery-smoke-{uuid_mod.uuid4()}"

    monkeypatch.setattr("src.api.routers.ui.HEARTBEAT_SECONDS", HEARTBEAT_TEST_SECONDS)
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", AsyncMock(return_value=False)
    )

    # Ingest a labeled interview into this project (Layer 1) BEFORE opening the
    # stream -- the persona lens re-run below is the event we watch for, not
    # this ingest (the watcher is from_end and would miss it anyway).
    input_file = tmp_path / "gallery_smoke.txt"
    input_file.write_text(LABELED)
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    response = await stream_events(_bare_request(), interview_id=None, project_id=project_id)
    assert response.media_type == "text/event-stream"
    generator = response.body_iterator

    async def next_project_notification():
        deadline = asyncio.get_running_loop().time() + NOTIFICATION_TIMEOUT_S
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                pytest.fail(
                    f"No project notification for {project_id} within {NOTIFICATION_TIMEOUT_S}s."
                )
            chunk = await asyncio.wait_for(generator.__anext__(), timeout=remaining)
            if chunk in (": keep-alive\n\n", ": connected\n\n"):
                continue
            assert chunk.startswith("data: ") and chunk.endswith("\n\n")
            payload = json.loads(chunk[len("data: "):].strip())
            if payload.get("surface") == "project" and payload.get("project_id") == project_id:
                return payload

    try:
        first_chunk = await asyncio.wait_for(
            generator.__anext__(), timeout=HEARTBEAT_TEST_SECONDS + 10
        )
        assert first_chunk == ": connected\n\n"
        await asyncio.sleep(SUBSCRIPTION_SETTLE_S)

        # --- Leg (a): a Project-stream resolution event -> project notification.
        repo = get_project_repository()
        project = Project(project_aggregate_id(project_id))
        project.identify_person(project_id, str(uuid_mod.uuid4()), "Jane Doe")
        await repo.save(project)
        assert await next_project_notification() == {"surface": "project", "project_id": project_id}

        # --- Leg (b): a persona-lens run -> stamped Interview-stream events ->
        # project notification (mechanism proof; executor mocked, LLM-free).
        executor = MagicMock()
        executor.run_spec_on_text = AsyncMock(
            side_effect=lambda spec, text, ctx=None: SpecOutcome(
                data={spec.name: []}, provider="anthropic", model="haiku"
            )
        )
        monkeypatch.setattr(LensEngine, "_build_executor", lambda self, lens: executor)
        await LensEngine().apply(interview_id, "persona")
        assert await next_project_notification() == {"surface": "project", "project_id": project_id}
    finally:
        await generator.aclose()
