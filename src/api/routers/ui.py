"""
src/api/routers/ui.py

The `/ui/*` read layer (M5.0 Task 1): backend contract for the Next.js
frontend. Thin router — session → reader → shape; zero writes, no auth.

M5.1 Task 4 adds the SSE live-feed route: param validation, subscribe to the
module-level NotificationHub, and a small generator that formats queued
Notifications as SSE frames (formatting itself lives in src/ui/notifications
per that module's no-framework-coupling rule).
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.events.project_events import person_id_for
from src.resolution.candidates import normalize_name
from src.ui import reader
from src.ui.notifications import format_sse_event, get_live_feed
from src.utils.neo4j_driver import Neo4jConnectionManager

router = APIRouter(prefix="/ui", tags=["ui"])

# Seconds of subscriber quiet before a `: keep-alive\n\n` comment is sent so
# intermediate proxies/load balancers don't time out an idle connection.
# Module constant with an injectable override: tests monkeypatch this name
# (src.api.routers.ui.HEARTBEAT_SECONDS) to a small value to exercise the
# heartbeat path without a real 15s wait.
HEARTBEAT_SECONDS = 15.0


async def _require_project(session, project_id: str) -> None:
    if not await reader.project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


def _shape_transcript_line(row: dict) -> dict:
    speaker = (
        {"speaker_id": row["speaker_id"], "display_name": row["speaker_display_name"]}
        if row["speaker_id"] is not None else None
    )
    person = (
        {"person_id": row["person_id"], "display_name": row["person_display_name"]}
        if row["person_id"] is not None else None
    )
    segment = (
        {"segment_id": row["segment_id"], "topic": row["segment_topic"]}
        if row["segment_id"] is not None else None
    )
    return {
        "fragment_id": row["fragment_id"],
        "sequence_order": row["sequence_order"],
        "text": row["text"],
        "speaker": speaker,
        "person": person,
        "utterance_id": row["utterance_id"],
        "segment": segment,
        "entities": row["entities"],
        "lens_items": row["lens_items"],
        "edited": bool(row["edited"]),
    }


_DIMENSION_NODE_TYPES = {
    "traits": "Trait",
    "goals": "Goal",
    "pain_points": "PainPoint",
    "notable_quotes": "NotableQuote",
}


def _shape_dimension_item(row: dict) -> dict:
    return {
        "item_id": row["item_id"],
        "text": row["text"],
        "confidence": row["confidence"],
        "interview_id": row["interview_id"],
        "interview_title": row["interview_title"],
    }


@router.get("/projects")
async def list_projects():
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.project_rows(session)
    return {"projects": rows}


@router.get("/projects/{project_id}/interviews")
async def list_interviews(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.interview_rows(session, project_id)
    return {"interviews": rows}


@router.get("/interviews/{interview_id}/transcript")
async def get_transcript(interview_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        header = await reader.interview_header_row(session, interview_id)
        if header is None:
            raise HTTPException(status_code=404, detail=f"Interview {interview_id} not found")
        line_rows = await reader.transcript_line_rows(session, interview_id)
    return {
        "interview_id": header["interview_id"],
        "title": header["title"],
        "metadata": header["metadata"],
        "lines": [_shape_transcript_line(row) for row in line_rows],
    }


@router.get("/projects/{project_id}/personas")
async def list_personas(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.persona_card_rows(session, project_id)
    return {"personas": rows}


@router.get("/personas/{project_id}/{person_id}")
async def get_persona(project_id: str, person_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        if not await reader.persona_exists(session, project_id, person_id):
            raise HTTPException(status_code=404, detail=f"Persona {person_id} not found")
        display_row = await reader.person_display_name_row(session, project_id, person_id)
        detail_rows = await reader.persona_detail_rows(session, project_id, person_id)

    dimensions = {key: [] for key in _DIMENSION_NODE_TYPES}
    by_node_type = {node_type: key for key, node_type in _DIMENSION_NODE_TYPES.items()}
    for row in detail_rows:
        key = by_node_type.get(row["node_type"])
        if key is not None:
            dimensions[key].append(_shape_dimension_item(row))

    return {
        "person_id": person_id,
        "display_name": display_row["display_name"] if display_row else None,
        "dimensions": dimensions,
    }


@router.get("/projects/{project_id}/persons")
async def list_persons(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.person_card_rows(session, project_id)
    return {"persons": rows}


@router.get("/persons/{project_id}/{person_id}")
async def get_person(project_id: str, person_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        if not await reader.person_exists(session, project_id, person_id):
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        display_row = await reader.person_display_name_row(session, project_id, person_id)
        links = await reader.person_detail_rows(session, project_id, person_id)
        contributes = await reader.person_contributes_to_persona(session, project_id, person_id)
    return {
        "person_id": person_id,
        "display_name": display_row["display_name"] if display_row else None,
        "links": links,
        "contributes_to_persona": contributes,
    }


@router.get("/projects/{project_id}/person-id")
async def derive_person_id(project_id: str, display_name: str = Query(..., min_length=1)):
    """Compute-only id derivation for the create-new-person flow — the
    frontend must never derive ids itself (loose-coupling requirement)."""
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
    return {"person_id": person_id_for(project_id, normalize_name(display_name))}


@router.get("/streams/events")
async def stream_events(
    request: Request,
    interview_id: Optional[str] = Query(default=None),
    project_id: Optional[str] = Query(default=None),
):
    """SSE live feed (M5.1): the browser subscribes by interview_id and/or
    project_id (at least one required, else 422) and receives one
    `data: {...}\\n\\n` frame per matching Notification, plus a
    `: keep-alive\\n\\n` comment after HEARTBEAT_SECONDS of quiet.

    Lazy lifecycle both directions: the shared EsdbWatcher is (idempotently)
    started on this connection and stopped once the last subscriber
    disconnects — see get_live_feed() / EsdbWatcher in src/ui/notifications.py.
    """
    if not interview_id and not project_id:
        raise HTTPException(status_code=422, detail="interview_id or project_id is required")

    hub, watcher = get_live_feed()

    async def event_stream():
        # All acquisition lives INSIDE the generator: an async generator
        # that is never started executes no code on aclose(), so a client
        # disconnect landing between handler return and the first chunk
        # acquires nothing and therefore leaks nothing. (Acquiring in the
        # handler had exactly that leak: cleanup lived only in this
        # generator's finally, which never runs for an unstarted generator.)
        subscription = None
        try:
            # INVARIANT (ordering): subscribe -- synchronous, registers the
            # subscriber immediately, no await -- must precede
            # ensure_started. A racing last-disconnect then either sees this
            # subscriber in its count check (and skips stop) or stops the
            # watcher first, in which case ensure_started below restarts it
            # (its liveness prune discards the stopped tasks). The reverse
            # order leaves a window where the watcher is stopped after a
            # no-op ensure_started but before our subscribe: a live
            # connection fed by a dead watcher.
            subscription = hub.subscribe(interview_id=interview_id, project_id=project_id)
            await watcher.ensure_started()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    notification = await asyncio.wait_for(subscription.queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield format_sse_event(notification)
        finally:
            if subscription is not None:
                subscription.close()
                # INVARIANT (no await): nothing may suspend between close()
                # (the count decrement) and the subscriber_count check, so
                # the stop decision is always made on fresh state. Racing
                # new connections serialize on the watcher's asyncio.Lock
                # inside stop()/ensure_started(), and ensure_started's
                # liveness prune restarts a watcher this stop tears down.
                if hub.subscriber_count == 0:
                    await watcher.stop()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
