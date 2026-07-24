"""/ui/* router tests (M5.0 Task 1): 200 shapes, 404 legs, person-id equality.
M5.1 Task 4 adds the SSE live-feed route tests below.

Session-mocking idiom mirrors tests/api/test_queries_router.py: patch
Neo4jConnectionManager.get_session and the reader functions the router calls.

SSE route tests use a real NotificationHub (already unit-tested in
tests/ui/test_notifications.py) with a stubbed EsdbWatcher (ensure_started /
stop as AsyncMocks) injected via `src.api.routers.ui.get_live_feed`, plus a
patched `Request.is_disconnected` so each test's StreamingResponse generator
terminates deterministically -- starlette's TestClient (like httpx's
ASGITransport) runs the whole ASGI app to completion before returning a
Response, so an SSE generator that loops forever would hang the test; making
`is_disconnected()` return True after a controlled number of iterations is
what lets the generator actually finish and the response body be inspected.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.events.project_events import person_id_for
from src.main import app
from src.resolution.candidates import normalize_name
from src.ui.notifications import Notification, NotificationHub

PID = "proj-1"
IID = "iv-1"
PERSON_ID = "person-1"


@pytest.fixture
def client():
    return TestClient(app)


def patch_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return patch(
        "src.api.routers.ui.Neo4jConnectionManager.get_session",
        new=AsyncMock(return_value=session),
    )


def patch_reader(**overrides):
    """Patch src.api.routers.ui.reader.<name> for every kwarg given."""
    patchers = [
        patch(f"src.api.routers.ui.reader.{name}", new=AsyncMock(return_value=value))
        for name, value in overrides.items()
    ]
    return patchers


class _MultiPatch:
    """Combine a list of mock.patch context managers (unittest.mock has no
    built-in for a dynamic list — ExitStack would work too, this is simpler)."""

    def __init__(self, patchers):
        self._patchers = patchers

    def __enter__(self):
        return [p.__enter__() for p in self._patchers]

    def __exit__(self, *exc):
        for p in reversed(self._patchers):
            p.__exit__(*exc)


# --- GET /ui/projects ---

def test_list_projects(client):
    rows = [{"project_id": PID, "interview_count": 2}]
    with patch_session(), _MultiPatch(patch_reader(project_rows=rows)):
        resp = client.get("/ui/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": rows}


# --- GET /ui/projects/{project_id}/interviews ---

def test_list_interviews(client):
    rows = [{"interview_id": IID, "title": "T", "created_at": "2026-01-01T00:00:00", "fragment_count": 3}]
    with patch_session(), _MultiPatch(
        patch_reader(project_exists=True, interview_rows=rows)
    ):
        resp = client.get(f"/ui/projects/{PID}/interviews")
    assert resp.status_code == 200
    assert resp.json() == {"interviews": rows}


def test_list_interviews_404_unknown_project(client):
    with patch_session(), _MultiPatch(patch_reader(project_exists=False)):
        resp = client.get(f"/ui/projects/{PID}/interviews")
    assert resp.status_code == 404


# --- GET /ui/interviews/{interview_id}/transcript ---

def test_transcript(client):
    header = {"interview_id": IID, "title": "T", "metadata": {"foo": "bar"}}
    line_rows = [{
        "fragment_id": "f1", "sequence_order": 0, "text": "Hi.", "edited": False,
        "speaker_id": "sp1", "speaker_display_name": "Alice",
        "person_id": "per1", "person_display_name": "Alice Jones",
        "utterance_id": "u1", "segment_id": "seg1", "segment_topic": "intro",
        "entities": [{"surface": "Acme", "entity_type": "ORG"}],
        "lens_items": [{"item_id": "li1", "lens": "persona", "node_type": "Trait",
                        "text": "curious", "confidence": 0.8, "human_locked": False}],
    }]
    with patch_session(), _MultiPatch(
        patch_reader(interview_header_row=header, transcript_line_rows=line_rows)
    ):
        resp = client.get(f"/ui/interviews/{IID}/transcript")
    assert resp.status_code == 200
    body = resp.json()
    assert body["interview_id"] == IID
    assert body["title"] == "T"
    assert body["metadata"] == {"foo": "bar"}
    line = body["lines"][0]
    assert line["fragment_id"] == "f1"
    assert line["sequence_order"] == 0
    assert line["speaker"] == {"speaker_id": "sp1", "display_name": "Alice"}
    assert line["person"] == {"person_id": "per1", "display_name": "Alice Jones"}
    assert line["utterance_id"] == "u1"
    assert line["segment"] == {"segment_id": "seg1", "topic": "intro"}
    assert line["entities"] == [{"surface": "Acme", "entity_type": "ORG"}]
    assert line["lens_items"][0]["human_locked"] is False
    assert line["edited"] is False


def test_transcript_null_speaker_person_segment(client):
    header = {"interview_id": IID, "title": "T", "metadata": {}}
    line_rows = [{
        "fragment_id": "f1", "sequence_order": 0, "text": "Hi.", "edited": True,
        "speaker_id": None, "speaker_display_name": None,
        "person_id": None, "person_display_name": None,
        "utterance_id": None, "segment_id": None, "segment_topic": None,
        "entities": [], "lens_items": [],
    }]
    with patch_session(), _MultiPatch(
        patch_reader(interview_header_row=header, transcript_line_rows=line_rows)
    ):
        resp = client.get(f"/ui/interviews/{IID}/transcript")
    assert resp.status_code == 200
    line = resp.json()["lines"][0]
    assert line["speaker"] is None
    assert line["person"] is None
    assert line["segment"] is None
    assert line["utterance_id"] is None
    assert line["edited"] is True


def test_transcript_404_unknown_interview(client):
    with patch_session(), _MultiPatch(patch_reader(interview_header_row=None)):
        resp = client.get(f"/ui/interviews/{IID}/transcript")
    assert resp.status_code == 404


# --- GET /ui/projects/{project_id}/personas ---

def test_list_personas(client):
    rows = [{
        "person_id": PERSON_ID, "display_name": "Alice Jones",
        "trait_count": 2, "goal_count": 1, "pain_point_count": 0, "quote_count": 1,
        "representative_quote": "I love it", "interview_ids": [IID],
    }]
    with patch_session(), _MultiPatch(
        patch_reader(project_exists=True, persona_card_rows=rows)
    ):
        resp = client.get(f"/ui/projects/{PID}/personas")
    assert resp.status_code == 200
    assert resp.json() == {"personas": rows}


def test_list_personas_404_unknown_project(client):
    with patch_session(), _MultiPatch(patch_reader(project_exists=False)):
        resp = client.get(f"/ui/projects/{PID}/personas")
    assert resp.status_code == 404


# --- GET /ui/personas/{project_id}/{person_id} ---

def test_persona_detail(client):
    detail_rows = [
        {"item_id": "li1", "node_type": "Trait", "text": "curious", "confidence": 0.8,
         "interview_id": IID, "interview_title": "T"},
        {"item_id": "li2", "node_type": "Goal", "text": "ship faster", "confidence": 0.7,
         "interview_id": IID, "interview_title": "T"},
        {"item_id": "li3", "node_type": "PainPoint", "text": "slow ci", "confidence": 0.6,
         "interview_id": IID, "interview_title": "T"},
        {"item_id": "li4", "node_type": "NotableQuote", "text": "I love it", "confidence": 0.9,
         "interview_id": IID, "interview_title": "T"},
    ]
    display_row = {"person_id": PERSON_ID, "display_name": "Alice Jones"}
    with patch_session(), _MultiPatch(
        patch_reader(
            persona_exists=True,
            person_display_name_row=display_row,
            persona_detail_rows=detail_rows,
        )
    ):
        resp = client.get(f"/ui/personas/{PID}/{PERSON_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["person_id"] == PERSON_ID
    assert body["display_name"] == "Alice Jones"
    assert len(body["dimensions"]["traits"]) == 1
    assert body["dimensions"]["traits"][0]["item_id"] == "li1"
    assert len(body["dimensions"]["goals"]) == 1
    assert len(body["dimensions"]["pain_points"]) == 1
    assert len(body["dimensions"]["notable_quotes"]) == 1


def test_persona_detail_404_unknown(client):
    with patch_session(), _MultiPatch(patch_reader(persona_exists=False)):
        resp = client.get(f"/ui/personas/{PID}/{PERSON_ID}")
    assert resp.status_code == 404


# --- GET /ui/projects/{project_id}/persons ---

def test_list_persons(client):
    rows = [{"person_id": PERSON_ID, "display_name": "Alice Jones", "speaker_count": 2, "interview_count": 2}]
    with patch_session(), _MultiPatch(
        patch_reader(project_exists=True, person_card_rows=rows)
    ):
        resp = client.get(f"/ui/projects/{PID}/persons")
    assert resp.status_code == 200
    assert resp.json() == {"persons": rows}


def test_list_persons_404_unknown_project(client):
    with patch_session(), _MultiPatch(patch_reader(project_exists=False)):
        resp = client.get(f"/ui/projects/{PID}/persons")
    assert resp.status_code == 404


# --- GET /ui/persons/{project_id}/{person_id} ---

def test_person_detail(client):
    links = [{"interview_id": IID, "interview_title": "T", "speaker_id": "sp1", "speaker_display_name": "Alice"}]
    with patch_session(), _MultiPatch(
        patch_reader(
            person_exists=True,
            person_display_name_row={"person_id": PERSON_ID, "display_name": "Alice Jones"},
            person_detail_rows=links,
            person_contributes_to_persona=True,
        )
    ):
        resp = client.get(f"/ui/persons/{PID}/{PERSON_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["person_id"] == PERSON_ID
    assert body["display_name"] == "Alice Jones"
    assert body["links"] == links
    assert body["contributes_to_persona"] is True


def test_person_detail_404_unknown(client):
    with patch_session(), _MultiPatch(patch_reader(person_exists=False)):
        resp = client.get(f"/ui/persons/{PID}/{PERSON_ID}")
    assert resp.status_code == 404


# --- GET /ui/projects/{project_id}/person-id ---

def test_person_id_derivation_matches_engine(client):
    with patch_session(), _MultiPatch(patch_reader(project_exists=True)):
        resp = client.get(f"/ui/projects/{PID}/person-id", params={"display_name": "Alice Jones"})
    assert resp.status_code == 200
    expected = person_id_for(PID, normalize_name("Alice Jones"))
    assert resp.json() == {"person_id": expected}


def test_person_id_404_unknown_project(client):
    with patch_session(), _MultiPatch(patch_reader(project_exists=False)):
        resp = client.get(f"/ui/projects/{PID}/person-id", params={"display_name": "Alice"})
    assert resp.status_code == 404


# --- GET /ui/streams/events (SSE live feed) ---


def make_watcher_stub():
    return SimpleNamespace(ensure_started=AsyncMock(), stop=AsyncMock())


def seed_next_subscription(hub: NotificationHub, notification: Notification):
    """Wrap hub.subscribe so the very next Subscription it creates already
    has `notification` sitting in its queue before the route's generator
    starts reading. This is what lets a single non-concurrent test call
    (TestClient runs the whole app to completion before returning) observe a
    real published notification without any timing dependency."""
    original_subscribe = hub.subscribe

    def seeded(*args, **kwargs):
        subscription = original_subscribe(*args, **kwargs)
        subscription.queue.put_nowait(notification)
        return subscription

    hub.subscribe = seeded


def test_stream_events_422_when_both_params_missing(client):
    resp = client.get("/ui/streams/events")
    assert resp.status_code == 422


def test_stream_events_first_notification_matches_contract_shape(client):
    hub = NotificationHub()
    seed_next_subscription(hub, Notification("transcript", interview_id=IID))
    watcher = make_watcher_stub()

    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False, True]),
    ):
        resp = client.get("/ui/streams/events", params={"interview_id": IID})

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = [frame for frame in resp.text.split("\n\n") if frame]
    assert frames[0].startswith("data: ")
    payload = json.loads(frames[0][len("data: "):])
    # Only non-None fields -- no stray "project_id": null.
    assert payload == {"surface": "transcript", "interview_id": IID}
    watcher.ensure_started.assert_awaited_once()


def test_stream_events_disconnect_closes_subscription_and_stops_watcher_at_zero(client):
    hub = NotificationHub()
    seed_next_subscription(hub, Notification("transcript", interview_id=IID))
    watcher = make_watcher_stub()

    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False, True]),
    ):
        resp = client.get("/ui/streams/events", params={"interview_id": IID})

    assert resp.status_code == 200
    assert hub.subscriber_count == 0
    watcher.stop.assert_awaited_once()


def test_stream_events_heartbeat_emitted_when_quiet(client):
    hub = NotificationHub()  # no seeding -- queue stays empty, every wait times out
    watcher = make_watcher_stub()

    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("src.api.routers.ui.HEARTBEAT_SECONDS", 0.01),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False, False, True]),
    ):
        resp = client.get("/ui/streams/events", params={"interview_id": IID})

    assert resp.status_code == 200
    assert resp.text.count(": keep-alive\n\n") == 2
    assert hub.subscriber_count == 0
    watcher.stop.assert_awaited_once()


def test_stream_events_heartbeat_default_is_fifteen_seconds():
    from src.api.routers.ui import HEARTBEAT_SECONDS

    assert HEARTBEAT_SECONDS == 15.0


def _bare_request():
    """Minimal starlette Request for driving stream_events directly. The
    generator-level tests below never let the route's loop touch it (either
    the generator is never started, or is_disconnected is patched)."""
    from starlette.requests import Request as StarletteRequest

    return StarletteRequest(
        {"type": "http", "method": "GET", "path": "/ui/streams/events", "headers": [], "query_string": b""}
    )


@pytest.mark.asyncio
async def test_stream_events_generator_never_started_leaks_nothing_and_next_connect_works():
    # A disconnect can land between handler return and the first chunk: the
    # ASGI server then aclose()s the response generator WITHOUT ever starting
    # it, and an unstarted async generator executes no code -- neither its
    # body nor its finally. All acquisition therefore lives inside the
    # generator: closing it unstarted must acquire nothing and leak nothing.
    from src.api.routers.ui import stream_events

    hub = NotificationHub()
    watcher = make_watcher_stub()

    with patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)):
        response = await stream_events(_bare_request(), interview_id=IID, project_id=None)
        await response.body_iterator.aclose()  # never started

    assert hub.subscriber_count == 0  # nothing subscribed => nothing leaked
    watcher.ensure_started.assert_not_awaited()
    watcher.stop.assert_not_awaited()

    # A subsequent connection on the same hub works end-to-end (count can
    # still reach zero => lazy stop still functions).
    seed_next_subscription(hub, Notification("transcript", interview_id=IID))
    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False, True]),
    ):
        response = await stream_events(_bare_request(), interview_id=IID, project_id=None)
        chunks = [chunk async for chunk in response.body_iterator]

    assert chunks and chunks[0].startswith("data: ")
    assert json.loads(chunks[0][len("data: "):]) == {"surface": "transcript", "interview_id": IID}
    assert hub.subscriber_count == 0
    watcher.ensure_started.assert_awaited_once()
    watcher.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_generator_aclosed_mid_stream_releases_subscription():
    # aclose() after the first yield models the ASGI server cancelling the
    # response mid-stream: GeneratorExit at the yield point must run the
    # finally -- subscription released, watcher stopped at zero subscribers.
    from src.api.routers.ui import stream_events

    hub = NotificationHub()
    seed_next_subscription(hub, Notification("transcript", interview_id=IID))
    watcher = make_watcher_stub()

    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False]),
    ):
        response = await stream_events(_bare_request(), interview_id=IID, project_id=None)
        generator = response.body_iterator
        first = await generator.__anext__()
        assert first.startswith("data: ")
        assert hub.subscriber_count == 1
        await generator.aclose()

    assert hub.subscriber_count == 0
    watcher.stop.assert_awaited_once()


def test_stream_events_disconnect_with_remaining_subscriber_keeps_watcher(client):
    hub = NotificationHub()
    # A second, still-connected subscriber (subscribed before the seeding
    # wrapper so it stays unseeded and untouched by the route).
    lingering = hub.subscribe(interview_id="iv-other", project_id=None)
    seed_next_subscription(hub, Notification("transcript", interview_id=IID))
    watcher = make_watcher_stub()

    with (
        patch("src.api.routers.ui.get_live_feed", return_value=(hub, watcher)),
        patch("starlette.requests.Request.is_disconnected", side_effect=[False, True]),
    ):
        resp = client.get("/ui/streams/events", params={"interview_id": IID})

    assert resp.status_code == 200
    assert hub.subscriber_count == 1  # only the lingering subscriber remains
    watcher.stop.assert_not_awaited()  # not the last subscriber => no stop
    lingering.close()
