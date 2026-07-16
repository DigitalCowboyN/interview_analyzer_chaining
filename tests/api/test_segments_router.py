from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.events.aggregates import Interview
from src.events.envelope import Actor, ActorType
from src.events.interview_events import segment_id_for
from src.main import app

IID = "22222222-2222-2222-2222-222222222222"

SYSTEM_ACTOR = Actor(actor_type=ActorType.SYSTEM, user_id="enrichment")


def make_repo(load_result):
    repo = AsyncMock()
    repo.load = AsyncMock(return_value=load_result)
    repo.save = AsyncMock()
    return repo


def make_interview_with_segment(topic="Vendor choice"):
    """A committed Interview with one live segment."""
    interview = Interview(IID)
    interview.create(title="t", source="s", language="en", actor=SYSTEM_ACTOR)
    sid = segment_id_for(IID, 0)
    interview.record_segment(sid, topic, 0, 2, 0.9, actor=SYSTEM_ACTOR)
    interview.mark_events_as_committed()
    return interview, sid


def patch_session(found=1):
    """found: value returned for the interview-existence count record."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    exists_result = AsyncMock()
    exists_result.single = AsyncMock(return_value={"found": found})
    session.run = AsyncMock(return_value=exists_result)
    return patch(
        "src.api.routers.segments.Neo4jConnectionManager.get_session",
        new=AsyncMock(return_value=session),
    )


@pytest.fixture
def client():
    return TestClient(app)


# --- GET -----------------------------------------------------------------


def test_list_segments_returns_200_with_rows(client):
    rows = [
        {"segment_id": "s1", "topic": "Vendor choice", "confidence": 0.9,
         "start_index": 0, "end_index": 2},
    ]
    with patch_session(found=1), \
         patch("src.api.routers.segments.reader.segment_rows", new=AsyncMock(return_value=rows)):
        resp = client.get(f"/interviews/{IID}/segments")
    assert resp.status_code == 200
    assert resp.json() == {"segments": rows}


def test_list_segments_known_interview_zero_segments_returns_200_empty(client):
    with patch_session(found=1), \
         patch("src.api.routers.segments.reader.segment_rows", new=AsyncMock(return_value=[])):
        resp = client.get(f"/interviews/{IID}/segments")
    assert resp.status_code == 200
    assert resp.json() == {"segments": []}


def test_list_segments_unknown_interview_returns_404(client):
    with patch_session(found=0), \
         patch("src.api.routers.segments.reader.segment_rows", new=AsyncMock(return_value=[])) as segment_rows:
        resp = client.get(f"/interviews/{IID}/segments")
    assert resp.status_code == 404
    segment_rows.assert_not_awaited()


# --- DELETE ----------------------------------------------------------------


def test_remove_segment_happy_path_returns_202(client):
    interview, sid = make_interview_with_segment()
    repo = make_repo(interview)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/{sid}?reason=wrong+split")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["version"] == interview.version
    repo.save.assert_awaited_once_with(interview)
    assert interview.segments[sid]["removed"] is True
    last_event = interview.get_uncommitted_events()[-1]
    assert last_event.event_type == "SegmentRemoved"
    assert last_event.data == {"segment_id": sid, "reason": "wrong split"}


def test_remove_segment_actor_defaults_to_anonymous(client):
    interview, sid = make_interview_with_segment()

    real_method = interview.remove_segment
    captured = {}

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return real_method(*args, **kwargs)

    interview.remove_segment = spy
    repo = make_repo(interview)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/{sid}")
    assert resp.status_code == 202
    actor = captured["kwargs"]["actor"]
    assert actor.user_id == "anonymous"
    assert actor.actor_type == "human"


def test_remove_segment_actor_user_id_from_header(client):
    interview, sid = make_interview_with_segment()

    real_method = interview.remove_segment
    captured = {}

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return real_method(*args, **kwargs)

    interview.remove_segment = spy
    repo = make_repo(interview)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/{sid}", headers={"X-User-ID": "nathan"})
    assert resp.status_code == 202
    actor = captured["kwargs"]["actor"]
    assert actor.user_id == "nathan"
    assert actor.actor_type == "human"


def test_remove_segment_unknown_interview_returns_404(client):
    repo = make_repo(None)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/does-not-exist")
    assert resp.status_code == 404
    repo.save.assert_not_awaited()


def test_remove_segment_unknown_segment_returns_409(client):
    interview, _sid = make_interview_with_segment()
    repo = make_repo(interview)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/does-not-exist")
    assert resp.status_code == 409
    repo.save.assert_not_awaited()


def test_remove_already_removed_segment_returns_409(client):
    interview, sid = make_interview_with_segment()
    interview.remove_segment(sid, actor=SYSTEM_ACTOR)
    interview.mark_events_as_committed()
    repo = make_repo(interview)
    with patch("src.api.routers.segments.get_interview_repository", return_value=repo):
        resp = client.delete(f"/segments/{IID}/{sid}")
    assert resp.status_code == 409
    repo.save.assert_not_awaited()
