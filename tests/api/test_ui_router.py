"""/ui/* router tests (M5.0 Task 1): 200 shapes, 404 legs, person-id equality.

Session-mocking idiom mirrors tests/api/test_queries_router.py: patch
Neo4jConnectionManager.get_session and the reader functions the router calls.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.events.project_events import person_id_for
from src.main import app
from src.resolution.candidates import normalize_name

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
