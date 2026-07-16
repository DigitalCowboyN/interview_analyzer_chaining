from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def client():
    return TestClient(app)


def patch_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return patch(
        "src.api.routers.queries.Neo4jConnectionManager.get_session",
        new=AsyncMock(return_value=session),
    )


def test_lens_items_endpoint_splits_fields(client):
    row = {"item_id": "i1", "node_type": "Decision", "lens_version": 1, "confidence": 0.9,
           "model": "haiku", "provider": "anthropic", "locked": False,
           "props": {"item_id": "i1", "lens": "meeting_minutes", "text": "Go with X"},
           "speaker_links": [], "supporting_fragment_ids": []}
    with patch_session(), \
         patch("src.api.routers.queries.reader.lens_item_rows", new=AsyncMock(return_value=[row])):
        resp = client.get(f"/interviews/{IID}/lenses/meeting_minutes/items")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["fields"] == {"text": "Go with X"}   # reserved props stripped
    assert "props" not in item


def test_worklist_endpoint(client):
    result = {"lens_items": [], "claims": []}
    with patch_session(), \
         patch("src.api.routers.queries.reader.worklist_rows", new=AsyncMock(return_value=result)), \
         patch("src.api.routers.queries.compute_suggestions", new=AsyncMock()) as suggestions_mock:
        resp = client.get("/review/worklist?threshold=0.5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["lens_items"] == [] and body["claims"] == []
    assert body["entity_merge_suggestions"] == []
    assert body["person_link_suggestions"] == []
    assert body["flags"] == []
    suggestions_mock.assert_not_awaited()


def test_worklist_endpoint_with_project_id_includes_suggestions(client):
    result = {"lens_items": [], "claims": []}
    suggestions = {
        "entity_merge_suggestions": [{"surviving_canonical_id": "a"}],
        "person_link_suggestions": [{"person_id": "b"}],
        "flags": ["embedding_unavailable"],
    }
    with patch_session(), \
         patch("src.api.routers.queries.reader.worklist_rows", new=AsyncMock(return_value=result)), \
         patch("src.api.routers.queries.get_project_repository") as get_repo_mock, \
         patch("src.api.routers.queries.compute_suggestions",
               new=AsyncMock(return_value=suggestions)) as suggestions_mock:
        get_repo_mock.return_value.load = AsyncMock(return_value=None)
        resp = client.get("/review/worklist?project_id=proj-1&threshold=0.5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_merge_suggestions"] == suggestions["entity_merge_suggestions"]
    assert body["person_link_suggestions"] == suggestions["person_link_suggestions"]
    assert body["flags"] == suggestions["flags"]
    suggestions_mock.assert_awaited_once()


def test_rollup_endpoint(client):
    with patch_session(), \
         patch("src.api.routers.queries.reader.speaker_rollup_rows", new=AsyncMock(return_value=[])):
        resp = client.get("/speakers/rollup?name=Alice")
    assert resp.status_code == 200 and resp.json() == {"speakers": []}
