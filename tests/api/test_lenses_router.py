from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"
ITEM = "88888888-8888-8888-8888-888888888801"


@pytest.fixture
def client():
    return TestClient(app)


def make_repo(load_result):
    repo = MagicMock()
    repo.load = AsyncMock(return_value=load_result)
    repo.save = AsyncMock()
    return repo


def test_override_returns_202_with_actor_from_header(client):
    interview = MagicMock()
    interview.version = 7
    repo = make_repo(interview)
    with patch("src.api.routers.lenses.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/lenses/{IID}/items/{ITEM}/override",
            json={"fields_overridden": {"text": "Go with Y"}, "note": "fixed"},
            headers={"X-User-ID": "nathan"},
        )
    assert resp.status_code == 202
    assert resp.json()["version"] == 7
    actor = interview.override_lens_extraction.call_args.kwargs["actor"]
    assert actor.user_id == "nathan"


def test_override_missing_interview_404(client):
    with patch("src.api.routers.lenses.get_interview_repository", return_value=make_repo(None)):
        resp = client.post(
            f"/lenses/{IID}/items/{ITEM}/override", json={"fields_overridden": {"text": "x"}}
        )
    assert resp.status_code == 404


def test_override_unknown_item_409(client):
    interview = MagicMock()
    interview.override_lens_extraction.side_effect = ValueError("Unknown lens item")
    with patch("src.api.routers.lenses.get_interview_repository", return_value=make_repo(interview)):
        resp = client.post(
            f"/lenses/{IID}/items/{ITEM}/override", json={"fields_overridden": {"text": "x"}}
        )
    assert resp.status_code == 409
