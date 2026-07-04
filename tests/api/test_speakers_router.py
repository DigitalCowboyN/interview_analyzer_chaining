from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"
U1 = "55555555-5555-5555-5555-555555555555"


@pytest.fixture
def client():
    return TestClient(app)


def make_interview_mock(version=5):
    interview = MagicMock()
    interview.version = version
    return interview


def make_repo(load_result):
    repo = MagicMock()
    repo.load = AsyncMock(return_value=load_result)
    repo.save = AsyncMock()
    return repo


def test_rename_speaker_returns_202(client):
    interview = make_interview_mock()
    repo = make_repo(interview)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/{SP1}/rename", json={"new_display_name": "Dana"}
        )
    assert resp.status_code == 202
    assert resp.json()["version"] == 5
    interview.rename_speaker.assert_called_once()
    repo.save.assert_awaited_once()


def test_rename_unknown_interview_returns_404(client):
    repo = make_repo(None)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/{SP1}/rename", json={"new_display_name": "Dana"}
        )
    assert resp.status_code == 404


def test_merge_speaker_domain_error_returns_409(client):
    interview = make_interview_mock()
    interview.merge_speakers.side_effect = ValueError("Cannot merge a speaker into itself")
    repo = make_repo(interview)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/merge",
            json={"surviving_speaker_id": SP1, "merged_speaker_id": SP1},
        )
    assert resp.status_code == 409


def test_reattribute_fragment_returns_202(client):
    sentence = MagicMock()
    sentence.version = 2
    interview_repo = make_repo(make_interview_mock())
    repo = make_repo(sentence)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=interview_repo), \
         patch("src.api.routers.speakers.get_sentence_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/fragments/3/reattribute", json={"new_speaker_id": SP2}
        )
    assert resp.status_code == 202
    sentence.reattribute_speaker.assert_called_once()


def test_reattribute_missing_fragment_returns_404(client):
    interview_repo = make_repo(make_interview_mock())
    repo = make_repo(None)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=interview_repo), \
         patch("src.api.routers.speakers.get_sentence_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/fragments/3/reattribute", json={"new_speaker_id": SP2}
        )
    assert resp.status_code == 404


def test_reattribute_missing_interview_returns_404(client):
    interview_repo = make_repo(None)
    sentence_repo = make_repo(MagicMock())
    with patch("src.api.routers.speakers.get_interview_repository", return_value=interview_repo), \
         patch("src.api.routers.speakers.get_sentence_repository", return_value=sentence_repo):
        resp = client.post(
            f"/speakers/{IID}/fragments/3/reattribute", json={"new_speaker_id": SP2}
        )
    assert resp.status_code == 404
    sentence_repo.load.assert_not_awaited()


def test_rename_domain_error_returns_409(client):
    interview = make_interview_mock()
    interview.rename_speaker.side_effect = ValueError("Unknown speaker")
    repo = make_repo(interview)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/{SP1}/rename", json={"new_display_name": "Dana"}
        )
    assert resp.status_code == 409


def test_split_speaker_creates_and_reattributes(client):
    interview = make_interview_mock()
    interview_repo = make_repo(interview)
    sentence = MagicMock()
    sentence.version = 1
    sentence_repo = make_repo(sentence)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=interview_repo), \
         patch("src.api.routers.speakers.get_sentence_repository", return_value=sentence_repo):
        resp = client.post(
            f"/speakers/{IID}/split",
            json={
                "new_handle": "S3",
                "new_display_name": "Third Voice",
                "fragment_indices": [1, 2],
            },
        )
    assert resp.status_code == 202
    interview.add_speaker.assert_called_once()
    assert sentence.reattribute_speaker.call_count == 2


def test_remove_stitch_returns_202(client):
    interview = make_interview_mock()
    repo = make_repo(interview)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.request(
            "DELETE", f"/stitches/{IID}/{U1}", json={"reason": "not a continuation"}
        )
    assert resp.status_code == 202
    interview.remove_stitch.assert_called_once()


def test_split_fragment_404_after_first_success_still_persists_speaker(client):
    # Append-only log: speaker creation is saved before fragment loop; a
    # missing fragment mid-loop returns 404 but does not roll back the speaker
    # (documented partial-application semantics).
    interview = make_interview_mock()
    interview_repo = make_repo(interview)
    sentence = MagicMock()
    sentence.version = 1
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=[sentence, None])
    sentence_repo.save = AsyncMock()
    with patch("src.api.routers.speakers.get_interview_repository", return_value=interview_repo), \
         patch("src.api.routers.speakers.get_sentence_repository", return_value=sentence_repo):
        resp = client.post(
            f"/speakers/{IID}/split",
            json={
                "new_handle": "S3",
                "new_display_name": "Third Voice",
                "fragment_indices": [1, 2],
            },
        )
    assert resp.status_code == 404
    interview_repo.save.assert_awaited_once()  # speaker creation persisted
    sentence.reattribute_speaker.assert_called_once()  # first fragment applied
