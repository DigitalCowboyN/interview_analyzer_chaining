from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.events.aggregates import Project
from src.events.envelope import Actor, ActorType
from src.events.project_events import canonical_entity_id, person_id_for, project_aggregate_id
from src.main import app
from src.resolution.candidates import normalize_surface

PROJECT_ID = "proj-1"
AGG_ID = project_aggregate_id(PROJECT_ID)

SYSTEM_ACTOR = Actor(actor_type=ActorType.SYSTEM, user_id="system")


def make_repo(load_result):
    repo = MagicMock()
    repo.load = AsyncMock(return_value=load_result)
    repo.save = AsyncMock()
    return repo


def make_project_with_two_entities(entity_type="PERSON"):
    """A committed Project with two live canonical entities of the same type."""
    project = Project(AGG_ID)
    cid_a = canonical_entity_id(PROJECT_ID, normalize_surface("Alice"), entity_type)
    cid_b = canonical_entity_id(PROJECT_ID, normalize_surface("Bob"), entity_type)
    project.canonicalize_entity(
        PROJECT_ID, cid_a, "Alice", entity_type, ["Alice"], "deterministic", 0.9,
        actor=SYSTEM_ACTOR,
    )
    project.canonicalize_entity(
        PROJECT_ID, cid_b, "Bob", entity_type, ["Bob"], "deterministic", 0.9,
        actor=SYSTEM_ACTOR,
    )
    project.mark_events_as_committed()
    return project, cid_a, cid_b


def make_project_with_splittable_entity(entity_type="PERSON"):
    """A committed Project with one canonical entity that owns 2+ surfaces."""
    project = Project(AGG_ID)
    cid = canonical_entity_id(PROJECT_ID, normalize_surface("Alice"), entity_type)
    project.canonicalize_entity(
        PROJECT_ID, cid, "Alice", entity_type, ["Alice", "Al"], "deterministic", 0.9,
        actor=SYSTEM_ACTOR,
    )
    project.mark_events_as_committed()
    return project, cid


def make_project_with_person(display_name="Alice"):
    project = Project(AGG_ID)
    pid = person_id_for(PROJECT_ID, normalize_surface(display_name))
    project.identify_person(PROJECT_ID, pid, display_name, actor=SYSTEM_ACTOR)
    project.mark_events_as_committed()
    return project, pid


def make_project_with_linked_person(interview_id, speaker_id, display_name="Alice"):
    project, pid = make_project_with_person(display_name)
    project.link_speaker_to_person(
        PROJECT_ID, interview_id, speaker_id, pid, "human", 1.0, actor=SYSTEM_ACTOR,
    )
    project.mark_events_as_committed()
    return project, pid


@pytest.fixture
def client():
    return TestClient(app)


# --- merge ---------------------------------------------------------------


def test_merge_happy_path_returns_202(client):
    project, cid_a, cid_b = make_project_with_two_entities()
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/entities/merge",
            json={"surviving_canonical_id": cid_a, "merged_canonical_id": cid_b},
            headers={"X-User-ID": "nathan"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["version"] == project.version
    repo.save.assert_awaited_once_with(project)
    assert project.canonical_entities[cid_b]["merged_into"] == cid_a


def test_merge_unknown_canonical_returns_409(client):
    project, cid_a, _cid_b = make_project_with_two_entities()
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/entities/merge",
            json={"surviving_canonical_id": cid_a, "merged_canonical_id": "does-not-exist"},
        )
    assert resp.status_code == 409
    repo.save.assert_not_awaited()


def test_merge_missing_project_returns_404(client):
    repo = make_repo(None)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/entities/merge",
            json={"surviving_canonical_id": "a", "merged_canonical_id": "b"},
        )
    assert resp.status_code == 404
    repo.save.assert_not_awaited()


# --- split -----------------------------------------------------------------


def test_split_happy_path_returns_202(client):
    project, cid = make_project_with_splittable_entity()
    repo = make_repo(project)
    entity_type = project.canonical_entities[cid]["entity_type"]
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/entities/{cid}/split",
            json={"surfaces": ["Al"], "new_name": "Al"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["version"] == project.version
    expected_new_id = canonical_entity_id(PROJECT_ID, normalize_surface("Al"), entity_type)
    assert expected_new_id in project.canonical_entities
    repo.save.assert_awaited_once_with(project)


def test_split_unknown_canonical_returns_404(client):
    project, _cid = make_project_with_splittable_entity()
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/entities/does-not-exist/split",
            json={"surfaces": ["Al"], "new_name": "Al"},
        )
    assert resp.status_code == 404
    repo.save.assert_not_awaited()


# --- link --------------------------------------------------------------


def test_link_existing_person_happy_path_returns_202(client):
    interview_id = "interview-1"
    speaker_id = "SPEAKER_00"
    project, pid = make_project_with_person("Alice")
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/link",
            json={"interview_id": interview_id, "speaker_id": speaker_id},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["version"] == project.version
    assert [interview_id, speaker_id] in project.persons[pid]["links"]
    repo.save.assert_awaited_once_with(project)


def test_link_unknown_person_with_display_name_creates_person(client):
    interview_id = "interview-1"
    speaker_id = "SPEAKER_00"
    project = Project(AGG_ID)
    project.mark_events_as_committed()
    repo = make_repo(project)
    pid = person_id_for(PROJECT_ID, normalize_surface("New Person"))
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/link",
            json={
                "interview_id": interview_id,
                "speaker_id": speaker_id,
                "display_name": "New Person",
            },
        )
    assert resp.status_code == 202
    assert pid in project.persons
    assert project.persons[pid]["display_name"] == "New Person"
    assert [interview_id, speaker_id] in project.persons[pid]["links"]
    repo.save.assert_awaited_once_with(project)


def test_link_unknown_person_without_display_name_returns_404(client):
    project = Project(AGG_ID)
    project.mark_events_as_committed()
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/does-not-exist/link",
            json={"interview_id": "interview-1", "speaker_id": "SPEAKER_00"},
        )
    assert resp.status_code == 404
    repo.save.assert_not_awaited()


def test_link_blocked_pair_via_human_actor_succeeds(client):
    interview_id = "interview-1"
    speaker_id = "SPEAKER_00"
    project, pid = make_project_with_linked_person(interview_id, speaker_id, "Alice")
    # Remove the link (human) so the pair becomes blocked from auto-linking.
    project.remove_person_link(PROJECT_ID, interview_id, speaker_id, pid, actor=SYSTEM_ACTOR)
    project.mark_events_as_committed()
    assert (interview_id, speaker_id) in project.blocked_links

    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/link",
            json={"interview_id": interview_id, "speaker_id": speaker_id},
        )
    assert resp.status_code == 202
    assert [interview_id, speaker_id] in project.persons[pid]["links"]
    repo.save.assert_awaited_once_with(project)


def test_link_method_is_human_confidence_one(client):
    interview_id = "interview-1"
    speaker_id = "SPEAKER_00"
    project, pid = make_project_with_person("Alice")

    real_method = project.link_speaker_to_person
    captured = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_method(*args, **kwargs)

    project.link_speaker_to_person = spy
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/link",
            json={"interview_id": interview_id, "speaker_id": speaker_id},
        )
    assert resp.status_code == 202
    assert captured["args"][4] == "human"
    assert captured["args"][5] == 1.0


# --- unlink ------------------------------------------------------------


def test_unlink_happy_path_returns_202(client):
    interview_id = "interview-1"
    speaker_id = "SPEAKER_00"
    project, pid = make_project_with_linked_person(interview_id, speaker_id, "Alice")
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/unlink",
            json={"interview_id": interview_id, "speaker_id": speaker_id, "note": "wrong pairing"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["version"] == project.version
    assert [interview_id, speaker_id] not in project.persons[pid]["links"]
    repo.save.assert_awaited_once_with(project)


def test_unlink_non_linked_pair_returns_409(client):
    project, pid = make_project_with_person("Alice")
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/unlink",
            json={"interview_id": "interview-1", "speaker_id": "SPEAKER_00"},
        )
    assert resp.status_code == 409
    repo.save.assert_not_awaited()


def test_unlink_missing_project_returns_404(client):
    repo = make_repo(None)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/some-pid/unlink",
            json={"interview_id": "interview-1", "speaker_id": "SPEAKER_00"},
        )
    assert resp.status_code == 404


# --- actor -------------------------------------------------------------


def test_missing_x_user_id_defaults_to_anonymous(client):
    project, pid = make_project_with_person("Alice")

    real_method = project.link_speaker_to_person
    captured = {}

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return real_method(*args, **kwargs)

    project.link_speaker_to_person = spy
    repo = make_repo(project)
    with patch("src.api.routers.resolution.get_project_repository", return_value=repo):
        resp = client.post(
            f"/resolution/{PROJECT_ID}/persons/{pid}/link",
            json={"interview_id": "interview-1", "speaker_id": "SPEAKER_00"},
        )
    assert resp.status_code == 202
    actor = captured["kwargs"]["actor"]
    assert actor.user_id == "anonymous"
    assert actor.actor_type == "human"
