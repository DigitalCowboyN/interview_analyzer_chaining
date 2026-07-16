"""On-demand worklist suggestions: entity merge band + person first-name matches (M4.5b)."""

from unittest.mock import AsyncMock

import pytest

from src.events.aggregates import Project
from src.events.project_events import canonical_entity_id, person_id_for, project_aggregate_id
from src.resolution.suggestions import compute_suggestions

PID = "proj-1"
I1 = "11111111-1111-1111-1111-111111111111"
I2 = "22222222-2222-2222-2222-222222222222"

VECTORS = {
    "acme corp": [1.0, 0.0],
    "acme inc": [0.85, 0.53],   # cos ~ 0.85 -> suggest band [0.80, 0.92)
    "zeta ltd": [0.0, 1.0],     # orthogonal -> never in band with acme*
    "acme co": [1.0, 0.001],    # cos ~ 1.0 with "acme corp" -> auto band (>= 0.92)
}


class _FakeEmbedder:
    """Fixed vector per text, looked up by dict; unknown text -> zero vector."""

    async def embed(self, texts):
        return [VECTORS.get(t, [0.0, 0.0]) for t in texts]


class _FailingEmbedder:
    """Simulates an embedder outage."""

    async def embed(self, texts):
        raise RuntimeError("embedder unavailable")


def entity_rows(surfaces_with_type):
    return [
        {"surface": s, "entity_type": t, "mentions": m}
        for s, t, m in surfaces_with_type
    ]


def speaker_row(interview_id, speaker_id, display_name, handle="S1", provisional=False):
    return {
        "interview_id": interview_id, "speaker_id": speaker_id,
        "display_name": display_name, "handle": handle, "provisional": provisional,
    }


def make_session():
    return AsyncMock()


def patch_reader(monkeypatch, entity_rows_, speaker_rows_):
    async def fake_entity_surface_rows(session, project_id):
        return entity_rows_

    async def fake_speaker_rows(session, project_id):
        return speaker_rows_

    monkeypatch.setattr(
        "src.resolution.suggestions.entity_surface_rows", fake_entity_surface_rows
    )
    monkeypatch.setattr("src.resolution.suggestions.speaker_rows", fake_speaker_rows)


@pytest.mark.asyncio
async def test_entity_pair_in_suggest_band_yields_one_suggestion(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
        ("zeta ltd", "ORG", 1),
    ])
    patch_reader(monkeypatch, rows, [])

    result = await compute_suggestions(
        make_session(), None, PID, _FakeEmbedder(),
    )

    assert len(result["entity_merge_suggestions"]) == 1
    suggestion = result["entity_merge_suggestions"][0]
    assert suggestion["surviving_canonical_id"] == canonical_entity_id(PID, "acme corp", "ORG")
    assert suggestion["merged_canonical_id"] == canonical_entity_id(PID, "acme inc", "ORG")
    assert suggestion["surfaces_a"] == ["acme corp"]
    assert suggestion["surfaces_b"] == ["acme inc"]
    assert 0.80 <= suggestion["score"] < 0.92
    assert suggestion["band"] == "suggest"
    assert result["person_link_suggestions"] == []
    assert result["flags"] == []


@pytest.mark.asyncio
async def test_locked_canonical_filters_out_entity_suggestion(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    patch_reader(monkeypatch, rows, [])

    project = Project(project_aggregate_id(PID))
    cid = canonical_entity_id(PID, "acme corp", "ORG")
    project.canonicalize_entity(
        PID, cid, "Acme Corp", "ORG", ["acme corp"], "human", 1.0,
    )
    project.mark_events_as_committed()
    assert project.canonical_entities[cid]["locked"] is True

    result = await compute_suggestions(make_session(), project, PID, _FakeEmbedder())

    assert result["entity_merge_suggestions"] == []


@pytest.mark.asyncio
async def test_project_none_still_computes_suggestions(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    patch_reader(monkeypatch, rows, [])

    result = await compute_suggestions(make_session(), None, PID, _FakeEmbedder())

    assert len(result["entity_merge_suggestions"]) == 1


@pytest.mark.asyncio
async def test_person_first_name_only_match_yields_suggestion(monkeypatch):
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
        speaker_row(I1, "s3", "Jane Smith"),
    ]
    patch_reader(monkeypatch, [], speakers)

    result = await compute_suggestions(make_session(), None, PID, _FakeEmbedder())

    assert result["entity_merge_suggestions"] == []
    assert len(result["person_link_suggestions"]) == 1
    suggestion = result["person_link_suggestions"][0]
    assert suggestion["person_id"] == person_id_for(PID, "jane doe")
    assert suggestion["display_name"] == "Jane Doe"
    assert suggestion["interview_id"] == I1
    assert suggestion["speaker_id"] == "s3"
    assert suggestion["speaker_display_name"] == "Jane Smith"
    assert suggestion["reason"] == "first_name_match"


@pytest.mark.asyncio
async def test_blocked_pair_filters_out_person_suggestion(monkeypatch):
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
        speaker_row(I1, "s3", "Jane Smith"),
    ]
    patch_reader(monkeypatch, [], speakers)

    project = Project(project_aggregate_id(PID))
    jane_id = person_id_for(PID, "jane doe")
    project.identify_person(PID, jane_id, "Jane Doe")
    project.link_speaker_to_person(PID, I1, "s3", jane_id, "human", 1.0)
    project.remove_person_link(PID, I1, "s3", jane_id, note="wrong person")
    project.mark_events_as_committed()
    assert (I1, "s3") in project.blocked_links

    result = await compute_suggestions(make_session(), project, PID, _FakeEmbedder())

    assert result["person_link_suggestions"] == []


@pytest.mark.asyncio
async def test_already_linked_pair_filters_out_person_suggestion(monkeypatch):
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
        speaker_row(I1, "s3", "Jane Smith"),
    ]
    patch_reader(monkeypatch, [], speakers)

    project = Project(project_aggregate_id(PID))
    jane_id = person_id_for(PID, "jane doe")
    project.identify_person(PID, jane_id, "Jane Doe")
    project.link_speaker_to_person(PID, I1, "s3", jane_id, "human", 1.0)
    project.mark_events_as_committed()
    assert project.link_for_speaker(I1, "s3") == jane_id

    result = await compute_suggestions(make_session(), project, PID, _FakeEmbedder())

    assert result["person_link_suggestions"] == []


@pytest.mark.asyncio
async def test_embedder_failure_degrades_with_flag_but_keeps_person_suggestions(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
    ]
    patch_reader(monkeypatch, rows, speakers)

    result = await compute_suggestions(make_session(), None, PID, _FailingEmbedder())

    assert result["flags"] == ["embedding_unavailable"]
    assert result["entity_merge_suggestions"] == []
    assert len(result["person_link_suggestions"]) == 0  # no first-name-only mismatch here
    # person groups still computed independent of the embedder outage:
    assert "person_link_suggestions" in result


@pytest.mark.asyncio
async def test_auto_band_pair_surfaced_when_both_sides_existing_unlocked_canonicals(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme co", "ORG", 2),
    ])
    patch_reader(monkeypatch, rows, [])

    project = Project(project_aggregate_id(PID))
    cid_a = canonical_entity_id(PID, "acme corp", "ORG")
    cid_b = canonical_entity_id(PID, "acme co", "ORG")
    project.canonicalize_entity(PID, cid_a, "Acme Corp", "ORG", ["acme corp"], "deterministic", 1.0)
    project.canonicalize_entity(PID, cid_b, "Acme Co", "ORG", ["acme co"], "deterministic", 1.0)
    project.mark_events_as_committed()
    assert project.canonical_entities[cid_a]["locked"] is False
    assert project.canonical_entities[cid_b]["locked"] is False

    result = await compute_suggestions(make_session(), project, PID, _FakeEmbedder())

    auto_rows = [s for s in result["entity_merge_suggestions"] if s["band"] == "auto"]
    assert len(auto_rows) == 1
    auto = auto_rows[0]
    assert {auto["surviving_canonical_id"], auto["merged_canonical_id"]} == {cid_a, cid_b}


@pytest.mark.asyncio
async def test_auto_band_pair_not_surfaced_without_project(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme co", "ORG", 2),
    ])
    patch_reader(monkeypatch, rows, [])

    result = await compute_suggestions(make_session(), None, PID, _FakeEmbedder())

    auto_rows = [s for s in result["entity_merge_suggestions"] if s["band"] == "auto"]
    assert auto_rows == []


@pytest.mark.asyncio
async def test_happy_path_flags_empty(monkeypatch):
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    patch_reader(monkeypatch, rows, [])

    result = await compute_suggestions(make_session(), None, PID, _FakeEmbedder())

    assert result["flags"] == []
