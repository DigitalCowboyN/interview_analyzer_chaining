"""ResolutionEngine: entity + person resolution over two interviews (M4.5b)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.aggregates import Interview, Project
from src.events.project_events import canonical_entity_id, person_id_for, project_aggregate_id
from src.resolution.engine import ResolutionEngine

PID = "proj-1"
I1 = "11111111-1111-1111-1111-111111111111"
I2 = "22222222-2222-2222-2222-222222222222"

VECTORS = {
    "acme corp": [1.0, 0.0],
    "acme inc": [0.99, 0.14],   # cos(acme corp, acme inc) ~ 0.990 -> auto-union
    "zeta ltd": [0.0, 1.0],     # orthogonal -> never unions with acme*
}


class _FakeEmbedder:
    """Fixed vector per text, looked up by dict; unknown text -> zero vector."""

    async def embed(self, texts):
        return [VECTORS.get(t, [0.0, 0.0]) for t in texts]


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
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def make_interview(interview_id, participants=None):
    interview = Interview(interview_id)
    metadata = {"front_matter": {"participants": participants}} if participants is not None else {}
    interview.create(title="t", source="s", metadata=metadata)
    interview.mark_events_as_committed()
    return interview


def patch_engine(
    project, entity_rows_, speaker_rows_, interviews_by_id, config_dict=None,
):
    """Mirrors tests/lens/test_engine.py's fixture style, patched by engine-module path."""
    project_repo = MagicMock()
    project_repo.load = AsyncMock(return_value=project)
    project_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())

    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(side_effect=lambda iid: interviews_by_id.get(iid))

    session = make_session()

    patches = [
        patch("src.resolution.engine.get_project_repository", return_value=project_repo),
        patch("src.resolution.engine.get_interview_repository", return_value=interview_repo),
        patch("src.resolution.engine.Neo4jConnectionManager.get_session",
              new=AsyncMock(return_value=session)),
        patch("src.resolution.engine.entity_surface_rows", new=AsyncMock(return_value=entity_rows_)),
        patch("src.resolution.engine.speaker_rows", new=AsyncMock(return_value=speaker_rows_)),
        patch.object(ResolutionEngine, "_build_embedder", return_value=_FakeEmbedder()),
    ]
    return patches, project_repo, interview_repo


def apply_patches(patches):
    for p in patches:
        p.start()
    return patches


def stop_patches(patches):
    for p in patches:
        p.stop()


@pytest.mark.asyncio
async def test_first_run_two_interviews_canonicalizes_entities_and_links_persons():
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("the acme corp", "ORG", 1),
        ("acme inc", "ORG", 2),
        ("zeta ltd", "ORG", 1),
    ])
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
    ]
    interviews = {I1: make_interview(I1), I2: make_interview(I2)}
    project = Project(project_aggregate_id(PID))

    events = []
    patches, project_repo, _ = patch_engine(project, rows, speakers, interviews)
    project_repo.save = AsyncMock(
        side_effect=lambda a, **k: (events.extend(a.get_uncommitted_events()), a.mark_events_as_committed())
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    # exact-group ("acme corp"/"the acme corp") unions with "acme inc" via embedding
    # into one cluster; "zeta ltd" is its own cluster -> 2 canonicals.
    assert result.entities_canonicalized == 2
    assert result.aliases_added == 0
    assert result.entity_suggestions == 0
    assert result.skipped_locked == 0

    acme_cid = canonical_entity_id(PID, "acme corp", "ORG")
    assert acme_cid in project.canonical_entities
    acme_entry = project.canonical_entities[acme_cid]
    assert sorted(acme_entry["surfaces"]) == ["acme corp", "acme inc", "the acme corp"]
    assert acme_entry["locked"] is False

    zeta_cid = canonical_entity_id(PID, "zeta ltd", "ORG")
    assert zeta_cid in project.canonical_entities
    assert project.canonical_entities[zeta_cid]["surfaces"] == ["zeta ltd"]

    # Jane Doe in both interviews -> one PersonIdentified + two SpeakerLinkedToPerson.
    assert result.persons_identified == 1
    assert result.speakers_linked == 2
    assert result.person_suggestions == 0
    assert result.skipped_blocked == 0

    jane_id = person_id_for(PID, "jane doe")
    assert jane_id in project.persons
    person = project.persons[jane_id]
    assert person["display_name"] == "Jane Doe"
    assert sorted(person["links"]) == [[I1, "s1"], [I2, "s2"]]
    for pid, sid in [(I1, "s1"), (I2, "s2")]:
        link_pid = project.link_for_speaker(pid, sid)
        assert link_pid == jane_id

    # Verify that both SpeakerLinkedToPerson events used exact_name method.
    speaker_link_events = [e for e in events if e.event_type == "SpeakerLinkedToPerson"]
    assert len(speaker_link_events) == 2
    for event in speaker_link_events:
        assert event.data["method"] == "exact_name"

    project_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_second_run_identical_inputs_emits_nothing():
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("the acme corp", "ORG", 1),
        ("acme inc", "ORG", 2),
        ("zeta ltd", "ORG", 1),
    ])
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
    ]
    interviews = {I1: make_interview(I1), I2: make_interview(I2)}

    # Run 1: capture the events it emits (repo.save marks them committed).
    project1 = Project(project_aggregate_id(PID))
    events = []
    patches1, project_repo1, _ = patch_engine(project1, rows, speakers, interviews)
    project_repo1.save = AsyncMock(
        side_effect=lambda a, **k: (events.extend(a.get_uncommitted_events()), a.mark_events_as_committed())
    )
    with patches1[0], patches1[1], patches1[2], patches1[3], patches1[4], patches1[5]:
        await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)
    assert events  # sanity: run 1 actually produced events

    # Run 2: replay run 1's events into a FRESH Project (per the brief) and
    # confirm the second run over identical inputs is a pure no-op.
    replayed = Project(project_aggregate_id(PID))
    replayed.load_from_history(events)
    replayed.mark_events_as_committed()

    patches2, project_repo2, _ = patch_engine(replayed, rows, speakers, interviews)
    with patches2[0], patches2[1], patches2[2], patches2[3], patches2[4], patches2[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    assert result.entities_canonicalized == 0
    assert result.aliases_added == 0
    assert result.persons_identified == 0
    assert result.speakers_linked == 0
    assert result.skipped_locked == 0
    assert result.skipped_blocked == 0
    project_repo2.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_locked_canonical_with_new_surface_skips_and_counts():
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    interviews = {}
    project = Project(project_aggregate_id(PID))
    cid = canonical_entity_id(PID, "acme corp", "ORG")
    project.canonicalize_entity(
        PID, cid, "Acme Corp", "ORG", ["acme corp"], "human", 1.0,
    )
    project.mark_events_as_committed()
    assert project.canonical_entities[cid]["locked"] is True

    patches, project_repo, _ = patch_engine(project, rows, [], interviews)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    assert result.aliases_added == 0
    assert result.entities_canonicalized == 0
    assert result.skipped_locked == 1
    assert "acme inc" not in project.canonical_entities[cid]["surfaces"]
    project_repo.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_pair_skips_and_counts():
    rows = []
    speakers = [
        speaker_row(I1, "s1", "Jane Doe"),
        speaker_row(I2, "s2", "Jane Doe"),
    ]
    interviews = {I1: make_interview(I1), I2: make_interview(I2)}
    project = Project(project_aggregate_id(PID))
    jane_id = person_id_for(PID, "jane doe")
    project.identify_person(PID, jane_id, "Jane Doe")
    project.link_speaker_to_person(PID, I1, "s1", jane_id, "exact_name", 1.0)
    project.remove_person_link(PID, I1, "s1", jane_id, note="wrong")
    project.mark_events_as_committed()
    assert (I1, "s1") in project.blocked_links

    patches, project_repo, _ = patch_engine(project, rows, speakers, interviews)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    assert result.skipped_blocked == 1
    assert result.speakers_linked == 1  # only i2/s2 links
    assert project.link_for_speaker(I1, "s1") is None
    assert project.link_for_speaker(I2, "s2") == jane_id
    project_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_two_existing_canonicals_in_one_cluster_yields_suggestion_not_merge():
    rows = entity_rows([
        ("acme corp", "ORG", 3),
        ("acme inc", "ORG", 2),
    ])
    interviews = {}
    project = Project(project_aggregate_id(PID))
    cid_a = canonical_entity_id(PID, "acme corp", "ORG")
    cid_b = canonical_entity_id(PID, "acme inc", "ORG")
    project.canonicalize_entity(PID, cid_a, "Acme Corp", "ORG", ["acme corp"], "deterministic", 1.0)
    project.canonicalize_entity(PID, cid_b, "Acme Inc", "ORG", ["acme inc"], "deterministic", 1.0)
    project.mark_events_as_committed()

    patches, project_repo, _ = patch_engine(project, rows, [], interviews)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    assert result.entity_suggestions == 1
    assert result.entities_canonicalized == 0
    assert result.aliases_added == 0
    # Neither canonical touched: no EntityMergeConfirmed, no alias.
    assert project.canonical_entities[cid_a]["surfaces"] == ["acme corp"]
    assert project.canonical_entities[cid_b]["surfaces"] == ["acme inc"]
    assert project.canonical_entities[cid_a]["merged_into"] is None
    assert project.canonical_entities[cid_b]["merged_into"] is None
    project_repo.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_front_matter_participant_single_speaker_uses_front_matter_method():
    rows = []
    speakers = [speaker_row(I1, "s1", "Jane Doe")]
    interviews = {I1: make_interview(I1, participants=["Jane Doe"])}
    project = Project(project_aggregate_id(PID))

    events = []
    patches, project_repo, _ = patch_engine(project, rows, speakers, interviews)
    project_repo.save = AsyncMock(
        side_effect=lambda a, **k: (events.extend(a.get_uncommitted_events()), a.mark_events_as_committed())
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)

    assert result.persons_identified == 1
    assert result.speakers_linked == 1
    jane_id = person_id_for(PID, "jane doe")
    person = project.persons[jane_id]
    assert person["links"] == [[I1, "s1"]]

    # Verify that SpeakerLinkedToPerson event used front_matter method.
    speaker_link_events = [e for e in events if e.event_type == "SpeakerLinkedToPerson"]
    assert len(speaker_link_events) == 1
    assert speaker_link_events[0].data["method"] == "front_matter"

    project_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_project_raises_value_error():
    project = Project(project_aggregate_id(PID))
    patches, _, _ = patch_engine(project, [], [], {})
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with pytest.raises(ValueError):
            await ResolutionEngine(config_dict={"resolution": {}}).apply(PID)
