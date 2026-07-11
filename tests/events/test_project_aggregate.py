"""Project aggregate: domain invariants, locking, replay round-trip (M4.5b)."""

import pytest

from src.events.aggregates import Project
from src.events.project_events import project_aggregate_id

P = "default-project"


def _project() -> Project:
    return Project(project_aggregate_id(P))


def _canonicalize(p, cid="c1", surfaces=("acme corp",), method="deterministic"):
    p.canonicalize_entity(P, cid, "Acme Corp", "ORG", list(surfaces), method, 1.0)


class TestEntityInvariants:
    def test_canonicalize_and_state(self):
        p = _project()
        _canonicalize(p)
        assert p.project_id == P
        entry = p.canonical_entities["c1"]
        assert entry["surfaces"] == ["acme corp"]
        assert entry["locked"] is False
        assert p.canonical_for_surface("acme corp", "ORG") == "c1"
        assert p.canonical_for_surface("acme corp", "PERSON") is None

    def test_human_canonicalize_locks(self):
        p = _project()
        _canonicalize(p, method="human")
        assert p.canonical_entities["c1"]["locked"] is True

    def test_duplicate_canonical_id_raises(self):
        p = _project()
        _canonicalize(p)
        with pytest.raises(ValueError):
            _canonicalize(p)

    def test_surface_already_owned_raises(self):
        p = _project()
        _canonicalize(p)
        with pytest.raises(ValueError):
            p.canonicalize_entity(P, "c2", "Acme", "ORG", ["acme corp"], "deterministic", 1.0)

    def test_alias_add_and_locked_guard(self):
        p = _project()
        _canonicalize(p)
        p.add_entity_alias(P, "c1", "acme", "deterministic", 0.95)
        assert "acme" in p.canonical_entities["c1"]["surfaces"]
        p.add_entity_alias(P, "c1", "acme inc", "human", 1.0)  # human alias locks
        assert p.canonical_entities["c1"]["locked"] is True
        with pytest.raises(ValueError):  # engine may not touch locked
            p.add_entity_alias(P, "c1", "acme co", "deterministic", 0.99)

    def test_alias_unknown_canonical_raises(self):
        p = _project()
        with pytest.raises(ValueError):
            p.add_entity_alias(P, "nope", "x", "deterministic", 1.0)

    def test_merge_moves_surfaces_and_locks_both(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp",))
        p.canonicalize_entity(P, "c2", "Acme Inc", "ORG", ["acme inc"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        assert "acme inc" in p.canonical_entities["c1"]["surfaces"]
        assert p.canonical_entities["c1"]["locked"] is True
        assert p.canonical_entities["c2"]["merged_into"] == "c1"
        # merged-away canonical no longer owns surfaces
        assert p.canonical_for_surface("acme inc", "ORG") == "c1"

    def test_merge_unknown_or_merged_raises(self):
        p = _project()
        _canonicalize(p, "c1")
        with pytest.raises(ValueError):
            p.confirm_entity_merge(P, "c1", "ghost")
        p.canonicalize_entity(P, "c2", "B", "ORG", ["b"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        with pytest.raises(ValueError):  # c2 already merged away
            p.confirm_entity_merge(P, "c1", "c2")
        with pytest.raises(ValueError):  # self-merge
            p.confirm_entity_merge(P, "c1", "c1")

    def test_split_moves_surfaces_and_locks(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp", "acme berlin"))
        p.split_entity(P, "c1", ["acme berlin"], "c9", "Acme Berlin")
        assert p.canonical_entities["c1"]["surfaces"] == ["acme corp"]
        new = p.canonical_entities["c9"]
        assert new["surfaces"] == ["acme berlin"]
        assert new["locked"] is True and p.canonical_entities["c1"]["locked"] is True
        assert new["entity_type"] == "ORG"

    def test_split_requires_proper_subset(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp", "acme berlin"))
        with pytest.raises(ValueError):  # all surfaces
            p.split_entity(P, "c1", ["acme corp", "acme berlin"], "c9", "X")
        with pytest.raises(ValueError):  # not owned
            p.split_entity(P, "c1", ["ghost"], "c9", "X")


class TestPersonInvariants:
    def test_identify_and_link(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        assert p.link_for_speaker("i1", "s1") == "per1"

    def test_duplicate_person_raises(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        with pytest.raises(ValueError):
            p.identify_person(P, "per1", "Jane Doe")

    def test_link_unknown_person_or_double_link_raises(self):
        p = _project()
        with pytest.raises(ValueError):
            p.link_speaker_to_person(P, "i1", "s1", "ghost", "exact_name", 1.0)
        p.identify_person(P, "per1", "Jane Doe")
        p.identify_person(P, "per2", "Jane D")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        with pytest.raises(ValueError):
            p.link_speaker_to_person(P, "i1", "s1", "per2", "exact_name", 1.0)

    def test_unlink_blocks_engine_but_not_human(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.remove_person_link(P, "i1", "s1", "per1", note="wrong Jane")
        assert p.link_for_speaker("i1", "s1") is None
        assert ("i1", "s1") in p.blocked_links
        with pytest.raises(ValueError):  # engine re-link blocked
            p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.link_speaker_to_person(P, "i1", "s1", "per1", "human", 1.0)  # human may
        assert ("i1", "s1") not in p.blocked_links

    def test_unlink_nonexistent_raises(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        with pytest.raises(ValueError):
            p.remove_person_link(P, "i1", "s1", "per1", note=None)


class TestReplay:
    def test_replay_round_trip(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp",))
        p.add_entity_alias(P, "c1", "acme", "deterministic", 0.95)
        p.canonicalize_entity(P, "c2", "B", "ORG", ["b"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.remove_person_link(P, "i1", "s1", "per1", note="oops")
        events = p.get_uncommitted_events()
        assert events[0].aggregate_type == "Project"  # wire value

        fresh = _project()
        fresh.load_from_history(events)
        assert fresh.canonical_entities == p.canonical_entities
        assert fresh.persons == p.persons
        assert fresh.blocked_links == p.blocked_links
        assert fresh.project_id == P
        assert fresh.version == len(events) - 1
