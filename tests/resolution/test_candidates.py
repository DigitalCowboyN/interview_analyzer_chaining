"""Deterministic candidate logic: normalization, grouping, thresholds (M4.5b)."""

import pytest

from src.resolution.candidates import (
    cosine,
    embedding_pairs,
    exact_groups,
    normalize_name,
    normalize_surface,
    person_groups,
    representative,
)


@pytest.mark.parametrize("raw,expected", [
    ("Acme Corp", "acme corp"),
    ("  The Acme   Corp ", "acme corp"),      # article + whitespace collapse
    ("acme's", "acme"),                        # possessive
    ("engineers'", "engineer"),                # trailing-apostrophe possessive + plural
    ("engineers", "engineer"),                 # naive plural fold
    ("bus", "bus"),                            # too short to deplural (len<=3)
    ("boss", "boss"),                          # 'ss' never folded
    ("An Apple", "apple"),
    ("A", "a"),                                # bare article is not stripped to empty
])
def test_normalize_surface(raw, expected):
    assert normalize_surface(raw) == expected


def test_normalize_name_casefold_and_collapse():
    assert normalize_name("  Jane   DOE ") == "jane doe"


def test_exact_groups_by_normalized_and_type():
    rows = [
        {"surface": "acme corp", "entity_type": "ORG", "mentions": 2},
        {"surface": "the acme corp", "entity_type": "ORG", "mentions": 1},
        {"surface": "acme corp", "entity_type": "PERSON", "mentions": 1},
    ]
    groups = exact_groups(rows)
    assert set(groups) == {("acme corp", "ORG"), ("acme corp", "PERSON")}
    assert len(groups[("acme corp", "ORG")]) == 2


def test_representative_prefers_mentions_then_lexicographic():
    rows = [
        {"surface": "acme", "entity_type": "ORG", "mentions": 1},
        {"surface": "acme corp", "entity_type": "ORG", "mentions": 3},
    ]
    assert representative(rows) == "acme corp"
    tied = [
        {"surface": "b", "entity_type": "ORG", "mentions": 2},
        {"surface": "a", "entity_type": "ORG", "mentions": 2},
    ]
    assert representative(tied) == "a"


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector guarded


def test_embedding_pairs_thresholds_and_type_isolation():
    keys = [("acme corp", "ORG"), ("acme inc", "ORG"), ("acme", "PERSON"), ("zeta", "ORG")]
    vectors = {
        ("acme corp", "ORG"): [1.0, 0.0],
        ("acme inc", "ORG"): [0.99, 0.14],   # cos ~0.990 -> auto
        ("acme", "PERSON"): [1.0, 0.0],       # different type -> never compared
        ("zeta", "ORG"): [0.62, 0.78],        # cos ~0.62 vs corp -> ignored
    }
    unions, suggestions = embedding_pairs(keys, vectors, auto_thr=0.92, suggest_thr=0.80)
    assert (("acme corp", "ORG"), ("acme inc", "ORG")) in unions
    assert all(a[1] == b[1] for a, b in unions)
    assert suggestions == []


def test_embedding_pairs_suggest_band():
    keys = [("a", "ORG"), ("b", "ORG")]
    vectors = {("a", "ORG"): [1.0, 0.0], ("b", "ORG"): [0.85, 0.53]}  # cos ~0.85
    unions, suggestions = embedding_pairs(keys, vectors, auto_thr=0.92, suggest_thr=0.80)
    assert unions == []
    assert suggestions == [{"key_a": ("a", "ORG"), "key_b": ("b", "ORG"),
                            "score": pytest.approx(0.849, abs=0.01)}]


def _speaker(iid, sid, display, handle="S1", provisional=False):
    return {"interview_id": iid, "speaker_id": sid, "display_name": display,
            "handle": handle, "provisional": provisional}


class TestPersonGroups:
    def test_exact_name_across_interviews_auto_links(self):
        auto, suggestions = person_groups(
            [_speaker("i1", "s1", "Jane Doe"), _speaker("i2", "s2", "jane doe")], {}
        )
        assert len(auto) == 1
        group = auto[0]
        assert group["person_key"] == "jane doe"
        assert group["method"] == "exact_name"
        assert sorted(group["links"]) == [("i1", "s1"), ("i2", "s2")]

    def test_front_matter_participant_single_speaker_auto_links(self):
        auto, _ = person_groups(
            [_speaker("i1", "s1", "Jane Doe")], {"i1": ["Jane Doe"]}
        )
        assert auto[0]["method"] == "front_matter"
        assert auto[0]["display_name"] == "Jane Doe"  # participant spelling wins

    def test_single_speaker_without_front_matter_not_auto(self):
        auto, _ = person_groups([_speaker("i1", "s1", "Jane Doe")], {})
        assert auto == []

    def test_provisional_and_handle_named_speakers_excluded(self):
        auto, _ = person_groups(
            [
                _speaker("i1", "s1", "S1", handle="S1"),           # unnamed
                _speaker("i2", "s2", "Jane Doe", provisional=True),
                _speaker("i3", "s3", "Jane Doe", provisional=True),
            ],
            {},
        )
        assert auto == []

    def test_first_name_only_becomes_suggestion(self):
        auto, suggestions = person_groups(
            [
                _speaker("i1", "s1", "Jane Doe"),
                _speaker("i2", "s2", "Jane Doe"),
                _speaker("i3", "s3", "Jane"),
            ],
            {},
        )
        assert len(auto) == 1
        assert len(suggestions) == 1
        s = suggestions[0]
        assert s["person_key"] == "jane doe"
        assert (s["interview_id"], s["speaker_id"]) == ("i3", "s3")
        assert s["reason"] == "first_name_match"
