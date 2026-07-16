"""validate_segments: drop-all-on-any-invalid (degrade, don't guess)."""

import pytest

from src.enrichment.segments import validate_segments


def seg(start, end, topic="t", confidence=0.9):
    return {"topic": topic, "start_index": start, "end_index": end, "confidence": confidence}


INDICES = {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    "segments",
    [
        [seg(2, 1)],                       # inverted range
        [seg(0, 5)],                       # end not an existing index
        [seg(9, 9)],                       # start not an existing index
        [seg(0, 2), seg(2, 4)],            # overlap at boundary
        [seg(3, 4), seg(0, 3)],            # overlap discovered after ordering
        [seg(0, 1), seg(1, 1)],            # containment
    ],
)
def test_any_invalid_segment_drops_all(segments):
    assert validate_segments(segments, INDICES) is None


def test_valid_segments_returned_ordered_by_start():
    result = validate_segments([seg(3, 4, "b"), seg(0, 1, "a")], INDICES)
    assert [s["topic"] for s in result] == ["a", "b"]


def test_gaps_are_allowed_and_empty_is_valid():
    assert validate_segments([seg(0, 0), seg(4, 4)], INDICES) is not None
    assert validate_segments([], INDICES) == []


def test_missing_interior_fragment_invalidates_range():
    # Range spans an index that doesn't exist (fragment never loaded).
    assert validate_segments([seg(0, 2)], {0, 2}) is None
