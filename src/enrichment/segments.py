"""Segment proposal validation (Layer 4, M4.5c).

Validation happens at emit time and is never a failed enrichment: an invalid
proposal drops ALL segments and raises a flag instead (stitcher precedent:
degrade, don't guess).
"""

from typing import Any, Dict, List, Optional, Set


def validate_segments(
    segments: List[Dict[str, Any]], existing_indices: Set[int]
) -> Optional[List[Dict[str, Any]]]:
    """Return the segments ordered by start_index, or None if ANY is invalid.

    Rules (spec): start <= end; every index in every range must be an
    existing fragment sequence number; ranges must not overlap once ordered.
    """
    for seg in segments:
        if seg["start_index"] > seg["end_index"]:
            return None
        if not all(
            i in existing_indices
            for i in range(seg["start_index"], seg["end_index"] + 1)
        ):
            return None
    ordered = sorted(segments, key=lambda s: s["start_index"])
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt["start_index"] <= prev["end_index"]:
            return None
    return ordered
