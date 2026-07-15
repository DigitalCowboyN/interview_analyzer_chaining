"""On-demand resolution suggestions for the review worklist (M4.5b).

Same candidate logic as the engine, but nothing is emitted: the worklist
shows the [suggest, auto) band and first-name person matches, each row
carrying exactly what the corrections endpoints need. Canonical/person ids
are the deterministic uuid5 derivations, so rows are actionable even before
(or between) engine runs.

Front-matter participants are skipped here deliberately — on-demand
suggestions must not load every interview aggregate on a GET; the engine
covers participant-based auto-links.
"""

from typing import Any, Dict, List, Optional

from src.events.aggregates import Project
from src.events.project_events import canonical_entity_id, person_id_for
from src.resolution.candidates import (
    embedding_pairs,
    exact_groups,
    person_groups,
    representative,
)
from src.resolution.reader import entity_surface_rows, speaker_rows


def _cid_for_key(project: Optional[Project], project_id: str, key, groups) -> Optional[str]:
    """Existing canonical for a group key, else the derived deterministic id.

    Returns None when the existing canonical is locked or merged away —
    the caller drops the suggestion.
    """
    normalized, entity_type = key
    surfaces = [row["surface"] for row in groups[key]]
    if project is not None:
        for surface in surfaces:
            cid = project.canonical_for_surface(surface, entity_type)
            if cid is not None:
                entry = project.canonical_entities[cid]
                if entry["locked"] or entry["merged_into"] is not None:
                    return None
                return cid
    return canonical_entity_id(project_id, normalized, entity_type)


async def compute_suggestions(
    session,
    project: Optional[Project],
    project_id: str,
    embedder,
    auto_thr: float = 0.92,
    suggest_thr: float = 0.80,
) -> Dict[str, List[Dict[str, Any]]]:
    entity_rows = await entity_surface_rows(session, project_id)
    speakers = await speaker_rows(session, project_id)

    entity_suggestions: List[Dict[str, Any]] = []
    groups = exact_groups(entity_rows)
    keys = sorted(groups)
    if len(keys) > 1:
        reps = {key: representative(groups[key]) for key in keys}
        vectors = await embedder.embed([reps[key] for key in keys])
        by_key = dict(zip(keys, vectors))
        _, band = embedding_pairs(keys, by_key, auto_thr, suggest_thr)
        for pair in band:
            cid_a = _cid_for_key(project, project_id, pair["key_a"], groups)
            cid_b = _cid_for_key(project, project_id, pair["key_b"], groups)
            if cid_a is None or cid_b is None or cid_a == cid_b:
                continue
            entity_suggestions.append({
                "surviving_canonical_id": cid_a,
                "merged_canonical_id": cid_b,
                "surfaces_a": sorted(row["surface"] for row in groups[pair["key_a"]]),
                "surfaces_b": sorted(row["surface"] for row in groups[pair["key_b"]]),
                "score": round(pair["score"], 4),
            })

    person_suggestions: List[Dict[str, Any]] = []
    _, candidates = person_groups(speakers, {})
    for suggestion in candidates:
        pair = (suggestion["interview_id"], suggestion["speaker_id"])
        if project is not None:
            if pair in project.blocked_links:
                continue
            if project.link_for_speaker(*pair) is not None:
                continue
        person_suggestions.append({
            "person_id": person_id_for(project_id, suggestion["person_key"]),
            "display_name": suggestion["display_name"],
            "interview_id": suggestion["interview_id"],
            "speaker_id": suggestion["speaker_id"],
            "speaker_display_name": suggestion["speaker_display_name"],
            "reason": suggestion["reason"],
        })

    return {
        "entity_merge_suggestions": entity_suggestions,
        "person_link_suggestions": person_suggestions,
    }
