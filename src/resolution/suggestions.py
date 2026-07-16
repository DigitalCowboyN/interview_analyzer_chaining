"""On-demand resolution suggestions for the review worklist (M4.5b).

Same candidate logic as the engine, but nothing is emitted. Suggest-band
entity rows carry deterministic derived canonical/person ids and become
actionable once the engine has minted the matching canonicals (a merge
confirm 404/409s until both canonicals exist). Auto-band entity rows are
surfaced ONLY when both sides already exist as unlocked canonicals — the
exact case the engine defers on its own run — and are immediately
actionable; derived-id auto pairs are the engine's job, not the worklist's.
Person rows are actionable immediately since linking a speaker mints the
person on accept.

If the embedder is unavailable, entity suggestions degrade to empty and the
response flags `"embedding_unavailable"`; person suggestions (no embedder
dependency) are still computed.

Front-matter participants are skipped here deliberately — on-demand
suggestions must not load every interview aggregate on a GET; the engine
covers participant-based auto-links.
"""

from typing import Any, Dict, List, Optional

from src.events.aggregates import Project
from src.events.project_events import canonical_entity_id, person_id_for
from src.resolution.candidates import (
    cosine,
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


def _existing_cid_for_key(project: Optional[Project], key, groups) -> Optional[str]:
    """EXISTING unlocked canonical for a group key — never a derived id.

    Auto-band pairs are only actionable on the worklist when both sides
    already exist as canonicals (the engine defers exactly that case);
    derived-id auto pairs are the engine's job on its next run.
    """
    if project is None:
        return None
    _, entity_type = key
    for row in groups[key]:
        cid = project.canonical_for_surface(row["surface"], entity_type)
        if cid is not None:
            entry = project.canonical_entities[cid]
            if entry["locked"] or entry["merged_into"] is not None:
                return None
            return cid
    return None


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
    flags: List[str] = []
    groups = exact_groups(entity_rows)
    keys = sorted(groups)
    if len(keys) > 1:
        reps = {key: representative(groups[key]) for key in keys}
        try:
            vectors = await embedder.embed([reps[key] for key in keys])
        except Exception:
            flags.append("embedding_unavailable")
            vectors = None
        if vectors is not None:
            by_key = dict(zip(keys, vectors))
            auto, band = embedding_pairs(keys, by_key, auto_thr, suggest_thr)
            pairs = [
                {"key_a": p["key_a"], "key_b": p["key_b"], "score": p["score"], "band": "suggest"}
                for p in band
            ] + [
                {
                    "key_a": key_a, "key_b": key_b,
                    "score": cosine(by_key[key_a], by_key[key_b]),
                    "band": "auto",
                }
                for key_a, key_b in auto
            ]
            for pair in pairs:
                band_name = pair["band"]
                if band_name == "auto":
                    # the engine defers auto pairs where BOTH sides already
                    # belong to existing canonicals — surface exactly those
                    cid_a = _existing_cid_for_key(project, pair["key_a"], groups)
                    cid_b = _existing_cid_for_key(project, pair["key_b"], groups)
                else:
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
                    "band": band_name,
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
        "flags": flags,
    }
