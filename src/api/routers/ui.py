"""
src/api/routers/ui.py

The `/ui/*` read layer (M5.0 Task 1): backend contract for the Next.js
frontend. Thin router — session → reader → shape; zero writes, no auth.
"""

from fastapi import APIRouter, HTTPException, Query

from src.events.project_events import person_id_for
from src.resolution.candidates import normalize_name
from src.ui import reader
from src.utils.neo4j_driver import Neo4jConnectionManager

router = APIRouter(prefix="/ui", tags=["ui"])


async def _require_project(session, project_id: str) -> None:
    if not await reader.project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


def _shape_transcript_line(row: dict) -> dict:
    speaker = (
        {"speaker_id": row["speaker_id"], "display_name": row["speaker_display_name"]}
        if row["speaker_id"] is not None else None
    )
    person = (
        {"person_id": row["person_id"], "display_name": row["person_display_name"]}
        if row["person_id"] is not None else None
    )
    segment = (
        {"segment_id": row["segment_id"], "topic": row["segment_topic"]}
        if row["segment_id"] is not None else None
    )
    return {
        "fragment_id": row["fragment_id"],
        "sequence_order": row["sequence_order"],
        "text": row["text"],
        "speaker": speaker,
        "person": person,
        "utterance_id": row["utterance_id"],
        "segment": segment,
        "entities": row["entities"],
        "lens_items": row["lens_items"],
        "edited": bool(row["edited"]),
    }


_DIMENSION_NODE_TYPES = {
    "traits": "Trait",
    "goals": "Goal",
    "pain_points": "PainPoint",
    "notable_quotes": "NotableQuote",
}


def _shape_dimension_item(row: dict) -> dict:
    return {
        "item_id": row["item_id"],
        "text": row["text"],
        "confidence": row["confidence"],
        "interview_id": row["interview_id"],
        "interview_title": row["interview_title"],
    }


@router.get("/projects")
async def list_projects():
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.project_rows(session)
    return {"projects": rows}


@router.get("/projects/{project_id}/interviews")
async def list_interviews(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.interview_rows(session, project_id)
    return {"interviews": rows}


@router.get("/interviews/{interview_id}/transcript")
async def get_transcript(interview_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        header = await reader.interview_header_row(session, interview_id)
        if header is None:
            raise HTTPException(status_code=404, detail=f"Interview {interview_id} not found")
        line_rows = await reader.transcript_line_rows(session, interview_id)
    return {
        "interview_id": header["interview_id"],
        "title": header["title"],
        "metadata": header["metadata"],
        "lines": [_shape_transcript_line(row) for row in line_rows],
    }


@router.get("/projects/{project_id}/personas")
async def list_personas(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.persona_card_rows(session, project_id)
    return {"personas": rows}


@router.get("/personas/{project_id}/{person_id}")
async def get_persona(project_id: str, person_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        if not await reader.persona_exists(session, project_id, person_id):
            raise HTTPException(status_code=404, detail=f"Persona {person_id} not found")
        display_row = await reader.person_display_name_row(session, project_id, person_id)
        detail_rows = await reader.persona_detail_rows(session, project_id, person_id)

    dimensions = {key: [] for key in _DIMENSION_NODE_TYPES}
    by_node_type = {node_type: key for key, node_type in _DIMENSION_NODE_TYPES.items()}
    for row in detail_rows:
        key = by_node_type.get(row["node_type"])
        if key is not None:
            dimensions[key].append(_shape_dimension_item(row))

    return {
        "person_id": person_id,
        "display_name": display_row["display_name"] if display_row else None,
        "dimensions": dimensions,
    }


@router.get("/projects/{project_id}/persons")
async def list_persons(project_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
        rows = await reader.person_card_rows(session, project_id)
    return {"persons": rows}


@router.get("/persons/{project_id}/{person_id}")
async def get_person(project_id: str, person_id: str):
    async with await Neo4jConnectionManager.get_session() as session:
        if not await reader.person_exists(session, project_id, person_id):
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        display_row = await reader.person_display_name_row(session, project_id, person_id)
        links = await reader.person_detail_rows(session, project_id, person_id)
        contributes = await reader.person_contributes_to_persona(session, project_id, person_id)
    return {
        "person_id": person_id,
        "display_name": display_row["display_name"] if display_row else None,
        "links": links,
        "contributes_to_persona": contributes,
    }


@router.get("/projects/{project_id}/person-id")
async def derive_person_id(project_id: str, display_name: str = Query(..., min_length=1)):
    """Compute-only id derivation for the create-new-person flow — the
    frontend must never derive ids itself (loose-coupling requirement)."""
    async with await Neo4jConnectionManager.get_session() as session:
        await _require_project(session, project_id)
    return {"person_id": person_id_for(project_id, normalize_name(display_name))}
