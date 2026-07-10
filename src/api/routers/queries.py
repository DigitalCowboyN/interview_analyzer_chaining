"""Richer read-model queries (Layer 5)."""

from typing import Optional

from fastapi import APIRouter, Query

from src.export import reader
from src.export.renderer import RESERVED_PROPS
from src.utils.neo4j_driver import Neo4jConnectionManager

router = APIRouter(tags=["queries"])


@router.get("/interviews/{interview_id}/lenses/{lens}/items")
async def lens_items(
    interview_id: str,
    lens: str,
    node_type: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.lens_item_rows(
            session, interview_id, lens, node_type=node_type,
            min_confidence=min_confidence, limit=limit, offset=offset,
        )
    items = []
    for row in rows:
        props = row.pop("props", {})
        row["fields"] = {k: v for k, v in props.items() if k not in RESERVED_PROPS}
        items.append(row)
    return {"items": items}


@router.get("/review/worklist")
async def review_worklist(
    project_id: Optional[str] = None,
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    async with await Neo4jConnectionManager.get_session() as session:
        return await reader.worklist_rows(
            session, project_id=project_id, threshold=threshold,
            limit=limit, offset=offset,
        )


@router.get("/speakers/rollup")
async def speakers_rollup(
    project_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    async with await Neo4jConnectionManager.get_session() as session:
        speakers = await reader.speaker_rollup_rows(
            session, project_id=project_id, name=name, limit=limit, offset=offset,
        )
    return {"speakers": speakers}
