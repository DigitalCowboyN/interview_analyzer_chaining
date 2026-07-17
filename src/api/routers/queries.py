"""Richer read-model queries (Layer 5)."""

from typing import Optional

from fastapi import APIRouter, Query

from src.config import config
from src.enrichment.embedder import get_embedder
from src.events.project_events import project_aggregate_id
from src.events.repository import get_project_repository
from src.export import reader
from src.export.renderer import RESERVED_PROPS
from src.resolution.suggestions import compute_suggestions
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
        payload = await reader.worklist_rows(
            session, project_id=project_id, threshold=threshold,
            limit=limit, offset=offset,
        )

        if project_id is not None:
            project = await get_project_repository().load(project_aggregate_id(project_id))
            resolution_cfg = config.get("resolution", {})
            suggestions = await compute_suggestions(
                session, project, project_id, get_embedder(config),
                auto_thr=resolution_cfg.get("auto_merge_threshold", 0.92),
                suggest_thr=resolution_cfg.get("suggest_threshold", 0.80),
            )
        else:
            suggestions = {
                "entity_merge_suggestions": [],
                "person_link_suggestions": [],
                "flags": [],
            }
        payload.update(suggestions)
    return payload


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
