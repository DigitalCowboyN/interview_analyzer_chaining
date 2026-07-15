"""
src/api/routers/segments.py

Segment read + correction endpoints (Layer 4, M4.5c). Reads come from the
Neo4j overlay; the only correction is removal — redraw = remove + re-run
(`python -m src.enrichment <interview_id> --force`).
"""

from typing import Optional, Tuple

from fastapi import APIRouter, Header, HTTPException

from src.events.aggregates import Interview
from src.events.envelope import Actor, ActorType
from src.events.repository import InterviewRepository, get_interview_repository
from src.export import reader
from src.utils.neo4j_driver import Neo4jConnectionManager

router = APIRouter(tags=["segments"])


def _human_actor(x_user_id: Optional[str]) -> Actor:
    """Build the HUMAN actor from the X-User-ID header (audit provenance)."""
    return Actor(actor_type=ActorType.HUMAN, user_id=x_user_id or "anonymous")


def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_interview(interview_id: str) -> Tuple[InterviewRepository, Interview]:
    repo = get_interview_repository()
    interview = await repo.load(interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail=f"Interview {interview_id} not found")
    return repo, interview


@router.get("/interviews/{interview_id}/segments")
async def list_segments(interview_id: str):
    """Topic segments with their fragment ranges, in transcript order."""
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.segment_rows(session, interview_id)
    return {"segments": rows}


@router.delete("/segments/{interview_id}/{segment_id}", status_code=202)
async def remove_segment(
    interview_id: str,
    segment_id: str,
    reason: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: a segment was wrong; remove it (redraw = remove + re-run).

    `reason` is a query parameter (DELETE request bodies are stripped by some
    proxies/clients).
    """
    repo, interview = await _load_interview(interview_id)
    try:
        interview.remove_segment(segment_id, reason=reason, actor=_human_actor(x_user_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)
