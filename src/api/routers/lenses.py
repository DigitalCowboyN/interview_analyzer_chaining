"""
src/api/routers/lenses.py

Corrections surface for lens items (Layer 3). A human override locks the item:
it survives lens re-runs and is never deleted by a LensApplied supersession.
"""

from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.events.aggregates import Interview
from src.events.envelope import Actor, ActorType
from src.events.repository import InterviewRepository, get_interview_repository
from src.utils.logger import get_logger

router = APIRouter(tags=["lenses"])
logger = get_logger()


def _human_actor(x_user_id: Optional[str]) -> Actor:
    """Build the HUMAN actor from the X-User-ID header (audit provenance)."""
    return Actor(actor_type=ActorType.HUMAN, user_id=x_user_id or "anonymous")


class OverrideLensItemRequest(BaseModel):
    fields_overridden: Dict[str, Any] = Field(..., min_length=1)
    note: Optional[str] = None


def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_interview(interview_id: str) -> Tuple[InterviewRepository, Interview]:
    repo = get_interview_repository()
    interview = await repo.load(interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Interview not found")
    return repo, interview


@router.post("/lenses/{interview_id}/items/{item_id}/override", status_code=202)
async def override_lens_item(
    interview_id: str,
    item_id: str,
    body: OverrideLensItemRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: fix a lens item's fields and lock it against re-runs."""
    repo, interview = await _load_interview(interview_id)
    try:
        interview.override_lens_extraction(
            item_id, body.fields_overridden, note=body.note, actor=_human_actor(x_user_id)
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)
