"""
src/api/routers/speakers.py

API endpoints for correcting speaker attribution and stitching (Layer 1).
Human corrections are emitted as events through the aggregates; the projection
service updates Neo4j downstream. All correction events lock their fields
against system regeneration.
"""

import uuid
from typing import List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.events.aggregates import Interview
from src.events.envelope import Actor, ActorType
from src.events.repository import (
    InterviewRepository,
    get_interview_repository,
    get_sentence_repository,
)
from src.utils.logger import get_logger

router = APIRouter(tags=["speakers"])
logger = get_logger()


def _human_actor(x_user_id: Optional[str]) -> Actor:
    """Build the HUMAN actor from the X-User-ID header (audit provenance)."""
    return Actor(actor_type=ActorType.HUMAN, user_id=x_user_id or "anonymous")


# --- Request models ---

class RenameSpeakerRequest(BaseModel):
    new_display_name: str = Field(..., min_length=1)


class MergeSpeakersRequest(BaseModel):
    surviving_speaker_id: str
    merged_speaker_id: str


class SplitSpeakerRequest(BaseModel):
    new_handle: str = Field(..., min_length=1)
    new_display_name: str = Field(..., min_length=1)
    fragment_indices: List[int] = Field(..., min_length=1)


class ReattributeRequest(BaseModel):
    new_speaker_id: str


# --- Helpers ---

def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_interview(interview_id: str) -> Tuple[InterviewRepository, Interview]:
    repo = get_interview_repository()
    interview = await repo.load(interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Interview not found")
    return repo, interview


def _fragment_uuid(interview_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))


# --- Endpoints ---

@router.post("/speakers/{interview_id}/{speaker_id}/rename", status_code=202)
async def rename_speaker(
    interview_id: str,
    speaker_id: str,
    body: RenameSpeakerRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: give a provisional speaker a real name."""
    repo, interview = await _load_interview(interview_id)
    try:
        interview.rename_speaker(speaker_id, body.new_display_name, actor=_human_actor(x_user_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/merge", status_code=202)
async def merge_speakers(
    interview_id: str,
    body: MergeSpeakersRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: two provisional handles were the same person."""
    repo, interview = await _load_interview(interview_id)
    try:
        interview.merge_speakers(
            body.surviving_speaker_id, body.merged_speaker_id, actor=_human_actor(x_user_id)
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/split", status_code=202)
async def split_speaker(
    interview_id: str,
    body: SplitSpeakerRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Create a new speaker and reattribute the listed fragments to them.

    Composition of existing events (SpeakerCreated + SpeakerReattributed per
    fragment); no dedicated split event type. All fragments are loaded before
    any event is emitted, so a missing fragment aborts the whole operation.
    """
    actor = _human_actor(x_user_id)
    repo, interview = await _load_interview(interview_id)

    # Load every fragment up front: missing fragments must abort before any
    # event is appended (the log cannot be rolled back).
    sentence_repo = get_sentence_repository()
    sentences = []
    for index in body.fragment_indices:
        sentence = await sentence_repo.load(_fragment_uuid(interview_id, index))
        if sentence is None:
            raise HTTPException(status_code=404, detail=f"Fragment {index} not found")
        sentences.append(sentence)

    new_speaker_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:speaker:{body.new_handle}")
    )
    try:
        interview.add_speaker(
            new_speaker_id,
            handle=body.new_handle,
            display_name=body.new_display_name,
            provisional=False,
            confidence=1.0,
            method="human",
            actor=actor,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)

    for sentence in sentences:
        sentence.reattribute_speaker(new_speaker_id, actor=actor)
        await sentence_repo.save(sentence)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/fragments/{index}/reattribute", status_code=202)
async def reattribute_fragment(
    interview_id: str,
    index: int,
    body: ReattributeRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: this fragment was said by someone else."""
    # interview_id is load-bearing for UUID derivation; verify it exists so a
    # fabricated id cannot reattribute an unrelated sentence.
    await _load_interview(interview_id)
    sentence_repo = get_sentence_repository()
    sentence = await sentence_repo.load(_fragment_uuid(interview_id, index))
    if sentence is None:
        raise HTTPException(status_code=404, detail="Fragment not found")
    try:
        sentence.reattribute_speaker(body.new_speaker_id, actor=_human_actor(x_user_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await sentence_repo.save(sentence)
    return _accepted(sentence.version)


@router.delete("/stitches/{interview_id}/{utterance_id}", status_code=202)
async def remove_stitch(
    interview_id: str,
    utterance_id: str,
    reason: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Human correction: an identified utterance overlay was wrong; remove it.

    `reason` is a query parameter (DELETE request bodies are stripped by some
    proxies/clients).
    """
    repo, interview = await _load_interview(interview_id)
    try:
        interview.remove_stitch(utterance_id, reason=reason, actor=_human_actor(x_user_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)
