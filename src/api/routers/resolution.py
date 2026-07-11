"""Human corrections for Layer 4 resolution (M4.5b).

Speakers-router pattern: load aggregate, call domain method, save; 202
{status, version} on accept, 404 unknown resource, 409 domain conflict.
"""

from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.events.envelope import Actor, ActorType
from src.events.project_events import canonical_entity_id, project_aggregate_id
from src.events.repository import get_project_repository
from src.resolution.candidates import normalize_surface

router = APIRouter(tags=["resolution"])


class MergeRequest(BaseModel):
    surviving_canonical_id: str
    merged_canonical_id: str


class SplitRequest(BaseModel):
    surfaces: List[str]
    new_name: str


class LinkRequest(BaseModel):
    interview_id: str
    speaker_id: str
    display_name: Optional[str] = None  # required only to create a new person


class UnlinkRequest(BaseModel):
    interview_id: str
    speaker_id: str
    note: Optional[str] = None


def _human_actor(x_user_id: Optional[str]) -> Actor:
    return Actor(actor_type=ActorType.HUMAN, user_id=x_user_id or "anonymous")


def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_project(project_id: str):
    repo = get_project_repository()
    project = await repo.load(project_aggregate_id(project_id))
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} has no resolution state")
    return repo, project


@router.post("/resolution/{project_id}/entities/merge", status_code=202)
async def merge_entities(
    project_id: str,
    body: MergeRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    try:
        project.confirm_entity_merge(
            project_id, body.surviving_canonical_id, body.merged_canonical_id,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/entities/{canonical_id}/split", status_code=202)
async def split_entity(
    project_id: str,
    canonical_id: str,
    body: SplitRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    entry = project.canonical_entities.get(canonical_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Canonical entity {canonical_id} not found")
    new_canonical_id = canonical_entity_id(
        project_id, normalize_surface(body.new_name), entry["entity_type"]
    )
    try:
        project.split_entity(
            project_id, canonical_id, body.surfaces, new_canonical_id, body.new_name,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/persons/{person_id}/link", status_code=202)
async def link_speaker(
    project_id: str,
    person_id: str,
    body: LinkRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    actor = _human_actor(x_user_id)
    try:
        if person_id not in project.persons:
            if body.display_name is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Person {person_id} not found (pass display_name to create)",
                )
            project.identify_person(project_id, person_id, body.display_name, actor=actor)
        project.link_speaker_to_person(
            project_id, body.interview_id, body.speaker_id, person_id, "human", 1.0,
            actor=actor,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/persons/{person_id}/unlink", status_code=202)
async def unlink_speaker(
    project_id: str,
    person_id: str,
    body: UnlinkRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    try:
        project.remove_person_link(
            project_id, body.interview_id, body.speaker_id, person_id, note=body.note,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)
