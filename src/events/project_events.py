"""Event payload models for the Project aggregate (M4.5b resolution core).

Wire format, frozen once merged: the event type names registered for these
payloads, AggregateType.PROJECT's "Project" value, and Project-{aggregate_id}
stream names. Envelope aggregate ids must be UUIDs, so the aggregate id is
uuid5 of the human project_id (project_aggregate_id); payloads carry the
human project_id for the graph handlers, which MERGE on it.
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ResolutionMethod = Literal["deterministic", "human"]
LinkMethod = Literal["exact_name", "front_matter", "human"]


def project_aggregate_id(project_id: str) -> str:
    """Deterministic UUID for a project's event stream."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"project:{project_id}"))


def canonical_entity_id(project_id: str, normalized_name: str, entity_type: str) -> str:
    """Spec derivation: uuid5('{project_id}:entity:{normalized_name}:{entity_type}')."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}:entity:{normalized_name}:{entity_type}"))


def person_id_for(project_id: str, normalized_display_name: str) -> str:
    """Spec derivation: uuid5('{project_id}:person:{normalized_display_name}')."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}:person:{normalized_display_name}"))


class EntityCanonicalizedData(BaseModel):
    project_id: str
    canonical_id: str
    name: str
    entity_type: str
    surfaces: List[str] = Field(..., min_length=1)
    method: ResolutionMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityAliasAddedData(BaseModel):
    project_id: str
    canonical_id: str
    surface: str
    method: ResolutionMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityMergeConfirmedData(BaseModel):
    project_id: str
    canonical_id: str  # survivor
    merged_canonical_id: str


class EntitySplitData(BaseModel):
    project_id: str
    canonical_id: str
    surfaces_removed: List[str] = Field(..., min_length=1)
    new_canonical_id: str
    new_name: str


class PersonIdentifiedData(BaseModel):
    project_id: str
    person_id: str
    display_name: str


class SpeakerLinkedToPersonData(BaseModel):
    project_id: str
    interview_id: str
    speaker_id: str
    person_id: str
    method: LinkMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class PersonLinkRemovedData(BaseModel):
    project_id: str
    interview_id: str
    speaker_id: str
    person_id: str
    note: Optional[str] = None
