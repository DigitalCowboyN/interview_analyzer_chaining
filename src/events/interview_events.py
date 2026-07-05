"""
Interview aggregate domain events.

Defines all events that can occur within the Interview aggregate lifecycle,
including creation, updates, status changes, and archival.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .envelope import AggregateType, EventEnvelope


class InterviewStatus(str, Enum):
    """Possible status values for an interview."""

    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class InterviewCreatedData(BaseModel):
    """Data payload for InterviewCreated event."""

    title: str = Field(..., description="Interview title/name")
    source: str = Field(..., description="Source of the interview (e.g., filename)")
    project_id: str = Field(..., description="ID of the project this interview belongs to")
    language: Optional[str] = Field(None, description="Language of the interview content")
    started_at: Optional[datetime] = Field(None, description="When the interview was conducted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional interview metadata")


class InterviewUpdatedData(BaseModel):
    """Data payload for InterviewUpdated event."""

    title: Optional[str] = Field(None, description="Updated interview title")
    language: Optional[str] = Field(None, description="Updated language")
    metadata_diff: Dict[str, Any] = Field(default_factory=dict, description="Changed metadata fields")


class StatusChangedData(BaseModel):
    """Data payload for StatusChanged event."""

    from_status: InterviewStatus = Field(..., description="Previous status")
    to_status: InterviewStatus = Field(..., description="New status")
    reason: Optional[str] = Field(None, description="Reason for status change")


class InterviewArchivedData(BaseModel):
    """Data payload for InterviewArchived event."""

    reason: Optional[str] = Field(None, description="Reason for archiving")


class InterviewDeletedData(BaseModel):
    """Data payload for InterviewDeleted event (rare; prefer Archive)."""

    reason: Optional[str] = Field(None, description="Reason for deletion")


class SpeakerCreatedData(BaseModel):
    """Data payload for SpeakerCreated event."""

    speaker_id: str = Field(..., description="Deterministic UUID of the speaker")
    handle: str = Field(..., description="Stable short handle, e.g. 'S1' or parsed label")
    display_name: str = Field(..., description="Human-readable name (initially the handle)")
    provisional: bool = Field(..., description="True when inferred rather than confirmed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence")
    method: str = Field(..., description="'parsed' | 'inference'")


class SpeakerRenamedData(BaseModel):
    """Data payload for SpeakerRenamed event (human correction)."""

    speaker_id: str = Field(..., description="UUID of the speaker")
    old_display_name: str = Field(..., description="Previous display name")
    new_display_name: str = Field(..., description="New display name")


class SpeakerMergedData(BaseModel):
    """Data payload for SpeakerMerged event (human correction: two handles, one person)."""

    surviving_speaker_id: str = Field(..., description="Speaker that remains")
    merged_speaker_id: str = Field(..., description="Speaker merged away")


class UtteranceIdentifiedData(BaseModel):
    """Data payload for UtteranceIdentified event (stitching overlay)."""

    utterance_id: str = Field(..., description="Deterministic UUID of the utterance")
    speaker_id: str = Field(..., description="Speaker whose continuous thought this is")
    fragment_ids: List[str] = Field(..., description="Ordered fragment UUIDs composing the utterance")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Stitching confidence")


class InterruptionRecordedData(BaseModel):
    """Data payload for InterruptionRecorded event."""

    interrupting_utterance_id: str = Field(..., description="Utterance that broke in")
    interrupted_utterance_id: str = Field(..., description="Utterance that was broken into")
    at_fragment_id: str = Field(..., description="First fragment of the interruption")


class StitchRemovedData(BaseModel):
    """Data payload for StitchRemoved event (human correction)."""

    utterance_id: str = Field(..., description="Utterance whose stitch is removed")
    reason: Optional[str] = Field(None, description="Why the stitch was wrong")


class ClaimExtractedData(BaseModel):
    """Data payload for ClaimExtracted event (Layer 2, utterance-scoped)."""

    claim_id: str = Field(..., description="Deterministic UUID of the claim")
    utterance_id: str = Field(..., description="Utterance the claim was extracted from")
    speaker_id: str = Field(..., description="Speaker who made the claim")
    text: str = Field(..., description="The claim text (quote or close paraphrase)")
    kind: str = Field(..., description="assertion | commitment | request")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str = Field(..., description="Model that extracted the claim")
    provider: str = Field(..., description="Provider that served the call")


def create_interview_created_event(
    aggregate_id: str,
    version: int,
    title: str,
    source: str,
    project_id: str,
    language: Optional[str] = None,
    started_at: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **envelope_kwargs
) -> EventEnvelope:
    """
    Create an InterviewCreated event.

    Args:
        aggregate_id: UUID of the interview
        version: Version number in the aggregate stream
        title: Interview title/name
        source: Source of the interview content
        project_id: ID of the project this interview belongs to
        language: Language of the interview content
        started_at: When the interview was conducted
        metadata: Additional interview metadata
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = InterviewCreatedData(
        title=title, source=source, project_id=project_id, language=language, started_at=started_at, metadata=metadata or {}
    )

    return EventEnvelope(
        event_type="InterviewCreated",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        project_id=project_id,  # Also store in envelope for querying
        **envelope_kwargs
    )


def create_interview_updated_event(
    aggregate_id: str,
    version: int,
    title: Optional[str] = None,
    language: Optional[str] = None,
    metadata_diff: Optional[Dict[str, Any]] = None,
    **envelope_kwargs
) -> EventEnvelope:
    """
    Create an InterviewUpdated event.

    Args:
        aggregate_id: UUID of the interview
        version: Version number in the aggregate stream
        title: Updated interview title
        language: Updated language
        metadata_diff: Changed metadata fields
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = InterviewUpdatedData(title=title, language=language, metadata_diff=metadata_diff or {})

    return EventEnvelope(
        event_type="InterviewUpdated",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_status_changed_event(
    aggregate_id: str,
    version: int,
    from_status: InterviewStatus,
    to_status: InterviewStatus,
    reason: Optional[str] = None,
    **envelope_kwargs
) -> EventEnvelope:
    """
    Create a StatusChanged event for an interview.

    Args:
        aggregate_id: UUID of the interview
        version: Version number in the aggregate stream
        from_status: Previous status
        to_status: New status
        reason: Reason for status change
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = StatusChangedData(from_status=from_status, to_status=to_status, reason=reason)

    return EventEnvelope(
        event_type="StatusChanged",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_interview_archived_event(
    aggregate_id: str, version: int, reason: Optional[str] = None, **envelope_kwargs
) -> EventEnvelope:
    """
    Create an InterviewArchived event.

    Args:
        aggregate_id: UUID of the interview
        version: Version number in the aggregate stream
        reason: Reason for archiving
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = InterviewArchivedData(reason=reason)

    return EventEnvelope(
        event_type="InterviewArchived",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_interview_deleted_event(
    aggregate_id: str, version: int, reason: Optional[str] = None, **envelope_kwargs
) -> EventEnvelope:
    """
    Create an InterviewDeleted event (rare; prefer Archive).

    Args:
        aggregate_id: UUID of the interview
        version: Version number in the aggregate stream
        reason: Reason for deletion
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = InterviewDeletedData(reason=reason)

    return EventEnvelope(
        event_type="InterviewDeleted",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )
