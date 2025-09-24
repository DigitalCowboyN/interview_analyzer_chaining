"""
Interview aggregate domain events.

Defines all events that can occur within the Interview aggregate lifecycle,
including creation, updates, status changes, and archival.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

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


def create_interview_created_event(
    aggregate_id: str,
    version: int,
    title: str,
    source: str,
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
        language: Language of the interview content
        started_at: When the interview was conducted
        metadata: Additional interview metadata
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = InterviewCreatedData(
        title=title, source=source, language=language, started_at=started_at, metadata=metadata or {}
    )

    return EventEnvelope(
        event_type="InterviewCreated",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=aggregate_id,
        version=version,
        data=data.dict(),
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
        data=data.dict(),
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
        data=data.dict(),
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
        data=data.dict(),
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
        data=data.dict(),
        **envelope_kwargs
    )
