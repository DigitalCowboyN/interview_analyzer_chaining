"""
Event envelope and metadata models.

Defines the core event structure with comprehensive metadata for audit trails,
correlation, and debugging. Uses UUIDv7 for event IDs to maintain temporal ordering.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ActorType(str, Enum):
    """Type of actor that initiated the event."""

    HUMAN = "human"
    SYSTEM = "system"
    AI = "ai"


class AggregateType(str, Enum):
    """Type of aggregate the event belongs to."""

    INTERVIEW = "Interview"
    SENTENCE = "Sentence"


class Actor(BaseModel):
    """Information about who/what initiated the event."""

    user_id: Optional[str] = None
    display: Optional[str] = None
    actor_type: ActorType

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class EventEnvelope(BaseModel):
    """
    Complete event envelope with all metadata.

    This is the standard structure for all events in the system, providing
    comprehensive metadata for audit trails, correlation, and debugging.
    """

    # Required fields (every event)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier (UUIDv4)")
    event_type: str = Field(..., description="Type of event (e.g., SentenceCreated)")
    aggregate_type: AggregateType = Field(..., description="Type of aggregate")
    aggregate_id: str = Field(..., description="UUID of the aggregate instance")
    version: int = Field(..., ge=0, description="Version number within the aggregate stream")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the event occurred (UTC)"
    )
    schema_version: str = Field(default="1.0.0", description="Schema version for event payload")

    # Event payload
    data: Dict[str, Any] = Field(..., description="Event-specific data payload")

    # Recommended fields (strongly encouraged)
    actor: Optional[Actor] = Field(None, description="Who/what initiated this event")
    correlation_id: Optional[str] = Field(None, description="Groups related events from one user action")
    causation_id: Optional[str] = Field(None, description="The command/event that triggered this")
    source: Optional[str] = Field(default="interview_analyzer", description="Service that generated the event")
    trace_id: Optional[str] = Field(None, description="APM/distributed tracing ID")
    project_id: Optional[str] = Field(None, description="Project/tenant ID if applicable")
    tags: List[str] = Field(default_factory=list, description="Free-form debugging labels")

    @validator("occurred_at")
    def occurred_at_must_be_utc(cls, v):
        """Ensure occurred_at is timezone-aware and in UTC."""
        if v.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if v.tzinfo != timezone.utc:
            v = v.astimezone(timezone.utc)
        return v

    @validator("event_id")
    def event_id_must_be_valid_uuid(cls, v):
        """Validate that event_id is a valid UUID."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("event_id must be a valid UUID")
        return v

    @validator("aggregate_id")
    def aggregate_id_must_be_valid_uuid(cls, v):
        """Validate that aggregate_id is a valid UUID."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("aggregate_id must be a valid UUID")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class EventMetadata(BaseModel):
    """
    Metadata-only portion of an event envelope.

    Useful for event filtering, querying, and processing without
    deserializing the full event payload.
    """

    event_id: str
    event_type: str
    aggregate_type: AggregateType
    aggregate_id: str
    version: int
    occurred_at: datetime
    schema_version: str
    actor: Optional[Actor] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    source: Optional[str] = None
    trace_id: Optional[str] = None
    project_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


def generate_event_id() -> str:
    """
    Generate a new event ID using UUIDv4.

    Note: In production, consider using UUIDv7 for better temporal ordering,
    but UUIDv4 is sufficient for initial implementation.

    Returns:
        str: A new UUID as a string
    """
    return str(uuid.uuid4())


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID for grouping related events.

    Returns:
        str: A new UUID as a string
    """
    return str(uuid.uuid4())
