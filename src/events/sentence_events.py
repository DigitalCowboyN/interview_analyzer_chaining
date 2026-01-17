"""
Sentence aggregate domain events.

Defines all events that can occur within the Sentence aggregate lifecycle,
including creation, editing, analysis generation, overrides, and tagging.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .envelope import AggregateType, EventEnvelope


class EditorType(str, Enum):
    """Type of editor that modified the sentence."""

    HUMAN = "human"
    AI = "ai"


class TagType(str, Enum):
    """Type of tag that can be applied to sentences."""

    KEYWORD = "keyword"
    TOPIC = "topic"


class SentenceStatus(str, Enum):
    """Possible status values for a sentence."""

    CREATED = "created"
    ANALYZED = "analyzed"
    EDITED = "edited"
    DELETED = "deleted"


class SentenceCreatedData(BaseModel):
    """Data payload for SentenceCreated event."""

    interview_id: str = Field(..., description="UUID of the parent interview")
    index: int = Field(..., ge=0, description="Sequence order within the interview")
    text: str = Field(..., description="The sentence text content")
    speaker: Optional[str] = Field(None, description="Speaker identifier if available")
    start_ms: Optional[int] = Field(None, ge=0, description="Start time in milliseconds")
    end_ms: Optional[int] = Field(None, ge=0, description="End time in milliseconds")


class SentenceEditedData(BaseModel):
    """Data payload for SentenceEdited event."""

    old_text: str = Field(..., description="Previous sentence text")
    new_text: str = Field(..., description="Updated sentence text")
    editor_type: EditorType = Field(..., description="Type of editor that made the change")


class SentenceRelocatedData(BaseModel):
    """Data payload for SentenceRelocated event (if reordering happens)."""

    old_index: int = Field(..., ge=0, description="Previous sequence order")
    new_index: int = Field(..., ge=0, description="New sequence order")


class SentenceTaggedData(BaseModel):
    """Data payload for SentenceTagged event."""

    tag_type: TagType = Field(..., description="Type of tag being added")
    value: str = Field(..., description="Tag value")


class SentenceUntaggedData(BaseModel):
    """Data payload for SentenceUntagged event."""

    tag_type: TagType = Field(..., description="Type of tag being removed")
    value: str = Field(..., description="Tag value")


class SentenceStatusChangedData(BaseModel):
    """Data payload for StatusChanged event."""

    from_status: SentenceStatus = Field(..., description="Previous status")
    to_status: SentenceStatus = Field(..., description="New status")
    reason: Optional[str] = Field(None, description="Reason for status change")


class SentenceDeletedData(BaseModel):
    """Data payload for SentenceDeleted event."""

    reason: Optional[str] = Field(None, description="Reason for deletion")


class AnalysisGeneratedData(BaseModel):
    """Data payload for AnalysisGenerated event."""

    model: str = Field(..., description="AI model used for analysis")
    version: str = Field(..., description="Model version")
    classification: Dict[str, Any] = Field(
        ..., description="Classification results (function_type, structure_type, purpose)"
    )
    keywords: List[str] = Field(default_factory=list, description="Overall keywords identified")
    topics: List[str] = Field(default_factory=list, description="Topics identified")
    domain_keywords: List[str] = Field(default_factory=list, description="Domain-specific keywords")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Analysis confidence score")
    raw_ref: Optional[str] = Field(None, description="Reference to raw analysis data")


class AnalysisRegeneratedData(BaseModel):
    """Data payload for AnalysisRegenerated event."""

    model: str = Field(..., description="AI model used for re-analysis")
    reason: str = Field(..., description="Reason for regeneration")
    delta: Optional[Dict[str, Any]] = Field(None, description="Changes from previous analysis")
    classification: Dict[str, Any] = Field(..., description="New classification results")
    keywords: List[str] = Field(default_factory=list, description="New overall keywords")
    topics: List[str] = Field(default_factory=list, description="New topics")
    domain_keywords: List[str] = Field(default_factory=list, description="New domain keywords")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="New confidence score")


class AnalysisOverriddenData(BaseModel):
    """Data payload for AnalysisOverridden event."""

    fields_overridden: Dict[str, Any] = Field(..., description="Fields that were manually overridden")
    note: Optional[str] = Field(None, description="Note explaining the override")


class AnalysisClearedData(BaseModel):
    """Data payload for AnalysisCleared event."""

    reason: Optional[str] = Field(None, description="Reason for clearing analysis")


def create_sentence_created_event(
    aggregate_id: str,
    version: int,
    interview_id: str,
    index: int,
    text: str,
    speaker: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    **envelope_kwargs
) -> EventEnvelope:
    """
    Create a SentenceCreated event.

    Args:
        aggregate_id: UUID of the sentence
        version: Version number in the aggregate stream
        interview_id: UUID of the parent interview
        index: Sequence order within the interview
        text: The sentence text content
        speaker: Speaker identifier if available
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = SentenceCreatedData(
        interview_id=interview_id, index=index, text=text, speaker=speaker, start_ms=start_ms, end_ms=end_ms
    )

    return EventEnvelope(
        event_type="SentenceCreated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_sentence_edited_event(
    aggregate_id: str, version: int, old_text: str, new_text: str, editor_type: EditorType, **envelope_kwargs
) -> EventEnvelope:
    """
    Create a SentenceEdited event.

    Args:
        aggregate_id: UUID of the sentence
        version: Version number in the aggregate stream
        old_text: Previous sentence text
        new_text: Updated sentence text
        editor_type: Type of editor that made the change
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = SentenceEditedData(old_text=old_text, new_text=new_text, editor_type=editor_type)

    return EventEnvelope(
        event_type="SentenceEdited",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_analysis_generated_event(
    aggregate_id: str,
    version: int,
    model: str,
    model_version: str,
    classification: Dict[str, Any],
    keywords: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    domain_keywords: Optional[List[str]] = None,
    confidence: Optional[float] = None,
    raw_ref: Optional[str] = None,
    **envelope_kwargs
) -> EventEnvelope:
    """
    Create an AnalysisGenerated event.

    Args:
        aggregate_id: UUID of the sentence
        version: Version number in the aggregate stream
        model: AI model used for analysis
        model_version: Model version
        classification: Classification results (function_type, structure_type, purpose)
        keywords: Overall keywords identified
        topics: Topics identified
        domain_keywords: Domain-specific keywords
        confidence: Analysis confidence score
        raw_ref: Reference to raw analysis data
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = AnalysisGeneratedData(
        model=model,
        version=model_version,
        classification=classification,
        keywords=keywords or [],
        topics=topics or [],
        domain_keywords=domain_keywords or [],
        confidence=confidence,
        raw_ref=raw_ref,
    )

    return EventEnvelope(
        event_type="AnalysisGenerated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )


def create_analysis_overridden_event(
    aggregate_id: str, version: int, fields_overridden: Dict[str, Any], note: Optional[str] = None, **envelope_kwargs
) -> EventEnvelope:
    """
    Create an AnalysisOverridden event.

    Args:
        aggregate_id: UUID of the sentence
        version: Version number in the aggregate stream
        fields_overridden: Fields that were manually overridden
        note: Note explaining the override
        **envelope_kwargs: Additional envelope fields (actor, correlation_id, etc.)

    Returns:
        EventEnvelope: Complete event ready for storage
    """
    data = AnalysisOverriddenData(fields_overridden=fields_overridden, note=note)

    return EventEnvelope(
        event_type="AnalysisOverridden",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=aggregate_id,
        version=version,
        data=data.model_dump(),
        **envelope_kwargs
    )
