"""
Aggregate root base classes and domain aggregates.

Provides the foundational classes for event-sourced aggregates with
proper version tracking, event application, and state management.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .envelope import Actor, EventEnvelope
from .interview_events import InterviewStatus
from .sentence_events import EditorType, SentenceStatus


class AggregateRoot(ABC):
    """
    Base class for all event-sourced aggregates.

    Provides common functionality for version tracking, event application,
    and uncommitted event management. All domain aggregates should inherit
    from this class.
    """

    def __init__(self, aggregate_id: str):
        """
        Initialize a new aggregate root.

        Args:
            aggregate_id: Unique identifier for this aggregate instance
        """
        self.aggregate_id = aggregate_id
        self.version = -1  # -1 indicates a new aggregate
        self._uncommitted_events: List[EventEnvelope] = []

    @abstractmethod
    def apply_event(self, event: EventEnvelope) -> None:
        """
        Apply an event to update the aggregate's state.

        This method should be implemented by each aggregate to handle
        its specific event types and update internal state accordingly.

        Args:
            event: Event to apply to the aggregate
        """
        pass

    def load_from_history(self, events: List[EventEnvelope]) -> None:
        """
        Rebuild aggregate state from historical events.

        Args:
            events: List of events in chronological order
        """
        for event in events:
            self.apply_event(event)
            self.version = event.version

    def get_uncommitted_events(self) -> List[EventEnvelope]:
        """
        Get the list of uncommitted events.

        Returns:
            List[EventEnvelope]: Events that haven't been persisted yet
        """
        return self._uncommitted_events.copy()

    def mark_events_as_committed(self) -> None:
        """Mark all uncommitted events as committed (clear the list)."""
        self._uncommitted_events.clear()

    def _add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        actor: Optional[Actor] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """
        Add a new event to the uncommitted events list.

        Args:
            event_type: Type of event being added
            data: Event-specific data payload
            actor: Who/what initiated this event
            correlation_id: Groups related events from one user action
            causation_id: The command/event that triggered this
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The created event
        """
        from .envelope import AggregateType

        # Determine aggregate type based on class
        if isinstance(self, Interview):
            aggregate_type = AggregateType.INTERVIEW
        elif isinstance(self, Sentence):
            aggregate_type = AggregateType.SENTENCE
        else:
            raise ValueError(f"Unknown aggregate type: {type(self)}")

        # Calculate next version: current version + 1
        # For new aggregates (version == -1), this gives version 0
        # For existing aggregates, this increments from the last committed version
        new_version = self.version + 1

        event = EventEnvelope(
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=self.aggregate_id,
            version=new_version,
            data=data,
            actor=actor,
            correlation_id=correlation_id,
            causation_id=causation_id,
            **envelope_kwargs,
        )

        self._uncommitted_events.append(event)
        self.apply_event(event)
        # Update version to reflect the new event
        self.version = event.version
        return event


class Interview(AggregateRoot):
    """
    Interview aggregate representing a complete interview session.

    Manages the lifecycle of an interview including creation, status changes,
    updates, and archival. Contains metadata about the interview source,
    language, and processing status.
    """

    def __init__(self, aggregate_id: str):
        """Initialize a new Interview aggregate."""
        super().__init__(aggregate_id)
        self.title: Optional[str] = None
        self.source: Optional[str] = None
        self.language: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
        self.status: InterviewStatus = InterviewStatus.CREATED
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

    def apply_event(self, event: EventEnvelope) -> None:
        """Apply an event to update the interview's state."""
        if event.event_type == "InterviewCreated":
            self._apply_interview_created(event)
        elif event.event_type == "InterviewUpdated":
            self._apply_interview_updated(event)
        elif event.event_type == "StatusChanged":
            self._apply_status_changed(event)
        elif event.event_type == "InterviewArchived":
            self._apply_interview_archived(event)
        elif event.event_type == "InterviewDeleted":
            self._apply_interview_deleted(event)
        else:
            raise ValueError(f"Unknown event type for Interview: {event.event_type}")

    def _apply_interview_created(self, event: EventEnvelope) -> None:
        """Apply InterviewCreated event."""
        data = event.data
        self.title = data.get("title")
        self.source = data.get("source")
        self.language = data.get("language")
        self.started_at = data.get("started_at")
        if self.started_at and isinstance(self.started_at, str):
            self.started_at = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        self.metadata = data.get("metadata", {})
        self.status = InterviewStatus.CREATED
        self.created_at = event.occurred_at
        self.updated_at = event.occurred_at

    def _apply_interview_updated(self, event: EventEnvelope) -> None:
        """Apply InterviewUpdated event."""
        data = event.data
        if data.get("title") is not None:
            self.title = data["title"]
        if data.get("language") is not None:
            self.language = data["language"]
        if data.get("metadata_diff"):
            self.metadata.update(data["metadata_diff"])
        self.updated_at = event.occurred_at

    def _apply_status_changed(self, event: EventEnvelope) -> None:
        """Apply StatusChanged event."""
        data = event.data
        self.status = InterviewStatus(data["to_status"])
        self.updated_at = event.occurred_at

    def _apply_interview_archived(self, event: EventEnvelope) -> None:
        """Apply InterviewArchived event."""
        self.status = InterviewStatus.ARCHIVED
        self.updated_at = event.occurred_at

    def _apply_interview_deleted(self, event: EventEnvelope) -> None:
        """Apply InterviewDeleted event."""
        # In event sourcing, we don't actually delete the aggregate
        # but we can mark it as deleted in the state
        self.updated_at = event.occurred_at

    # Command methods (business logic)
    def create(
        self,
        title: str,
        source: str,
        language: Optional[str] = None,
        started_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """
        Create a new interview.

        Args:
            title: Interview title/name
            source: Source of the interview content
            language: Language of the interview content
            started_at: When the interview was conducted
            metadata: Additional interview metadata
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The InterviewCreated event
        """
        if self.version >= 0:
            raise ValueError("Interview has already been created")

        return self._add_event(
            event_type="InterviewCreated",
            data={
                "title": title,
                "source": source,
                "language": language,
                "started_at": started_at.isoformat() if started_at else None,
                "metadata": metadata or {},
            },
            **envelope_kwargs,
        )

    def update(
        self,
        title: Optional[str] = None,
        language: Optional[str] = None,
        metadata_diff: Optional[Dict[str, Any]] = None,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """
        Update interview information.

        Args:
            title: Updated interview title
            language: Updated language
            metadata_diff: Changed metadata fields
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The InterviewUpdated event
        """
        if self.version < 0:
            raise ValueError("Interview must be created before updating")

        return self._add_event(
            event_type="InterviewUpdated",
            data={"title": title, "language": language, "metadata_diff": metadata_diff or {}},
            **envelope_kwargs,
        )

    def change_status(
        self, new_status: InterviewStatus, reason: Optional[str] = None, **envelope_kwargs
    ) -> EventEnvelope:
        """
        Change the interview status.

        Args:
            new_status: New status to transition to
            reason: Reason for status change
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The StatusChanged event
        """
        if self.version < 0:
            raise ValueError("Interview must be created before changing status")

        if self.status == new_status:
            raise ValueError(f"Interview is already in status: {new_status}")

        return self._add_event(
            event_type="StatusChanged",
            data={"from_status": self.status.value, "to_status": new_status.value, "reason": reason},
            **envelope_kwargs,
        )


class Sentence(AggregateRoot):
    """
    Sentence aggregate representing a single sentence within an interview.

    Manages sentence content, editing, analysis results, and manual overrides.
    Tracks the relationship to its parent interview and maintains sequence order.
    """

    def __init__(self, aggregate_id: str):
        """Initialize a new Sentence aggregate."""
        super().__init__(aggregate_id)
        self.interview_id: Optional[str] = None
        self.index: Optional[int] = None
        self.text: Optional[str] = None
        self.speaker: Optional[str] = None
        self.start_ms: Optional[int] = None
        self.end_ms: Optional[int] = None
        self.status: SentenceStatus = SentenceStatus.CREATED
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

        # Analysis fields
        self.analysis_model: Optional[str] = None
        self.analysis_version: Optional[str] = None
        self.classification: Dict[str, Any] = {}
        self.keywords: List[str] = []
        self.topics: List[str] = []
        self.domain_keywords: List[str] = []
        self.confidence: Optional[float] = None
        self.raw_ref: Optional[str] = None

        # Override tracking
        self.overridden_fields: Dict[str, Any] = {}
        self.override_note: Optional[str] = None

    def apply_event(self, event: EventEnvelope) -> None:
        """Apply an event to update the sentence's state."""
        if event.event_type == "SentenceCreated":
            self._apply_sentence_created(event)
        elif event.event_type == "SentenceEdited":
            self._apply_sentence_edited(event)
        elif event.event_type == "AnalysisGenerated":
            self._apply_analysis_generated(event)
        elif event.event_type == "AnalysisRegenerated":
            self._apply_analysis_regenerated(event)
        elif event.event_type == "AnalysisOverridden":
            self._apply_analysis_overridden(event)
        elif event.event_type == "AnalysisCleared":
            self._apply_analysis_cleared(event)
        else:
            raise ValueError(f"Unknown event type for Sentence: {event.event_type}")

    def _apply_sentence_created(self, event: EventEnvelope) -> None:
        """Apply SentenceCreated event."""
        data = event.data
        self.interview_id = data.get("interview_id")
        self.index = data.get("index")
        self.text = data.get("text")
        self.speaker = data.get("speaker")
        self.start_ms = data.get("start_ms")
        self.end_ms = data.get("end_ms")
        self.status = SentenceStatus.CREATED
        self.created_at = event.occurred_at
        self.updated_at = event.occurred_at

    def _apply_sentence_edited(self, event: EventEnvelope) -> None:
        """Apply SentenceEdited event."""
        data = event.data
        self.text = data.get("new_text")
        self.status = SentenceStatus.EDITED
        self.updated_at = event.occurred_at

    def _apply_analysis_generated(self, event: EventEnvelope) -> None:
        """Apply AnalysisGenerated event."""
        data = event.data
        self.analysis_model = data.get("model")
        self.analysis_version = data.get("version")
        self.classification = data.get("classification", {})
        self.keywords = data.get("keywords", [])
        self.topics = data.get("topics", [])
        self.domain_keywords = data.get("domain_keywords", [])
        self.confidence = data.get("confidence")
        self.raw_ref = data.get("raw_ref")
        self.status = SentenceStatus.ANALYZED
        self.updated_at = event.occurred_at

    def _apply_analysis_regenerated(self, event: EventEnvelope) -> None:
        """Apply AnalysisRegenerated event."""
        data = event.data
        self.analysis_model = data.get("model")
        self.classification = data.get("classification", {})
        self.keywords = data.get("keywords", [])
        self.topics = data.get("topics", [])
        self.domain_keywords = data.get("domain_keywords", [])
        self.confidence = data.get("confidence")
        self.updated_at = event.occurred_at

    def _apply_analysis_overridden(self, event: EventEnvelope) -> None:
        """Apply AnalysisOverridden event."""
        data = event.data
        self.overridden_fields.update(data.get("fields_overridden", {}))
        self.override_note = data.get("note")
        self.updated_at = event.occurred_at

    def _apply_analysis_cleared(self, event: EventEnvelope) -> None:
        """Apply AnalysisCleared event."""
        self.analysis_model = None
        self.analysis_version = None
        self.classification = {}
        self.keywords = []
        self.topics = []
        self.domain_keywords = []
        self.confidence = None
        self.raw_ref = None
        self.overridden_fields = {}
        self.override_note = None
        self.updated_at = event.occurred_at

    # Command methods (business logic)
    def create(
        self,
        interview_id: str,
        index: int,
        text: str,
        speaker: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """
        Create a new sentence.

        Args:
            interview_id: UUID of the parent interview
            index: Sequence order within the interview
            text: The sentence text content
            speaker: Speaker identifier if available
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The SentenceCreated event
        """
        if self.version >= 0:
            raise ValueError("Sentence has already been created")

        return self._add_event(
            event_type="SentenceCreated",
            data={
                "interview_id": interview_id,
                "index": index,
                "text": text,
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
            **envelope_kwargs,
        )

    def edit(self, new_text: str, editor_type: EditorType, **envelope_kwargs) -> EventEnvelope:
        """
        Edit the sentence text.

        Args:
            new_text: Updated sentence text
            editor_type: Type of editor that made the change
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The SentenceEdited event
        """
        if self.version < 0:
            raise ValueError("Sentence must be created before editing")

        if self.text == new_text:
            raise ValueError("New text is the same as current text")

        return self._add_event(
            event_type="SentenceEdited",
            data={"old_text": self.text, "new_text": new_text, "editor_type": editor_type.value},
            **envelope_kwargs,
        )

    def generate_analysis(
        self,
        model: str,
        model_version: str,
        classification: Dict[str, Any],
        keywords: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        domain_keywords: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        raw_ref: Optional[str] = None,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """
        Generate AI analysis for the sentence.

        Args:
            model: AI model used for analysis
            model_version: Model version
            classification: Classification results
            keywords: Overall keywords identified
            topics: Topics identified
            domain_keywords: Domain-specific keywords
            confidence: Analysis confidence score
            raw_ref: Reference to raw analysis data
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The AnalysisGenerated event
        """
        if self.version < 0:
            raise ValueError("Sentence must be created before generating analysis")

        return self._add_event(
            event_type="AnalysisGenerated",
            data={
                "model": model,
                "version": model_version,
                "classification": classification,
                "keywords": keywords or [],
                "topics": topics or [],
                "domain_keywords": domain_keywords or [],
                "confidence": confidence,
                "raw_ref": raw_ref,
            },
            **envelope_kwargs,
        )

    def override_analysis(
        self, fields_overridden: Dict[str, Any], note: Optional[str] = None, **envelope_kwargs
    ) -> EventEnvelope:
        """
        Manually override analysis results.

        Args:
            fields_overridden: Fields that were manually overridden
            note: Note explaining the override
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The AnalysisOverridden event
        """
        if self.version < 0:
            raise ValueError("Sentence must be created before overriding analysis")

        return self._add_event(
            event_type="AnalysisOverridden",
            data={"fields_overridden": fields_overridden, "note": note},
            **envelope_kwargs,
        )
