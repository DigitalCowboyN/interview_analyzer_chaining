"""
Aggregate root base classes and domain aggregates.

Provides the foundational classes for event-sourced aggregates with
proper version tracking, event application, and state management.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .envelope import Actor, EventEnvelope
from .interview_events import (
    ClaimExtractedData,
    InterruptionRecordedData,
    InterviewStatus,
    SpeakerCreatedData,
    SpeakerMergedData,
    SpeakerRenamedData,
    StitchRemovedData,
    UtteranceIdentifiedData,
)
from .sentence_events import (
    EditorType,
    EntitiesExtractedData,
    SentenceStatus,
    SpeakerAttributedData,
    SpeakerReattributedData,
)


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

        # Conversation structure (Layer 1): speakers and utterances keyed by id
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self.utterances: Dict[str, Dict[str, Any]] = {}

        # Enrichment (Layer 2): claims keyed by claim_id
        self.claims: Dict[str, Dict[str, Any]] = {}

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
        elif event.event_type == "SpeakerCreated":
            self._apply_speaker_created(event)
        elif event.event_type == "SpeakerRenamed":
            self._apply_speaker_renamed(event)
        elif event.event_type == "SpeakerMerged":
            self._apply_speaker_merged(event)
        elif event.event_type == "UtteranceIdentified":
            self._apply_utterance_identified(event)
        elif event.event_type == "InterruptionRecorded":
            self._apply_interruption_recorded(event)
        elif event.event_type == "StitchRemoved":
            self._apply_stitch_removed(event)
        elif event.event_type == "ClaimExtracted":
            self._apply_claim_extracted(event)
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

    def _apply_speaker_created(self, event: EventEnvelope) -> None:
        """Apply SpeakerCreated event."""
        data = event.data
        self.speakers[data["speaker_id"]] = {
            "handle": data["handle"],
            "display_name": data["display_name"],
            "provisional": data["provisional"],
            "merged_into": None,
        }
        self.updated_at = event.occurred_at

    def _apply_speaker_renamed(self, event: EventEnvelope) -> None:
        """Apply SpeakerRenamed event (human correction; confirms the speaker)."""
        data = event.data
        speaker = self.speakers[data["speaker_id"]]
        speaker["display_name"] = data["new_display_name"]
        speaker["provisional"] = False
        self.updated_at = event.occurred_at

    def _apply_speaker_merged(self, event: EventEnvelope) -> None:
        """Apply SpeakerMerged event."""
        data = event.data
        self.speakers[data["merged_speaker_id"]]["merged_into"] = data["surviving_speaker_id"]
        self.updated_at = event.occurred_at

    def _apply_utterance_identified(self, event: EventEnvelope) -> None:
        """Apply UtteranceIdentified event."""
        data = event.data
        self.utterances[data["utterance_id"]] = {
            "speaker_id": data["speaker_id"],
            "fragment_ids": list(data["fragment_ids"]),
            "removed": False,
        }
        self.updated_at = event.occurred_at

    def _apply_interruption_recorded(self, event: EventEnvelope) -> None:
        """Apply InterruptionRecorded event (relationship only; no state to mutate)."""
        self.updated_at = event.occurred_at

    def _apply_stitch_removed(self, event: EventEnvelope) -> None:
        """Apply StitchRemoved event (human correction)."""
        self.utterances[event.data["utterance_id"]]["removed"] = True
        self.updated_at = event.occurred_at

    def _apply_claim_extracted(self, event: EventEnvelope) -> None:
        """Apply ClaimExtracted event (Layer 2 enrichment)."""
        data = event.data
        self.claims[data["claim_id"]] = {
            "utterance_id": data["utterance_id"],
            "speaker_id": data["speaker_id"],
            "text": data["text"],
            "kind": data["kind"],
            "confidence": data["confidence"],
        }
        self.updated_at = event.occurred_at

    # Command methods (business logic)
    def create(
        self,
        title: str,
        source: str,
        language: Optional[str] = None,
        started_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
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
            project_id: Project/tenant ID; carried in BOTH the event data (the
                projection handler reads it there) and the envelope (queryable)
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The InterviewCreated event
        """
        if self.version >= 0:
            raise ValueError("Interview has already been created")

        data = {
            "title": title,
            "source": source,
            "language": language,
            "started_at": started_at.isoformat() if started_at else None,
            "metadata": metadata or {},
        }
        # Omit the key when absent so the projection handler's
        # data.get("project_id", "default") fallback still applies.
        if project_id is not None:
            data["project_id"] = project_id

        return self._add_event(
            event_type="InterviewCreated",
            data=data,
            project_id=project_id,
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

    def add_speaker(
        self,
        speaker_id: str,
        handle: str,
        display_name: str,
        provisional: bool,
        confidence: float,
        method: str,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Register a speaker discovered by parsing or inference."""
        if self.version < 0:
            raise ValueError("Interview must be created before adding speakers")
        if speaker_id in self.speakers:
            raise ValueError(f"Speaker {speaker_id} already exists")

        data = SpeakerCreatedData(
            speaker_id=speaker_id,
            handle=handle,
            display_name=display_name,
            provisional=provisional,
            confidence=confidence,
            method=method,
        )

        return self._add_event(
            event_type="SpeakerCreated",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def rename_speaker(self, speaker_id: str, new_display_name: str, **envelope_kwargs) -> EventEnvelope:
        """Human correction: give a provisional speaker a real name."""
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}")
        if self.speakers[speaker_id]["merged_into"] is not None:
            raise ValueError(f"Speaker {speaker_id} has already been merged")

        data = SpeakerRenamedData(
            speaker_id=speaker_id,
            old_display_name=self.speakers[speaker_id]["display_name"],
            new_display_name=new_display_name,
        )

        return self._add_event(
            event_type="SpeakerRenamed",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def merge_speakers(
        self, surviving_speaker_id: str, merged_speaker_id: str, **envelope_kwargs
    ) -> EventEnvelope:
        """Human correction: two provisional handles were the same person."""
        if surviving_speaker_id == merged_speaker_id:
            raise ValueError("Cannot merge a speaker into itself")
        for sid in (surviving_speaker_id, merged_speaker_id):
            if sid not in self.speakers:
                raise ValueError(f"Unknown speaker: {sid}")
            if self.speakers[sid]["merged_into"] is not None:
                raise ValueError(f"Speaker {sid} has already been merged")

        data = SpeakerMergedData(
            surviving_speaker_id=surviving_speaker_id,
            merged_speaker_id=merged_speaker_id,
        )

        return self._add_event(
            event_type="SpeakerMerged",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def identify_utterance(
        self,
        utterance_id: str,
        speaker_id: str,
        fragment_ids: List[str],
        confidence: float,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Record a stitched utterance: one speaker's continuous thought (overlay only)."""
        if self.version < 0:
            raise ValueError("Interview must be created before identifying utterances")
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}")
        if not fragment_ids:
            raise ValueError("Utterance requires at least one fragment")
        if utterance_id in self.utterances:
            raise ValueError(f"Utterance {utterance_id} already identified")

        data = UtteranceIdentifiedData(
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            fragment_ids=fragment_ids,
            confidence=confidence,
        )

        return self._add_event(
            event_type="UtteranceIdentified",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def record_interruption(
        self,
        interrupting_utterance_id: str,
        interrupted_utterance_id: str,
        at_fragment_id: str,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Record that one utterance broke into another."""
        for uid in (interrupting_utterance_id, interrupted_utterance_id):
            if uid not in self.utterances:
                raise ValueError(f"Unknown utterance: {uid}")

        data = InterruptionRecordedData(
            interrupting_utterance_id=interrupting_utterance_id,
            interrupted_utterance_id=interrupted_utterance_id,
            at_fragment_id=at_fragment_id,
        )

        return self._add_event(
            event_type="InterruptionRecorded",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def remove_stitch(
        self, utterance_id: str, reason: Optional[str] = None, **envelope_kwargs
    ) -> EventEnvelope:
        """Human correction: an identified utterance was wrong; remove the overlay."""
        if utterance_id not in self.utterances:
            raise ValueError(f"Unknown utterance: {utterance_id}")
        if self.utterances[utterance_id].get("removed"):
            raise ValueError(f"Utterance {utterance_id} stitch already removed")

        data = StitchRemovedData(utterance_id=utterance_id, reason=reason)

        return self._add_event(
            event_type="StitchRemoved",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def record_claim(
        self,
        claim_id: str,
        utterance_id: str,
        text: str,
        kind: str,
        confidence: float,
        model: str,
        provider: str,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Record a claim extracted from an utterance (Layer 2 enrichment).

        The speaker is derived from the utterance — a claim is always made by
        whoever spoke the utterance it came from.
        """
        if self.version < 0:
            raise ValueError("Interview must be created before recording claims")
        if utterance_id not in self.utterances:
            raise ValueError(f"Unknown utterance: {utterance_id}")
        if self.utterances[utterance_id].get("removed"):
            raise ValueError(f"Utterance {utterance_id} has been removed")
        if claim_id in self.claims:
            raise ValueError(f"Claim {claim_id} already recorded")

        data = ClaimExtractedData(
            claim_id=claim_id,
            utterance_id=utterance_id,
            speaker_id=self.utterances[utterance_id]["speaker_id"],
            text=text,
            kind=kind,
            confidence=confidence,
            model=model,
            provider=provider,
        )

        return self._add_event(
            event_type="ClaimExtracted",
            data=data.model_dump(),
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

        # Map grounding and speaker attribution (Layer 1)
        self.start_char: Optional[int] = None
        self.end_char: Optional[int] = None
        self.speaker_id: Optional[str] = None
        self.speaker_confidence: Optional[float] = None
        self.speaker_locked: bool = False

        # Enrichment v2 (Layer 2)
        self.dimension_confidences: Dict[str, float] = {}
        self.flags: Dict[str, str] = {}
        self.analysis_provider: Optional[str] = None
        self.entities: List[Dict[str, Any]] = []

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
        elif event.event_type == "SpeakerAttributed":
            self._apply_speaker_attributed(event)
        elif event.event_type == "SpeakerReattributed":
            self._apply_speaker_reattributed(event)
        elif event.event_type == "EntitiesExtracted":
            self._apply_entities_extracted(event)
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
        self.start_char = data.get("start_char")
        self.end_char = data.get("end_char")
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
        self.dimension_confidences = data.get("dimension_confidences", {})
        self.flags = data.get("flags", {})
        self.analysis_provider = data.get("provider")
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

    def _apply_speaker_attributed(self, event: EventEnvelope) -> None:
        """Apply SpeakerAttributed event."""
        self.speaker_id = event.data.get("speaker_id")
        self.speaker_confidence = event.data.get("confidence")
        self.updated_at = event.occurred_at

    def _apply_speaker_reattributed(self, event: EventEnvelope) -> None:
        """Apply SpeakerReattributed event (human correction locks attribution)."""
        self.speaker_id = event.data.get("new_speaker_id")
        self.speaker_confidence = 1.0
        self.speaker_locked = True
        self.updated_at = event.occurred_at

    def _apply_entities_extracted(self, event: EventEnvelope) -> None:
        """Apply EntitiesExtracted event (re-extraction replaces the set)."""
        self.entities = list(event.data.get("entities", []))
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
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
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
            start_char: Offset into the immutable source text
            end_char: End offset into the immutable source text
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
                "start_char": start_char,
                "end_char": end_char,
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

    def attribute_speaker(
        self, speaker_id: str, confidence: float, method: str, **envelope_kwargs
    ) -> EventEnvelope:
        """Attribute this fragment to a speaker (system inference or parsed label)."""
        if self.version < 0:
            raise ValueError("Sentence must be created before attributing a speaker")
        if self.speaker_locked:
            raise ValueError("Speaker attribution is locked by a human correction")

        # Validate through the payload model so out-of-range confidence is
        # rejected at command time, not just at (optional) deserialization.
        # interview_id rides in the payload because projection lane routing
        # partitions Sentence-stream events by data["interview_id"].
        data = SpeakerAttributedData(
            interview_id=self.interview_id,
            speaker_id=speaker_id,
            confidence=confidence,
            method=method,
        )

        return self._add_event(
            event_type="SpeakerAttributed",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def record_entities(
        self, entities: List[Dict[str, Any]], model: str, provider: str, **envelope_kwargs
    ) -> EventEnvelope:
        """Record span-grounded entity mentions (Layer 2 enrichment).

        interview_id rides in the payload because projection lane routing
        partitions Sentence-stream events by data["interview_id"].
        """
        if self.version < 0:
            raise ValueError("Sentence must be created before recording entities")

        data = EntitiesExtractedData(
            interview_id=self.interview_id,
            entities=entities,
            model=model,
            provider=provider,
        )

        return self._add_event(
            event_type="EntitiesExtracted",
            data=data.model_dump(),
            **envelope_kwargs,
        )

    def reattribute_speaker(self, new_speaker_id: str, **envelope_kwargs) -> EventEnvelope:
        """Human correction of speaker attribution; locks against system overwrite."""
        if self.version < 0:
            raise ValueError("Sentence must be created before reattributing a speaker")

        data = SpeakerReattributedData(
            interview_id=self.interview_id,
            old_speaker_id=self.speaker_id,
            new_speaker_id=new_speaker_id,
        )

        return self._add_event(
            event_type="SpeakerReattributed",
            data=data.model_dump(),
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
        dimension_confidences: Optional[Dict[str, float]] = None,
        flags: Optional[Dict[str, str]] = None,
        provider: Optional[str] = None,
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
            dimension_confidences: Per-dimension numeric confidence (0-1)
            flags: Review flags (e.g., spaCy disagreement, invalid responses)
            provider: Provider that served the calls (chain provenance)
            **envelope_kwargs: Additional envelope fields

        Returns:
            EventEnvelope: The AnalysisGenerated event
        """
        if self.version < 0:
            raise ValueError("Sentence must be created before generating analysis")

        # Validate through the payload model (command-time bounds enforcement).
        from .sentence_events import AnalysisGeneratedData

        data = AnalysisGeneratedData(
            model=model,
            version=model_version,
            classification=classification,
            keywords=keywords or [],
            topics=topics or [],
            domain_keywords=domain_keywords or [],
            confidence=confidence,
            raw_ref=raw_ref,
            dimension_confidences=dimension_confidences or {},
            flags=flags or {},
            provider=provider,
        )

        return self._add_event(
            event_type="AnalysisGenerated",
            data=data.model_dump(),
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
