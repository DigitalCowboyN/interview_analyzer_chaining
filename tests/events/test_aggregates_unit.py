"""
Unit tests for aggregates.py (src/events/aggregates.py).

Tests the domain aggregates: Interview and Sentence, including:
- Event application
- State transitions
- Command validation
- Error handling
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.events.aggregates import AggregateRoot, Interview, Sentence
from src.events.envelope import Actor, ActorType, AggregateType, EventEnvelope
from src.events.interview_events import InterviewStatus
from src.events.sentence_events import EditorType, SentenceStatus


def create_event(
    event_type: str,
    aggregate_type: AggregateType,
    aggregate_id: str,
    version: int,
    data: dict,
) -> EventEnvelope:
    """Helper to create test events."""
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        version=version,
        data=data,
        actor=Actor(actor_type=ActorType.SYSTEM),
        correlation_id=str(uuid.uuid4()),
    )


class TestAggregateRootBase:
    """Test AggregateRoot base class functionality."""

    def test_aggregate_init(self):
        """Test aggregate initialization."""
        interview = Interview(aggregate_id="test-123")

        assert interview.aggregate_id == "test-123"
        assert interview.version == -1  # New aggregate
        assert interview.get_uncommitted_events() == []

    def test_get_uncommitted_events_returns_copy(self):
        """Test that get_uncommitted_events returns a copy."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")

        events1 = interview.get_uncommitted_events()
        events2 = interview.get_uncommitted_events()

        assert events1 == events2
        assert events1 is not events2  # Should be different list objects

    def test_mark_events_as_committed(self):
        """Test clearing uncommitted events."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")

        assert len(interview.get_uncommitted_events()) == 1

        interview.mark_events_as_committed()

        assert len(interview.get_uncommitted_events()) == 0

    def test_load_from_history(self):
        """Test rebuilding aggregate from event history."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id=aggregate_id)

        # Create events simulating history
        events = [
            create_event(
                event_type="InterviewCreated",
                aggregate_type=AggregateType.INTERVIEW,
                aggregate_id=aggregate_id,
                version=0,
                data={"title": "Test Interview", "source": "test.txt", "language": "en"},
            ),
            create_event(
                event_type="StatusChanged",
                aggregate_type=AggregateType.INTERVIEW,
                aggregate_id=aggregate_id,
                version=1,
                data={"from_status": "created", "to_status": "processing"},
            ),
        ]

        interview.load_from_history(events)

        assert interview.version == 1
        assert interview.title == "Test Interview"
        assert interview.status == InterviewStatus.PROCESSING


class TestInterviewAggregate:
    """Test Interview aggregate."""

    def test_create_interview(self):
        """Test creating a new interview."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))

        event = interview.create(
            title="Technical Interview",
            source="interview.txt",
            language="en",
            actor=Actor(actor_type=ActorType.SYSTEM),
        )

        assert interview.title == "Technical Interview"
        assert interview.source == "interview.txt"
        assert interview.language == "en"
        assert interview.status == InterviewStatus.CREATED
        assert interview.version == 0
        assert event.event_type == "InterviewCreated"

    def test_create_interview_already_created_raises_error(self):
        """Test that creating an already-created interview raises error."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")

        with pytest.raises(ValueError, match="already been created"):
            interview.create(title="Another", source="another.txt")

    def test_update_interview(self):
        """Test updating interview information."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Original", source="test.txt")

        event = interview.update(
            title="Updated Title",
            language="es",
            metadata_diff={"key": "value"},
        )

        assert interview.title == "Updated Title"
        assert interview.language == "es"
        assert interview.metadata == {"key": "value"}
        assert event.event_type == "InterviewUpdated"

    def test_update_interview_not_created_raises_error(self):
        """Test that updating a non-created interview raises error."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))

        with pytest.raises(ValueError, match="must be created before updating"):
            interview.update(title="New Title")

    def test_change_status(self):
        """Test changing interview status."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")

        event = interview.change_status(
            new_status=InterviewStatus.PROCESSING,
            reason="Starting processing",
        )

        assert interview.status == InterviewStatus.PROCESSING
        assert event.event_type == "StatusChanged"
        assert event.data["from_status"] == "created"
        assert event.data["to_status"] == "processing"

    def test_change_status_not_created_raises_error(self):
        """Test that changing status of non-created interview raises error."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))

        with pytest.raises(ValueError, match="must be created before changing status"):
            interview.change_status(InterviewStatus.PROCESSING)

    def test_change_status_same_status_raises_error(self):
        """Test that changing to same status raises error."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")

        with pytest.raises(ValueError, match="already in status"):
            interview.change_status(InterviewStatus.CREATED)

    def test_apply_unknown_event_raises_error(self):
        """Test that applying unknown event type raises error."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))

        unknown_event = create_event(
            event_type="UnknownEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=interview.aggregate_id,
            version=0,
            data={},
        )

        with pytest.raises(ValueError, match="Unknown event type for Interview"):
            interview.apply_event(unknown_event)

    def test_apply_interview_archived(self):
        """Test applying InterviewArchived event."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id=aggregate_id)

        # First create the interview
        created_event = create_event(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=aggregate_id,
            version=0,
            data={"title": "Test", "source": "test.txt"},
        )
        interview.apply_event(created_event)

        # Then archive it
        archived_event = create_event(
            event_type="InterviewArchived",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=aggregate_id,
            version=1,
            data={"reason": "completed"},
        )
        interview.apply_event(archived_event)

        assert interview.status == InterviewStatus.ARCHIVED

    def test_apply_interview_deleted(self):
        """Test applying InterviewDeleted event."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id=aggregate_id)

        # First create the interview
        created_event = create_event(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=aggregate_id,
            version=0,
            data={"title": "Test", "source": "test.txt"},
        )
        interview.apply_event(created_event)

        # Then delete it
        deleted_event = create_event(
            event_type="InterviewDeleted",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=aggregate_id,
            version=1,
            data={"reason": "user requested"},
        )
        interview.apply_event(deleted_event)

        # Should still have state (event sourcing doesn't truly delete)
        assert interview.title == "Test"
        assert interview.updated_at is not None

    def test_apply_interview_created_with_string_datetime(self):
        """Test applying InterviewCreated with datetime as ISO string."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id=aggregate_id)

        created_event = create_event(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=aggregate_id,
            version=0,
            data={
                "title": "Test",
                "source": "test.txt",
                "started_at": "2024-01-15T10:30:00Z",
            },
        )
        interview.apply_event(created_event)

        assert interview.started_at is not None
        assert isinstance(interview.started_at, datetime)


class TestSentenceAggregate:
    """Test Sentence aggregate."""

    def test_create_sentence(self):
        """Test creating a new sentence."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        interview_id = str(uuid.uuid4())

        event = sentence.create(
            interview_id=interview_id,
            index=0,
            text="This is a test sentence.",
            speaker="Interviewer",
            start_ms=1000,
            end_ms=5000,
        )

        assert sentence.interview_id == interview_id
        assert sentence.index == 0
        assert sentence.text == "This is a test sentence."
        assert sentence.speaker == "Interviewer"
        assert sentence.start_ms == 1000
        assert sentence.end_ms == 5000
        assert sentence.status == SentenceStatus.CREATED
        assert event.event_type == "SentenceCreated"

    def test_create_sentence_already_created_raises_error(self):
        """Test that creating an already-created sentence raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Test")

        with pytest.raises(ValueError, match="already been created"):
            sentence.create(interview_id=str(uuid.uuid4()), index=1, text="Another")

    def test_edit_sentence(self):
        """Test editing a sentence."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Original text")

        event = sentence.edit(
            new_text="Updated text",
            editor_type=EditorType.HUMAN,
        )

        assert sentence.text == "Updated text"
        assert sentence.status == SentenceStatus.EDITED
        assert event.event_type == "SentenceEdited"
        assert event.data["old_text"] == "Original text"
        assert event.data["new_text"] == "Updated text"

    def test_edit_sentence_not_created_raises_error(self):
        """Test that editing a non-created sentence raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))

        with pytest.raises(ValueError, match="must be created before editing"):
            sentence.edit(new_text="New text", editor_type=EditorType.HUMAN)

    def test_edit_sentence_same_text_raises_error(self):
        """Test that editing with same text raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Same text")

        with pytest.raises(ValueError, match="same as current text"):
            sentence.edit(new_text="Same text", editor_type=EditorType.HUMAN)

    def test_generate_analysis(self):
        """Test generating analysis for a sentence."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Test sentence")

        event = sentence.generate_analysis(
            model="gpt-4",
            model_version="1.0",
            classification={"function": "declarative", "purpose": "information"},
            keywords=["test", "sentence"],
            topics=["testing"],
            domain_keywords=["unit_test"],
            confidence=0.95,
            raw_ref="ref-123",
        )

        assert sentence.analysis_model == "gpt-4"
        assert sentence.analysis_version == "1.0"
        assert sentence.classification == {"function": "declarative", "purpose": "information"}
        assert sentence.keywords == ["test", "sentence"]
        assert sentence.topics == ["testing"]
        assert sentence.domain_keywords == ["unit_test"]
        assert sentence.confidence == 0.95
        assert sentence.status == SentenceStatus.ANALYZED
        assert event.event_type == "AnalysisGenerated"

    def test_generate_analysis_not_created_raises_error(self):
        """Test that generating analysis on non-created sentence raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))

        with pytest.raises(ValueError, match="must be created before generating"):
            sentence.generate_analysis(
                model="gpt-4",
                model_version="1.0",
                classification={},
            )

    def test_override_analysis(self):
        """Test overriding analysis results."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Test")
        sentence.generate_analysis(model="gpt-4", model_version="1.0", classification={})

        event = sentence.override_analysis(
            fields_overridden={"purpose": "corrected_purpose"},
            note="Manual correction by reviewer",
        )

        assert sentence.overridden_fields == {"purpose": "corrected_purpose"}
        assert sentence.override_note == "Manual correction by reviewer"
        assert event.event_type == "AnalysisOverridden"

    def test_override_analysis_not_created_raises_error(self):
        """Test that overriding analysis on non-created sentence raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))

        with pytest.raises(ValueError, match="must be created before overriding"):
            sentence.override_analysis(fields_overridden={"purpose": "test"})

    def test_apply_unknown_event_raises_error(self):
        """Test that applying unknown event type raises error."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))

        unknown_event = create_event(
            event_type="UnknownEvent",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=sentence.aggregate_id,
            version=0,
            data={},
        )

        with pytest.raises(ValueError, match="Unknown event type for Sentence"):
            sentence.apply_event(unknown_event)

    def test_apply_analysis_regenerated(self):
        """Test applying AnalysisRegenerated event."""
        aggregate_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id=aggregate_id)

        # First create and analyze
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Test")
        sentence.generate_analysis(
            model="gpt-3.5",
            model_version="1.0",
            classification={"function": "old"},
        )

        # Apply regeneration event
        regen_event = create_event(
            event_type="AnalysisRegenerated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=aggregate_id,
            version=2,
            data={
                "model": "gpt-4",
                "classification": {"function": "new"},
                "keywords": ["updated"],
                "topics": ["new_topic"],
                "domain_keywords": ["new_domain"],
                "confidence": 0.99,
            },
        )
        sentence.apply_event(regen_event)

        assert sentence.analysis_model == "gpt-4"
        assert sentence.classification == {"function": "new"}
        assert sentence.keywords == ["updated"]

    def test_apply_analysis_cleared(self):
        """Test applying AnalysisCleared event."""
        aggregate_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id=aggregate_id)

        # First create and analyze
        sentence.create(interview_id=str(uuid.uuid4()), index=0, text="Test")
        sentence.generate_analysis(
            model="gpt-4",
            model_version="1.0",
            classification={"function": "declarative"},
            keywords=["test"],
            confidence=0.9,
        )

        # Apply cleared event
        cleared_event = create_event(
            event_type="AnalysisCleared",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=aggregate_id,
            version=2,
            data={"reason": "reset requested"},
        )
        sentence.apply_event(cleared_event)

        assert sentence.analysis_model is None
        assert sentence.analysis_version is None
        assert sentence.classification == {}
        assert sentence.keywords == []
        assert sentence.topics == []
        assert sentence.domain_keywords == []
        assert sentence.confidence is None
        assert sentence.raw_ref is None
        assert sentence.overridden_fields == {}
        assert sentence.override_note is None


class TestAggregateTypeDetection:
    """Test aggregate type detection in _add_event."""

    def test_interview_aggregate_type(self):
        """Test that Interview correctly identifies its aggregate type."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        event = interview.create(title="Test", source="test.txt")

        assert event.aggregate_type == AggregateType.INTERVIEW

    def test_sentence_aggregate_type(self):
        """Test that Sentence correctly identifies its aggregate type."""
        sentence = Sentence(aggregate_id=str(uuid.uuid4()))
        event = sentence.create(
            interview_id=str(uuid.uuid4()),
            index=0,
            text="Test",
        )

        assert event.aggregate_type == AggregateType.SENTENCE


class TestVersionManagement:
    """Test version tracking across events."""

    def test_version_increments_with_events(self):
        """Test that version increments correctly with each event."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))

        assert interview.version == -1  # New aggregate

        interview.create(title="Test", source="test.txt")
        assert interview.version == 0

        interview.change_status(InterviewStatus.PROCESSING)
        assert interview.version == 1

        interview.update(title="Updated")
        assert interview.version == 2

    def test_uncommitted_events_have_correct_versions(self):
        """Test that uncommitted events have sequential versions."""
        interview = Interview(aggregate_id=str(uuid.uuid4()))
        interview.create(title="Test", source="test.txt")
        interview.change_status(InterviewStatus.PROCESSING)

        events = interview.get_uncommitted_events()

        assert len(events) == 2
        assert events[0].version == 0
        assert events[1].version == 1
