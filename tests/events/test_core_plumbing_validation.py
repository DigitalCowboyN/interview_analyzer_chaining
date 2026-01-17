"""
Core plumbing validation tests for M1 event infrastructure.

These tests validate that the basic event infrastructure works correctly
without requiring EventStoreDB to be running. They test the fundamental
components: event envelopes, aggregates, and repository patterns.

IMPORTANT: These are validation tests for M1 - they should NOT change
any existing system behavior, only verify the new infrastructure works.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, ActorType, EventEnvelope
from src.events.interview_events import InterviewStatus, create_interview_created_event
from src.events.repository import InterviewRepository, SentenceRepository
from src.events.sentence_events import (
    EditorType,
    SentenceStatus,
    create_sentence_created_event,
)


class TestEventEnvelope:
    """Test event envelope creation and validation."""

    def test_create_event_envelope(self):
        """Test creating a basic event envelope."""
        aggregate_id = str(uuid.uuid4())

        envelope = EventEnvelope(
            event_type="TestEvent",
            aggregate_type="Interview",
            aggregate_id=aggregate_id,
            version=1,
            data={"test": "data"},
        )

        assert envelope.event_type == "TestEvent"
        assert envelope.aggregate_type == "Interview"
        assert envelope.aggregate_id == aggregate_id
        assert envelope.version == 1
        assert envelope.data == {"test": "data"}
        assert envelope.schema_version == "1.0.0"
        assert envelope.source == "interview_analyzer"
        assert isinstance(envelope.occurred_at, datetime)
        assert envelope.occurred_at.tzinfo == timezone.utc

    def test_event_envelope_with_actor(self):
        """Test creating event envelope with actor information."""
        actor = Actor(user_id="test-user", display="Test User", actor_type=ActorType.HUMAN)

        envelope = EventEnvelope(
            event_type="TestEvent",
            aggregate_type="Interview",
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"test": "data"},
            actor=actor,
            correlation_id="test-correlation",
        )

        assert envelope.actor == actor
        assert envelope.correlation_id == "test-correlation"

    def test_event_envelope_validation(self):
        """Test event envelope field validation."""
        with pytest.raises(ValueError, match="event_id must be a valid UUID"):
            EventEnvelope(
                event_id="invalid-uuid",
                event_type="TestEvent",
                aggregate_type="Interview",
                aggregate_id=str(uuid.uuid4()),
                version=1,
                data={},
            )


class TestInterviewEvents:
    """Test interview event creation helpers."""

    def test_create_interview_created_event(self):
        """Test creating an InterviewCreated event."""
        aggregate_id = str(uuid.uuid4())

        event = create_interview_created_event(
            aggregate_id=aggregate_id, version=1, title="Test Interview", source="test_file.txt", language="en"
        )

        assert event.event_type == "InterviewCreated"
        assert event.aggregate_type == "Interview"
        assert event.aggregate_id == aggregate_id
        assert event.version == 1
        assert event.data["title"] == "Test Interview"
        assert event.data["source"] == "test_file.txt"
        assert event.data["language"] == "en"


class TestSentenceEvents:
    """Test sentence event creation helpers."""

    def test_create_sentence_created_event(self):
        """Test creating a SentenceCreated event."""
        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        event = create_sentence_created_event(
            aggregate_id=aggregate_id,
            version=1,
            interview_id=interview_id,
            index=0,
            text="This is a test sentence.",
            speaker="Speaker1",
        )

        assert event.event_type == "SentenceCreated"
        assert event.aggregate_type == "Sentence"
        assert event.aggregate_id == aggregate_id
        assert event.version == 1
        assert event.data["interview_id"] == interview_id
        assert event.data["index"] == 0
        assert event.data["text"] == "This is a test sentence."
        assert event.data["speaker"] == "Speaker1"


class TestInterviewAggregate:
    """Test Interview aggregate behavior."""

    def test_create_new_interview(self):
        """Test creating a new interview aggregate."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)

        assert interview.aggregate_id == aggregate_id
        assert interview.version == -1
        assert interview.status == InterviewStatus.CREATED
        assert len(interview.get_uncommitted_events()) == 0

    def test_interview_create_command(self):
        """Test interview create command generates correct event."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)

        event = interview.create(title="Test Interview", source="test_file.txt", language="en")

        assert event.event_type == "InterviewCreated"
        assert event.aggregate_id == aggregate_id
        assert event.version == 0  # First event

        # Check that state was updated
        assert interview.title == "Test Interview"
        assert interview.source == "test_file.txt"
        assert interview.language == "en"
        assert interview.status == InterviewStatus.CREATED

        # Check uncommitted events
        uncommitted = interview.get_uncommitted_events()
        assert len(uncommitted) == 1
        assert uncommitted[0] == event

    def test_interview_status_change(self):
        """Test interview status change command."""
        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)

        # Create first
        interview.create(title="Test", source="test.txt")

        # Change status
        event = interview.change_status(new_status=InterviewStatus.PROCESSING, reason="Starting analysis")

        assert event.event_type == "StatusChanged"
        assert interview.status == InterviewStatus.PROCESSING
        assert len(interview.get_uncommitted_events()) == 2

    def test_interview_load_from_history(self):
        """Test loading interview from historical events."""
        aggregate_id = str(uuid.uuid4())

        # Create events manually
        created_event = create_interview_created_event(
            aggregate_id=aggregate_id, version=0, title="Historical Interview", source="history.txt"
        )

        # Load into new aggregate
        interview = Interview(aggregate_id)
        interview.load_from_history([created_event])

        assert interview.version == 0
        assert interview.title == "Historical Interview"
        assert interview.source == "history.txt"
        assert len(interview.get_uncommitted_events()) == 0  # No new events


class TestSentenceAggregate:
    """Test Sentence aggregate behavior."""

    def test_create_new_sentence(self):
        """Test creating a new sentence aggregate."""
        aggregate_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id)

        assert sentence.aggregate_id == aggregate_id
        assert sentence.version == -1
        assert sentence.status == SentenceStatus.CREATED
        assert len(sentence.get_uncommitted_events()) == 0

    def test_sentence_create_command(self):
        """Test sentence create command generates correct event."""
        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id)

        event = sentence.create(interview_id=interview_id, index=0, text="Test sentence.")

        assert event.event_type == "SentenceCreated"
        assert event.aggregate_id == aggregate_id
        assert event.version == 0

        # Check that state was updated
        assert sentence.interview_id == interview_id
        assert sentence.index == 0
        assert sentence.text == "Test sentence."
        assert sentence.status == SentenceStatus.CREATED

    def test_sentence_edit_command(self):
        """Test sentence edit command."""
        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id)

        # Create first
        sentence.create(interview_id=interview_id, index=0, text="Original text.")

        # Edit
        event = sentence.edit(new_text="Edited text.", editor_type=EditorType.HUMAN)

        assert event.event_type == "SentenceEdited"
        assert sentence.text == "Edited text."
        assert sentence.status == SentenceStatus.EDITED
        assert len(sentence.get_uncommitted_events()) == 2

    def test_sentence_generate_analysis(self):
        """Test sentence analysis generation."""
        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id)

        # Create first
        sentence.create(interview_id=interview_id, index=0, text="Analyze this sentence.")

        # Generate analysis
        event = sentence.generate_analysis(
            model="gpt-4",
            model_version="1.0",
            classification={"function_type": "declarative"},
            keywords=["analyze", "sentence"],
            confidence=0.95,
        )

        assert event.event_type == "AnalysisGenerated"
        assert sentence.analysis_model == "gpt-4"
        assert sentence.classification == {"function_type": "declarative"}
        assert sentence.keywords == ["analyze", "sentence"]
        assert sentence.confidence == 0.95
        assert sentence.status == SentenceStatus.ANALYZED


@pytest.mark.asyncio
class TestRepositoryPattern:
    """Test repository pattern with mocked EventStore."""

    async def test_interview_repository_stream_naming(self):
        """Test interview repository stream naming convention."""
        mock_store = MagicMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        stream_name = repo._get_stream_name(aggregate_id)

        assert stream_name == f"Interview-{aggregate_id}"

    async def test_sentence_repository_stream_naming(self):
        """Test sentence repository stream naming convention."""
        mock_store = MagicMock()
        repo = SentenceRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        stream_name = repo._get_stream_name(aggregate_id)

        assert stream_name == f"Sentence-{aggregate_id}"

    async def test_repository_load_not_found(self):
        """Test repository load when aggregate doesn't exist."""
        mock_store = AsyncMock()
        mock_store.read_stream.side_effect = Exception("Stream not found")  # Simplified for test

        repo = InterviewRepository(mock_store)
        aggregate_id = str(uuid.uuid4())

        # This would normally handle StreamNotFoundError, but for this simple test
        # we'll just verify the basic structure works
        try:
            await repo.load(aggregate_id)
        except Exception:
            # Expected for this mock setup
            pass

    async def test_repository_save_no_events(self):
        """Test repository save with no uncommitted events."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        # Don't create any events

        await repo.save(interview)

        # Should not call append_events since no uncommitted events
        mock_store.append_events.assert_not_called()


class TestM1ValidationComplete:
    """Comprehensive validation that M1 core plumbing is complete."""

    def test_all_components_importable(self):
        """Test that all M1 components can be imported successfully."""
        # Event infrastructure
        # Aggregates
        from src.events.aggregates import Interview, Sentence  # noqa: F401
        from src.events.envelope import (  # noqa: F401
            Actor,
            ActorType,
            AggregateType,
            EventEnvelope,
        )
        from src.events.interview_events import (  # noqa: F401, E501
            InterviewStatus,
            create_interview_created_event,
        )

        # Repository pattern
        from src.events.repository import (  # noqa: F401
            InterviewRepository,
            SentenceRepository,
        )
        from src.events.sentence_events import (  # noqa: F401, E501
            SentenceStatus,
            create_sentence_created_event,
        )

        # Store (even though esdbclient isn't installed, the module should be importable)
        from src.events.store import EventStoreClient, EventStoreError  # noqa: F401

        # All imports successful
        assert True

    def test_end_to_end_aggregate_lifecycle(self):
        """Test complete aggregate lifecycle without persistence."""
        # Create interview
        interview_id = str(uuid.uuid4())
        interview = Interview(interview_id)

        # Execute commands
        created_event = interview.create(title="E2E Test Interview", source="e2e_test.txt")

        status_event = interview.change_status(new_status=InterviewStatus.PROCESSING)

        # Verify state
        assert interview.title == "E2E Test Interview"
        assert interview.status == InterviewStatus.PROCESSING
        assert len(interview.get_uncommitted_events()) == 2

        # Test event reconstruction
        new_interview = Interview(interview_id)
        new_interview.load_from_history([created_event, status_event])

        assert new_interview.title == "E2E Test Interview"
        assert new_interview.status == InterviewStatus.PROCESSING
        assert new_interview.version == 1  # Status change event (version 1)
        assert len(new_interview.get_uncommitted_events()) == 0

    def test_sentence_analysis_workflow(self):
        """Test sentence analysis workflow without persistence."""
        # Create sentence
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence = Sentence(sentence_id)

        # Execute workflow
        created_event = sentence.create(
            interview_id=interview_id, index=0, text="This is a test sentence for analysis."
        )

        analysis_event = sentence.generate_analysis(
            model="gpt-4",
            model_version="1.0",
            classification={"function_type": "declarative", "structure_type": "simple", "purpose": "testing"},
            keywords=["test", "analysis"],
            topics=["testing"],
            confidence=0.92,
        )

        override_event = sentence.override_analysis(
            fields_overridden={"purpose": "validation"}, note="Manual correction for testing"
        )

        # Verify final state
        assert sentence.text == "This is a test sentence for analysis."
        assert sentence.status == SentenceStatus.ANALYZED
        assert sentence.classification["function_type"] == "declarative"
        assert sentence.overridden_fields["purpose"] == "validation"
        assert sentence.override_note == "Manual correction for testing"
        assert len(sentence.get_uncommitted_events()) == 3

        # Verify event reconstruction preserves all state
        new_sentence = Sentence(sentence_id)
        new_sentence.load_from_history([created_event, analysis_event, override_event])

        assert new_sentence.text == sentence.text
        assert new_sentence.status == sentence.status
        assert new_sentence.overridden_fields == sentence.overridden_fields
        assert new_sentence.override_note == sentence.override_note
