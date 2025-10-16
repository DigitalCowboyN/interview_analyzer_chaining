"""
Unit tests for PipelineEventEmitter.

Tests event emission logic with mocked EventStoreClient.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.events.envelope import ActorType, AggregateType
from src.pipeline_event_emitter import PipelineEventEmitter


@pytest.fixture
def mock_event_store():
    """Create a mocked EventStoreClient."""
    mock = MagicMock()
    mock.append_events = AsyncMock()
    return mock


@pytest.fixture
def emitter(mock_event_store):
    """Create PipelineEventEmitter with mocked client."""
    return PipelineEventEmitter(mock_event_store)


@pytest.mark.asyncio
class TestPipelineEventEmitter:
    """Test PipelineEventEmitter functionality."""

    async def test_emit_interview_created_success(self, emitter, mock_event_store):
        """Test successful InterviewCreated event emission."""
        interview_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        await emitter.emit_interview_created(
            interview_id=interview_id,
            project_id=project_id,
            title="test.txt",
            source="/path/to/test.txt",
            language="en",
            correlation_id=correlation_id,
        )

        # Verify append_events was called
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args

        # Check stream name
        stream_name = call_args[0][0]
        assert stream_name == f"Interview-{interview_id}"

        # Check events list
        events = call_args[0][1]
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "InterviewCreated"
        assert event.aggregate_type == AggregateType.INTERVIEW
        assert event.aggregate_id == interview_id
        assert event.version == 0
        assert event.correlation_id == correlation_id
        assert event.project_id == project_id

        # Check actor
        assert event.actor.actor_type == ActorType.SYSTEM
        assert event.actor.user_id == "pipeline"

        # Check data
        assert event.data["title"] == "test.txt"
        assert event.data["source"] == "/path/to/test.txt"
        assert event.data["language"] == "en"
        # project_id is in envelope, not in data

        # Check expected_version
        expected_version = call_args[1]["expected_version"]
        assert expected_version == -1  # Allow any version

    async def test_emit_interview_created_handles_exception(
        self, emitter, mock_event_store, caplog
    ):
        """Test that InterviewCreated emission logs errors but doesn't raise."""
        mock_event_store.append_events.side_effect = Exception("ESDB connection failed")

        interview_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())

        # Should not raise exception
        await emitter.emit_interview_created(
            interview_id=interview_id,
            project_id=project_id,
            title="test.txt",
            source="/path/to/test.txt",
        )

        # Check error was logged
        assert "Failed to emit InterviewCreated event" in caplog.text
        assert interview_id in caplog.text

    async def test_emit_interview_status_changed_success(self, emitter, mock_event_store):
        """Test successful InterviewStatusChanged event emission."""
        interview_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        await emitter.emit_interview_status_changed(
            interview_id=interview_id,
            new_status="completed",
            correlation_id=correlation_id,
        )

        # Verify append_events was called
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args

        # Check stream name
        stream_name = call_args[0][0]
        assert stream_name == f"Interview-{interview_id}"

        # Check event
        events = call_args[0][1]
        event = events[0]
        assert event.event_type == "StatusChanged"
        assert event.aggregate_id == interview_id
        assert event.data["from_status"] == "processing"
        assert event.data["to_status"] == "completed"

    async def test_emit_sentence_created_success(self, emitter, mock_event_store):
        """Test successful SentenceCreated event emission."""
        interview_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        await emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="This is a test sentence.",
            speaker="Speaker A",
            correlation_id=correlation_id,
        )

        # Verify append_events was called
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args

        # Check events
        events = call_args[0][1]
        event = events[0]
        assert event.event_type == "SentenceCreated"
        assert event.aggregate_type == AggregateType.SENTENCE
        assert event.version == 0
        assert event.correlation_id == correlation_id

        # Check data
        assert event.data["interview_id"] == interview_id
        assert event.data["index"] == 0
        assert event.data["text"] == "This is a test sentence."
        assert event.data["speaker"] == "Speaker A"

        # Check sentence_id is deterministic UUID
        sentence_id = event.aggregate_id
        expected_sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))
        assert sentence_id == expected_sentence_id

        # Check stream name uses sentence_id
        stream_name = call_args[0][0]
        assert stream_name == f"Sentence-{sentence_id}"

    async def test_emit_sentence_created_deterministic_uuid(self, emitter, mock_event_store):
        """Test that sentence UUIDs are deterministic."""
        interview_id = str(uuid.uuid4())

        # Emit same sentence twice
        await emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="Test sentence",
        )

        await emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="Test sentence",
        )

        # Both should have same sentence_id
        call1 = mock_event_store.append_events.call_args_list[0]
        call2 = mock_event_store.append_events.call_args_list[1]

        event1 = call1[0][1][0]
        event2 = call2[0][1][0]

        assert event1.aggregate_id == event2.aggregate_id

    async def test_emit_sentence_created_handles_exception(
        self, emitter, mock_event_store, caplog
    ):
        """Test that SentenceCreated emission logs errors but doesn't raise."""
        mock_event_store.append_events.side_effect = Exception("ESDB connection failed")

        interview_id = str(uuid.uuid4())

        # Should not raise exception
        await emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="Test sentence",
        )

        # Check error was logged
        assert "Failed to emit SentenceCreated event" in caplog.text
        assert interview_id in caplog.text

    async def test_emit_analysis_generated_success(self, emitter, mock_event_store):
        """Test successful AnalysisGenerated event emission."""
        interview_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        analysis_data = {
            "model": "gpt-4",
            "model_version": "1.0",
            "function_type": "question",
            "structure_type": "simple",
            "purpose": "inquiry",
            "overall_keywords": ["test", "keyword"],
            "topics": ["testing"],
            "domain_keywords": ["qa"],
            "confidence": 0.95,
        }

        await emitter.emit_analysis_generated(
            interview_id=interview_id,
            sentence_index=0,
            analysis_data=analysis_data,
            correlation_id=correlation_id,
        )

        # Verify append_events was called
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args

        # Check events
        events = call_args[0][1]
        event = events[0]
        assert event.event_type == "AnalysisGenerated"
        assert event.aggregate_type == AggregateType.SENTENCE
        assert event.correlation_id == correlation_id

        # Check data
        assert event.data["model"] == "gpt-4"
        assert event.data["version"] == "1.0"  # Note: field name is 'version', not 'model_version'
        assert event.data["classification"]["function_type"] == "question"
        assert event.data["classification"]["structure_type"] == "simple"
        assert event.data["classification"]["purpose"] == "inquiry"
        assert event.data["keywords"] == ["test", "keyword"]
        assert event.data["topics"] == ["testing"]
        assert event.data["domain_keywords"] == ["qa"]
        assert event.data["confidence"] == 0.95

        # Check sentence_id is deterministic
        sentence_id = event.aggregate_id
        expected_sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))
        assert sentence_id == expected_sentence_id

    async def test_emit_analysis_generated_handles_exception(
        self, emitter, mock_event_store, caplog
    ):
        """Test that AnalysisGenerated emission logs errors but doesn't raise."""
        mock_event_store.append_events.side_effect = Exception("ESDB connection failed")

        interview_id = str(uuid.uuid4())
        analysis_data = {
            "function_type": "question",
            "overall_keywords": ["test"],
        }

        # Should not raise exception
        await emitter.emit_analysis_generated(
            interview_id=interview_id,
            sentence_index=0,
            analysis_data=analysis_data,
        )

        # Check error was logged
        assert "Failed to emit AnalysisGenerated event" in caplog.text
        assert interview_id in caplog.text

    async def test_generate_sentence_uuid_consistency(self, emitter):
        """Test that sentence UUID generation is consistent."""
        interview_id = str(uuid.uuid4())

        # Generate UUID multiple times
        uuid1 = emitter._generate_sentence_uuid(interview_id, 0)
        uuid2 = emitter._generate_sentence_uuid(interview_id, 0)
        uuid3 = emitter._generate_sentence_uuid(interview_id, 1)

        # Same interview + index should give same UUID
        assert uuid1 == uuid2

        # Different index should give different UUID
        assert uuid1 != uuid3

    async def test_multiple_sentences_in_same_interview(self, emitter, mock_event_store):
        """Test emitting multiple sentences for same interview."""
        interview_id = str(uuid.uuid4())

        # Emit 3 sentences
        for i in range(3):
            await emitter.emit_sentence_created(
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i}",
            )

        # Should have 3 calls
        assert mock_event_store.append_events.call_count == 3

        # Each should have different sentence_id
        sentence_ids = set()
        for call_args in mock_event_store.append_events.call_args_list:
            event = call_args[0][1][0]
            sentence_ids.add(event.aggregate_id)

        assert len(sentence_ids) == 3

