"""
Integration tests for dual-write pipeline (M2.2).

Tests that the pipeline correctly emits events to EventStoreDB
alongside Neo4j writes during file processing.

These tests validate:
1. InterviewCreated event emission on file upload
2. SentenceCreated event emission per sentence
3. AnalysisGenerated event emission per analysis
4. Correlation ID propagation across events
5. Deterministic UUID generation for sentences
6. Non-blocking error handling (Neo4j succeeds even if events fail)
"""

import pytest
import uuid
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.io.neo4j_map_storage import Neo4jMapStorage
from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.pipeline_event_emitter import PipelineEventEmitter
from src.events.store import EventStoreClient


@pytest.mark.asyncio
class TestDualWritePipelineEventEmission:
    """Test event emission during pipeline processing (dual-write phase)."""

    @pytest.fixture
    def mock_event_store(self):
        """Create a mock EventStoreClient for testing."""
        mock_client = MagicMock(spec=EventStoreClient)
        mock_client.append_events = AsyncMock()
        return mock_client

    @pytest.fixture
    def event_emitter(self, mock_event_store):
        """Create PipelineEventEmitter with mocked EventStoreClient."""
        return PipelineEventEmitter(mock_event_store)

    @pytest.fixture
    def correlation_id(self):
        """Generate a correlation ID for test."""
        return str(uuid.uuid4())

    @pytest.fixture
    def project_id(self):
        """Test project ID."""
        return "test-project-123"

    @pytest.fixture
    def interview_id(self):
        """Test interview ID."""
        return str(uuid.uuid4())

    async def test_interview_created_event_emission(
        self, event_emitter, mock_event_store, correlation_id, project_id, interview_id
    ):
        """Test that InterviewCreated event is emitted with correct data."""
        # Arrange
        title = "test_file.txt"
        source = "/path/to/test_file.txt"
        language = "en"

        # Act
        await event_emitter.emit_interview_created(
            interview_id=interview_id,
            project_id=project_id,
            title=title,
            source=source,
            language=language,
            correlation_id=correlation_id,
        )

        # Assert
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args
        stream_name = call_args[0][0]
        events = call_args[0][1]

        assert stream_name == f"Interview-{interview_id}"
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "InterviewCreated"
        assert event.aggregate_type == "Interview"
        assert event.aggregate_id == interview_id
        assert event.project_id == project_id
        assert event.correlation_id == correlation_id
        assert event.version == 0
        assert event.data["title"] == title
        assert event.data["source"] == source
        assert event.data["language"] == language
        assert event.actor.actor_type == "system"
        assert event.actor.user_id == "pipeline"

    async def test_sentence_created_event_emission(
        self, event_emitter, mock_event_store, correlation_id, interview_id
    ):
        """Test that SentenceCreated event is emitted with correct data."""
        # Arrange
        index = 0
        text = "This is a test sentence."
        speaker = "Interviewer"
        start_ms = 1000
        end_ms = 2000

        # Act
        await event_emitter.emit_sentence_created(
            interview_id=interview_id,
            index=index,
            text=text,
            speaker=speaker,
            start_ms=start_ms,
            end_ms=end_ms,
            correlation_id=correlation_id,
        )

        # Assert
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args
        stream_name = call_args[0][0]
        events = call_args[0][1]

        # Verify deterministic UUID generation
        expected_sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))
        assert stream_name == f"Sentence-{expected_sentence_id}"
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "SentenceCreated"
        assert event.aggregate_type == "Sentence"
        assert event.aggregate_id == expected_sentence_id
        assert event.correlation_id == correlation_id
        assert event.version == 0
        assert event.data["interview_id"] == interview_id
        assert event.data["index"] == index
        assert event.data["text"] == text
        assert event.data["speaker"] == speaker
        assert event.data["start_ms"] == start_ms
        assert event.data["end_ms"] == end_ms

    async def test_analysis_generated_event_emission(
        self, event_emitter, mock_event_store, correlation_id, interview_id
    ):
        """Test that AnalysisGenerated event is emitted with correct data."""
        # Arrange
        sentence_index = 0
        analysis_data = {
            "sentence_id": 1,
            "classification": "question",
            "overall_keywords": ["test", "keyword"],
            "topics": ["testing"],
            "domain_keywords": ["python"],
            "confidence": 0.95,
        }

        # Act
        await event_emitter.emit_analysis_generated(
            interview_id=interview_id,
            sentence_index=sentence_index,
            analysis_data=analysis_data,
            correlation_id=correlation_id,
        )

        # Assert
        mock_event_store.append_events.assert_called_once()
        call_args = mock_event_store.append_events.call_args
        stream_name = call_args[0][0]
        events = call_args[0][1]

        # Verify deterministic UUID for sentence
        expected_sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        assert stream_name == f"Sentence-{expected_sentence_id}"
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "AnalysisGenerated"
        assert event.aggregate_type == "Sentence"
        assert event.aggregate_id == expected_sentence_id
        assert event.correlation_id == correlation_id
        assert event.version == 1  # Version 1 (after SentenceCreated at v0)
        # Classification is a dict with function_type, structure_type, purpose
        assert "classification" in event.data
        assert event.data["keywords"] == ["test", "keyword"]
        assert event.data["topics"] == ["testing"]
        assert event.data["domain_keywords"] == ["python"]
        assert event.data["confidence"] == 0.95

    async def test_correlation_id_propagation(
        self, event_emitter, mock_event_store, correlation_id, project_id, interview_id
    ):
        """Test that correlation_id is propagated across all events."""
        # Act - emit multiple events with same correlation_id
        await event_emitter.emit_interview_created(
            interview_id=interview_id,
            project_id=project_id,
            title="test.txt",
            source="/test.txt",
            language="en",
            correlation_id=correlation_id,
        )

        await event_emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="Test sentence.",
            correlation_id=correlation_id,
        )

        await event_emitter.emit_analysis_generated(
            interview_id=interview_id,
            sentence_index=0,
            analysis_data={"sentence_id": 1, "classification": "statement"},
            correlation_id=correlation_id,
        )

        # Assert - all events have same correlation_id
        assert mock_event_store.append_events.call_count == 3
        for call in mock_event_store.append_events.call_args_list:
            events = call[0][1]
            for event in events:
                assert event.correlation_id == correlation_id

    async def test_deterministic_sentence_uuid_generation(
        self, event_emitter, mock_event_store, correlation_id, interview_id
    ):
        """Test that sentence UUIDs are generated deterministically."""
        # Act - emit same sentence twice
        for _ in range(2):
            await event_emitter.emit_sentence_created(
                interview_id=interview_id,
                index=5,
                text="Same sentence",
                correlation_id=correlation_id,
            )

        # Assert - both calls used same stream (deterministic UUID)
        assert mock_event_store.append_events.call_count == 2
        call1_stream = mock_event_store.append_events.call_args_list[0][0][0]
        call2_stream = mock_event_store.append_events.call_args_list[1][0][0]
        assert call1_stream == call2_stream

        # Verify expected UUID
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:5"))
        assert call1_stream == f"Sentence-{expected_uuid}"


@pytest.mark.asyncio
class TestNeo4jMapStorageEventIntegration:
    """Test Neo4jMapStorage with event emission enabled."""

    @pytest.fixture
    def mock_neo4j_session(self):
        """Mock Neo4j session."""
        mock_session = MagicMock()
        mock_session.run = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        return mock_session

    @pytest.fixture
    def mock_event_emitter(self):
        """Mock PipelineEventEmitter."""
        mock = MagicMock(spec=PipelineEventEmitter)
        mock.emit_sentence_created = AsyncMock()
        return mock

    @pytest.fixture
    def project_id(self):
        return "test-project"

    @pytest.fixture
    def interview_id(self):
        return str(uuid.uuid4())

    @pytest.fixture
    def correlation_id(self):
        return str(uuid.uuid4())

    async def test_map_storage_emits_sentence_created_event(
        self,
        mock_neo4j_session,
        mock_event_emitter,
        project_id,
        interview_id,
        correlation_id,
    ):
        """Test that Neo4jMapStorage emits SentenceCreated after successful write."""
        # Arrange
        with patch(
            "src.io.neo4j_map_storage.Neo4jConnectionManager.get_session",
            return_value=mock_neo4j_session,
        ):
            storage = Neo4jMapStorage(
                project_id=project_id,
                interview_id=interview_id,
                event_emitter=mock_event_emitter,
                correlation_id=correlation_id,
            )

            entry = {
                "sentence_id": 1,
                "sequence_order": 0,
                "sentence": "Test sentence.",
                "speaker": "Interviewer",
                "start_time": 1000,
                "end_time": 2000,
            }

            # Act
            await storage.write_entry(entry)

            # Assert - Neo4j write happened
            mock_neo4j_session.run.assert_called()

            # Assert - Event emitted with correct data
            mock_event_emitter.emit_sentence_created.assert_called_once()
            call_kwargs = mock_event_emitter.emit_sentence_created.call_args[1]
            assert call_kwargs["interview_id"] == interview_id
            assert call_kwargs["index"] == 0
            assert call_kwargs["text"] == "Test sentence."
            assert call_kwargs["speaker"] == "Interviewer"
            assert call_kwargs["start_ms"] == 1000
            assert call_kwargs["end_ms"] == 2000
            assert call_kwargs["correlation_id"] == correlation_id

    async def test_map_storage_neo4j_succeeds_even_if_event_fails(
        self,
        mock_neo4j_session,
        mock_event_emitter,
        project_id,
        interview_id,
        correlation_id,
    ):
        """Test that Neo4j write succeeds even if event emission fails."""
        # Arrange - event emission will fail
        mock_event_emitter.emit_sentence_created.side_effect = Exception("EventStoreDB down")

        with patch(
            "src.io.neo4j_map_storage.Neo4jConnectionManager.get_session",
            return_value=mock_neo4j_session,
        ):
            storage = Neo4jMapStorage(
                project_id=project_id,
                interview_id=interview_id,
                event_emitter=mock_event_emitter,
                correlation_id=correlation_id,
            )

            entry = {
                "sentence_id": 1,
                "sequence_order": 0,
                "sentence": "Test sentence.",
            }

            # Act - should not raise despite event emission failure
            await storage.write_entry(entry)

            # Assert - Neo4j write still happened
            mock_neo4j_session.run.assert_called()

            # Assert - Event emission was attempted
            mock_event_emitter.emit_sentence_created.assert_called_once()


@pytest.mark.asyncio
class TestNeo4jAnalysisWriterEventIntegration:
    """Test Neo4jAnalysisWriter with event emission enabled."""

    @pytest.fixture
    def mock_neo4j_session(self):
        """Mock Neo4j session."""
        mock_session = MagicMock()
        mock_session.run = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        return mock_session

    @pytest.fixture
    def mock_event_emitter(self):
        """Mock PipelineEventEmitter."""
        mock = MagicMock(spec=PipelineEventEmitter)
        mock.emit_analysis_generated = AsyncMock()
        return mock

    @pytest.fixture
    def project_id(self):
        return "test-project"

    @pytest.fixture
    def interview_id(self):
        return str(uuid.uuid4())

    @pytest.fixture
    def correlation_id(self):
        return str(uuid.uuid4())

    async def test_analysis_writer_emits_analysis_generated_event(
        self,
        mock_neo4j_session,
        mock_event_emitter,
        project_id,
        interview_id,
        correlation_id,
    ):
        """Test that Neo4jAnalysisWriter emits AnalysisGenerated after successful write."""
        # Arrange
        with patch(
            "src.io.neo4j_analysis_writer.Neo4jConnectionManager.get_session",
            return_value=mock_neo4j_session,
        ):
            writer = Neo4jAnalysisWriter(
                project_id=project_id,
                interview_id=interview_id,
                event_emitter=mock_event_emitter,
                correlation_id=correlation_id,
            )

            result = {
                "sentence_id": 1,
                "classification": "question",
                "overall_keywords": ["test"],
                "topics": ["testing"],
                "domain_keywords": [],
                "confidence": 0.9,
            }

            # Act
            await writer.write_result(result)

            # Assert - Neo4j write happened
            mock_neo4j_session.run.assert_called()

            # Assert - Event emitted with correct data
            mock_event_emitter.emit_analysis_generated.assert_called_once()
            call_kwargs = mock_event_emitter.emit_analysis_generated.call_args[1]
            assert call_kwargs["interview_id"] == interview_id
            assert call_kwargs["sentence_index"] == 1
            assert call_kwargs["analysis_data"] == result
            assert call_kwargs["correlation_id"] == correlation_id

    async def test_analysis_writer_skips_event_for_error_results(
        self,
        mock_neo4j_session,
        mock_event_emitter,
        project_id,
        interview_id,
        correlation_id,
    ):
        """Test that Neo4jAnalysisWriter does NOT emit events for error results."""
        # Arrange
        with patch(
            "src.io.neo4j_analysis_writer.Neo4jConnectionManager.get_session",
            return_value=mock_neo4j_session,
        ):
            writer = Neo4jAnalysisWriter(
                project_id=project_id,
                interview_id=interview_id,
                event_emitter=mock_event_emitter,
                correlation_id=correlation_id,
            )

            error_result = {
                "sentence_id": 1,
                "error": True,
                "error_message": "Analysis failed",
            }

            # Act
            await writer.write_result(error_result)

            # Assert - Neo4j write happened
            mock_neo4j_session.run.assert_called()

            # Assert - NO event emitted (error results don't generate events)
            mock_event_emitter.emit_analysis_generated.assert_not_called()

    async def test_analysis_writer_neo4j_succeeds_even_if_event_fails(
        self,
        mock_neo4j_session,
        mock_event_emitter,
        project_id,
        interview_id,
        correlation_id,
    ):
        """Test that Neo4j write succeeds even if event emission fails."""
        # Arrange - event emission will fail
        mock_event_emitter.emit_analysis_generated.side_effect = Exception("EventStoreDB down")

        with patch(
            "src.io.neo4j_analysis_writer.Neo4jConnectionManager.get_session",
            return_value=mock_neo4j_session,
        ):
            writer = Neo4jAnalysisWriter(
                project_id=project_id,
                interview_id=interview_id,
                event_emitter=mock_event_emitter,
                correlation_id=correlation_id,
            )

            result = {
                "sentence_id": 1,
                "classification": "statement",
            }

            # Act - should not raise despite event emission failure
            await writer.write_result(result)

            # Assert - Neo4j write still happened
            mock_neo4j_session.run.assert_called()

            # Assert - Event emission was attempted
            mock_event_emitter.emit_analysis_generated.assert_called_once()
