"""
Unit tests for projection handlers with mocked Neo4j sessions.

Tests version checking, idempotency, retry logic, and Neo4j query correctness
without requiring actual Neo4j or EventStoreDB.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.base_handler import BaseProjectionHandler
from src.projections.handlers.interview_handlers import (
    InterviewCreatedHandler,
    InterviewMetadataUpdatedHandler,
    InterviewStatusChangedHandler,
)
from src.projections.handlers.sentence_handlers import (
    AnalysisGeneratedHandler,
    SentenceCreatedHandler,
    SentenceEditedHandler,
)


@pytest.mark.asyncio
class TestBaseHandlerVersionChecking:
    """Test base handler version checking and idempotency."""

    async def test_skips_already_applied_event(self):
        """Test that handler skips event if version already applied."""

        # Create a concrete handler for testing
        class TestHandler(BaseProjectionHandler):
            async def apply(self, tx, event):
                pass  # Should not be called

        handler = TestHandler()

        # Mock Neo4j session that returns current version = 5
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"version": 5})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        handler.neo4j_manager = MagicMock()
        handler.neo4j_manager.get_session = MagicMock(return_value=mock_session)

        # Event with version 3 (older than current version 5)
        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=3,
            data={},
        )

        # Handle event
        await handler.handle(event)

        # Should have checked version but not started transaction
        assert mock_session.run.called
        assert not mock_session.begin_transaction.called

    async def test_applies_new_event(self):
        """Test that handler applies event if version is new."""
        apply_called = False

        class TestHandler(BaseProjectionHandler):
            async def apply(self, tx, event):
                nonlocal apply_called
                apply_called = True

        handler = TestHandler()

        # Mock Neo4j session that returns current version = 3
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()
        mock_tx.commit = AsyncMock()
        mock_tx.rollback = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"version": 3})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = MagicMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        handler.neo4j_manager = MagicMock()
        handler.neo4j_manager.get_session = MagicMock(return_value=mock_session)

        # Event with version 4 (newer than current version 3)
        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=4,
            data={},
        )

        # Handle event
        await handler.handle(event)

        # Should have applied event and committed
        assert apply_called
        mock_tx.commit.assert_called_once()

    async def test_handles_new_aggregate(self):
        """Test that handler applies event for new aggregate (no version)."""
        apply_called = False

        class TestHandler(BaseProjectionHandler):
            async def apply(self, tx, event):
                nonlocal apply_called
                apply_called = True

        handler = TestHandler()

        # Mock Neo4j session that returns None (aggregate doesn't exist)
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()
        mock_tx.commit = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = MagicMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        handler.neo4j_manager = MagicMock()
        handler.neo4j_manager.get_session = MagicMock(return_value=mock_session)

        # Event with version 0 (first event)
        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={},
        )

        # Handle event
        await handler.handle(event)

        # Should have applied event
        assert apply_called
        mock_tx.commit.assert_called_once()


@pytest.mark.asyncio
class TestBaseHandlerRetryLogic:
    """Test base handler retry-to-park logic."""

    async def test_retries_on_transient_error(self):
        """Test that handler retries on transient errors."""
        attempt_count = 0

        class TestHandler(BaseProjectionHandler):
            async def apply(self, tx, event):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise Exception("Transient error")
                # Success on 3rd attempt

        handler = TestHandler()

        # Mock successful version check and transaction
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()
        mock_tx.commit = AsyncMock()

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = MagicMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        handler.neo4j_manager = MagicMock()
        handler.neo4j_manager.get_session = MagicMock(return_value=mock_session)

        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={},
        )

        # Handle with retry
        await handler.handle_with_retry(event, lane_id=0)

        # Should have attempted 3 times
        assert attempt_count == 3

    async def test_parks_event_after_max_retries(self):
        """Test that handler parks event after max retries."""

        class TestHandler(BaseProjectionHandler):
            async def apply(self, tx, event):
                raise Exception("Permanent error")

        handler = TestHandler()

        # Mock parked events manager
        handler.parked_events_manager = MagicMock()
        handler.parked_events_manager.park_event = AsyncMock()

        # Mock Neo4j session
        mock_tx = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = MagicMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        handler.neo4j_manager = MagicMock()
        handler.neo4j_manager.get_session = MagicMock(return_value=mock_session)

        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={},
        )

        # Handle with retry - should raise after parking
        with pytest.raises(Exception):
            await handler.handle_with_retry(event, lane_id=0)

        # Should have parked the event
        handler.parked_events_manager.park_event.assert_called_once()


@pytest.mark.asyncio
class TestInterviewHandlers:
    """Test Interview event handlers."""

    async def test_interview_created_handler(self):
        """Test InterviewCreated handler creates correct nodes."""
        handler = InterviewCreatedHandler()

        # Mock transaction
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        event = EventEnvelope(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={
                "project_id": "test-project",
                "title": "Test Interview",
                "source": "test.txt",
                "language": "en",
                "status": "created",
            },
        )

        # Apply event
        await handler.apply(mock_tx, event)

        # Should have run Cypher query
        mock_tx.run.assert_called_once()
        call_args = mock_tx.run.call_args

        # Verify query creates Project and Interview nodes
        query = call_args[0][0]
        assert "MERGE (p:Project" in query
        assert "CREATE (i:Interview" in query
        assert "CONTAINS_INTERVIEW" in query

        # Verify parameters
        params = call_args[1]
        assert params["project_id"] == "test-project"
        assert params["title"] == "Test Interview"
        assert params["interview_id"] == event.aggregate_id

    async def test_interview_metadata_updated_handler(self):
        """Test InterviewMetadataUpdated handler updates correct fields."""
        handler = InterviewMetadataUpdatedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "title": "Updated Title",
                "language": "es",
            },
        )

        await handler.apply(mock_tx, event)

        mock_tx.run.assert_called_once()
        call_args = mock_tx.run.call_args

        query = call_args[0][0]
        assert "SET" in query
        assert "i.title" in query
        assert "i.language" in query

    async def test_interview_status_changed_handler(self):
        """Test StatusChanged handler updates status."""
        handler = InterviewStatusChangedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        event = EventEnvelope(
            event_type="StatusChanged",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "new_status": "processing",
            },
        )

        await handler.apply(mock_tx, event)

        mock_tx.run.assert_called_once()
        call_args = mock_tx.run.call_args

        query = call_args[0][0]
        params = call_args[1]

        assert "SET" in query
        assert "i.status" in query
        assert params["new_status"] == "processing"


@pytest.mark.asyncio
class TestSentenceHandlers:
    """Test Sentence event handlers."""

    async def test_sentence_created_handler(self):
        """Test SentenceCreated handler creates sentence node."""
        handler = SentenceCreatedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=sentence_id,
            version=0,
            data={
                "interview_id": interview_id,
                "index": 0,
                "text": "This is a test sentence.",
                "speaker": "Speaker A",
            },
        )

        await handler.apply(mock_tx, event)

        mock_tx.run.assert_called_once()
        call_args = mock_tx.run.call_args

        query = call_args[0][0]
        params = call_args[1]

        assert "MATCH (i:Interview" in query
        assert "CREATE (s:Sentence" in query
        assert "HAS_SENTENCE" in query
        assert params["interview_id"] == interview_id
        assert params["sentence_id"] == sentence_id
        assert params["text"] == "This is a test sentence."

    async def test_sentence_edited_handler(self):
        """Test SentenceEdited handler updates text and sets edited flag."""
        handler = SentenceEditedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        event = EventEnvelope(
            event_type="SentenceEdited",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "new_text": "Edited text.",
                "editor_type": "human",
            },
        )

        await handler.apply(mock_tx, event)

        mock_tx.run.assert_called_once()
        call_args = mock_tx.run.call_args

        query = call_args[0][0]
        params = call_args[1]

        assert "SET" in query
        assert "s.text" in query
        assert "s.is_edited = true" in query
        assert params["new_text"] == "Edited text."

    async def test_analysis_generated_handler_creates_nodes(self):
        """Test AnalysisGenerated handler creates analysis and dimension nodes."""
        handler = AnalysisGeneratedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()

        event = EventEnvelope(
            event_type="AnalysisGenerated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "model": "gpt-4",
                "model_version": "1.0",
                "classification": {
                    "function_type": "question",
                    "structure_type": "simple",
                    "purpose": "inquiry",
                },
                "keywords": ["test", "keyword"],
                "topics": ["testing"],
                "domain_keywords": ["qa"],
                "confidence": 0.95,
            },
        )

        await handler.apply(mock_tx, event)

        # Should have made multiple calls: analysis + dimensions
        assert mock_tx.run.call_count >= 1

        # Check first call creates Analysis node
        first_call = mock_tx.run.call_args_list[0]
        query = first_call[0][0]
        assert "CREATE (a:Analysis" in query
        assert "HAS_ANALYSIS" in query
