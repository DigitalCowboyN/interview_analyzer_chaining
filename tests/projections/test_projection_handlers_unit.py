"""
Unit tests for projection handlers with mocked Neo4j sessions.

Tests version checking, idempotency, retry logic, and Neo4j query correctness
without requiring actual Neo4j or EventStoreDB.
"""

import json
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
from src.projections.handlers.speaker_handlers import (
    ReferentNotReadyError,
    SpeakerCreatedHandler,
    _raise_if_no_writes,
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

        # Mock Neo4jConnectionManager.get_session at module level
        from unittest.mock import patch

        with patch(
            "src.projections.handlers.base_handler.Neo4jConnectionManager.get_session", return_value=mock_session
        ):
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
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock Neo4jConnectionManager.get_session at module level
        from unittest.mock import patch

        with patch(
            "src.projections.handlers.base_handler.Neo4jConnectionManager.get_session", return_value=mock_session
        ):
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
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock Neo4jConnectionManager.get_session at module level
        from unittest.mock import patch

        with patch(
            "src.projections.handlers.base_handler.Neo4jConnectionManager.get_session", return_value=mock_session
        ):
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
        from unittest.mock import patch

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
        mock_tx.rollback = AsyncMock()

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={},
        )

        # Patch Neo4jConnectionManager.get_session at module level
        with patch(
            "src.projections.handlers.base_handler.Neo4jConnectionManager.get_session",
            return_value=mock_session,
        ):
            # Handle with retry
            await handler.handle_with_retry(event, lane_id=0)

        # Should have attempted 3 times
        assert attempt_count == 3

    async def test_parks_event_after_max_retries(self):
        """Test that handler parks event after max retries."""
        from unittest.mock import patch

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
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        event = EventEnvelope(
            event_type="TestEvent",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={},
        )

        # Patch Neo4jConnectionManager.get_session at module level
        with patch(
            "src.projections.handlers.base_handler.Neo4jConnectionManager.get_session",
            return_value=mock_session,
        ):
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

        # Mock transaction with proper result consumption
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5  # Properties updated
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

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

        # Verify query creates Project and Interview nodes (using MERGE for dual-write safety)
        query = call_args[0][0]
        assert "MERGE (p:Project" in query
        assert "MERGE (i:Interview" in query  # Changed to MERGE for deduplication during dual-write
        assert "CONTAINS_INTERVIEW" in query

        # Verify parameters
        params = call_args[1]
        assert params["project_id"] == "test-project"
        assert params["title"] == "Test Interview"
        assert params["interview_id"] == event.aggregate_id

        # Verify result was consumed
        mock_result.consume.assert_called_once()

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

    async def test_interview_metadata_updated_diff_onto_empty_node_sets_full_dict(self):
        """metadata_diff onto a node with no existing metadata_json sets the full dict."""
        handler = InterviewMetadataUpdatedHandler()

        mock_read_result = AsyncMock()
        mock_read_result.single = AsyncMock(return_value=None)  # no metadata_json yet

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(side_effect=[mock_read_result, AsyncMock()])

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"metadata_diff": {"interviewer": "Alice", "location": "Remote"}},
        )

        await handler.apply(mock_tx, event)

        assert mock_tx.run.call_count == 2
        write_call = mock_tx.run.call_args_list[1]
        write_query = write_call[0][0]
        write_params = write_call[1]
        assert "i.metadata_json" in write_query
        assert json.loads(write_params["metadata_json"]) == {
            "interviewer": "Alice",
            "location": "Remote",
        }

    async def test_interview_metadata_updated_diff_merges_over_existing_keys(self):
        """metadata_diff overwrites matching keys, keeps untouched ones."""
        handler = InterviewMetadataUpdatedHandler()

        existing = json.dumps({"interviewer": "Bob", "location": "Onsite"}, sort_keys=True)
        mock_read_result = AsyncMock()
        mock_read_result.single = AsyncMock(return_value={"metadata_json": existing})

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(side_effect=[mock_read_result, AsyncMock()])

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"metadata_diff": {"interviewer": "Alice"}},
        )

        await handler.apply(mock_tx, event)

        write_params = mock_tx.run.call_args_list[1][1]
        assert json.loads(write_params["metadata_json"]) == {
            "interviewer": "Alice",
            "location": "Onsite",
        }

    async def test_interview_metadata_updated_none_value_deletes_key(self):
        """A metadata_diff key with value None deletes that key from the merged dict."""
        handler = InterviewMetadataUpdatedHandler()

        existing = json.dumps({"interviewer": "Alice", "location": "Onsite"}, sort_keys=True)
        mock_read_result = AsyncMock()
        mock_read_result.single = AsyncMock(return_value={"metadata_json": existing})

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(side_effect=[mock_read_result, AsyncMock()])

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"metadata_diff": {"location": None}},
        )

        await handler.apply(mock_tx, event)

        write_params = mock_tx.run.call_args_list[1][1]
        assert json.loads(write_params["metadata_json"]) == {"interviewer": "Alice"}

    async def test_interview_metadata_updated_query_text_pins_metadata_json(self):
        """Both the read-merge query and the write SET clause reference metadata_json."""
        handler = InterviewMetadataUpdatedHandler()

        mock_read_result = AsyncMock()
        mock_read_result.single = AsyncMock(return_value=None)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(side_effect=[mock_read_result, AsyncMock()])

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"metadata_diff": {"interviewer": "Alice"}},
        )

        await handler.apply(mock_tx, event)

        read_query = mock_tx.run.call_args_list[0][0][0]
        write_query = mock_tx.run.call_args_list[1][0][0]
        assert "metadata_json" in read_query
        assert "i.metadata_json" in write_query

    async def test_interview_metadata_updated_none_metadata_diff_skips_write(self):
        """Payload with metadata_diff=None skips metadata write (guard consistency)."""
        handler = InterviewMetadataUpdatedHandler()

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=AsyncMock())

        event = EventEnvelope(
            event_type="InterviewMetadataUpdated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={"metadata_diff": None},
        )

        await handler.apply(mock_tx, event)

        # Should have no calls since metadata_diff is None (early return due to no updates)
        mock_tx.run.assert_not_called()

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

        # Mock transaction with proper result consumption
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5  # Properties updated
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

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
        assert "MERGE (s:Fragment" in query  # MERGE for deduplication during dual-write
        assert "HAS_SENTENCE" in query
        assert params["interview_id"] == interview_id
        assert params["sentence_id"] == sentence_id
        assert params["text"] == "This is a test sentence."

        # Verify result was consumed
        mock_result.consume.assert_called_once()

    async def test_sentence_created_uses_fragment_anchor_only(self):
        """Test SentenceCreated handler anchors on :Fragment alone (shim dropped)."""
        handler = SentenceCreatedHandler()

        # Mock transaction with proper result consumption
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5  # Properties updated
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

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

        call_args = mock_tx.run.call_args
        query = call_args[0][0]

        assert "MERGE (s:Fragment {sentence_id: $sentence_id})" in query  # new anchor
        assert "SET s:Fragment" not in query  # shim dual-label SET is gone

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

        # Mock result for "find edited relationships" query (first query)
        mock_edited_result = AsyncMock()
        mock_edited_result.single = AsyncMock(return_value=None)  # No existing edited rels

        # Mock result for "delete old analysis" query (second query)
        mock_delete_result = AsyncMock()
        mock_delete_summary = MagicMock()
        mock_delete_summary.counters.nodes_deleted = 0  # No nodes deleted (first analysis)
        mock_delete_result.consume = AsyncMock(return_value=mock_delete_summary)

        # Mock results for subsequent queries (create analysis, link dimensions)
        mock_generic_result = AsyncMock()

        # Configure mock_tx.run to return different results for each call
        # Total calls: find_edited(1) + delete(1) + create_analysis(1) + classification(3) + keywords(2) + topics(1) + domain(1) = 10
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(side_effect=[
            mock_edited_result,      # 1: find edited rels
            mock_delete_result,      # 2: delete old analysis
            mock_generic_result,     # 3: create analysis
            mock_generic_result,     # 4-6: link classification (function, structure, purpose)
            mock_generic_result,
            mock_generic_result,
            mock_generic_result,     # 7-8: link keywords (2 values)
            mock_generic_result,
            mock_generic_result,     # 9: link topics (1 value)
            mock_generic_result,     # 10: link domain keywords (1 value)
        ])

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

        # Should have made multiple calls: find edited, delete, analysis + dimensions
        assert mock_tx.run.call_count >= 3

        # Check third call creates Analysis node (after find edited & delete)
        third_call = mock_tx.run.call_args_list[2]
        query = third_call[0][0]
        assert "CREATE (a:Analysis" in query
        assert "HAS_ANALYSIS" in query


@pytest.mark.asyncio
class TestProjectionHandlerDeduplication:
    """Test deduplication behavior during dual-write phase."""

    async def test_sentence_created_skips_duplicate_from_direct_write(self):
        """Test that projection service skips sentence when direct write already created it."""
        handler = SentenceCreatedHandler()

        # Mock transaction that returns 0 properties_set (duplicate skipped)
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 0  # No properties updated = duplicate skipped
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=sentence_id,
            version=1,  # Projection service version
            data={
                "interview_id": interview_id,
                "index": 0,
                "text": "Test sentence.",
                "speaker": "Speaker A",
            },
        )

        # Apply event
        await handler.apply(mock_tx, event)

        # Should have run query
        mock_tx.run.assert_called_once()

        # Should have consumed result to check counters
        mock_result.consume.assert_called_once()

    async def test_sentence_created_updates_when_version_is_newer(self):
        """Test that projection service updates sentence when event version is newer."""
        handler = SentenceCreatedHandler()

        # Mock transaction that returns >0 properties_set (update succeeded)
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5  # Properties updated = version was newer
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=sentence_id,
            version=1,
            data={
                "interview_id": interview_id,
                "index": 0,
                "text": "Test sentence.",
            },
        )

        # Apply event
        await handler.apply(mock_tx, event)

        # Should have run query and consumed result
        mock_tx.run.assert_called_once()
        mock_result.consume.assert_called_once()

    async def test_interview_created_skips_duplicate_from_direct_write(self):
        """Test that projection service skips interview when direct write already created it."""
        handler = InterviewCreatedHandler()

        # Mock transaction that returns 0 properties_set (duplicate skipped)
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 0  # No properties updated = duplicate skipped
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        interview_id = str(uuid.uuid4())

        event = EventEnvelope(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=interview_id,
            version=1,  # Projection service version
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

        # Should have run query and consumed result
        mock_tx.run.assert_called_once()
        mock_result.consume.assert_called_once()

    async def test_interview_created_updates_when_version_is_newer(self):
        """Test that projection service updates interview when event version is newer."""
        handler = InterviewCreatedHandler()

        # Mock transaction that returns >0 properties_set (update succeeded)
        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5  # Properties updated = version was newer
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        interview_id = str(uuid.uuid4())

        event = EventEnvelope(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=interview_id,
            version=1,
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

        # Should have run query and consumed result
        mock_tx.run.assert_called_once()
        mock_result.consume.assert_called_once()


class TestReferentNotReadyErrorType:
    """Pin the type contract: _raise_if_no_writes raises ReferentNotReadyError,
    which IS-A ValueError so existing retry/park handling still engages."""

    def test_raise_if_no_writes_raises_referent_not_ready_error(self):
        summary = MagicMock()
        summary.counters.nodes_created = 0
        summary.counters.properties_set = 0
        summary.counters.relationships_created = 0

        with pytest.raises(ReferentNotReadyError):
            _raise_if_no_writes(summary, "SomeEvent", "agg-123")

    def test_referent_not_ready_error_is_a_value_error(self):
        assert issubclass(ReferentNotReadyError, ValueError)

    def test_raise_if_no_writes_does_not_raise_when_writes_occurred(self):
        summary = MagicMock()
        summary.counters.nodes_created = 0
        summary.counters.properties_set = 1
        summary.counters.relationships_created = 0

        # Should not raise
        _raise_if_no_writes(summary, "SomeEvent", "agg-123")


@pytest.mark.asyncio
class TestSpeakerCreatedReferentGuard:
    """SpeakerCreated depends on a not-yet-projected Interview referent.
    The MERGE/SET is unconditional, so _raise_if_no_writes (write-counter
    based) correctly distinguishes 'Interview missing' (0 writes) from an
    idempotent replay (SET always fires -> properties_set > 0)."""

    async def test_speaker_created_raises_when_interview_missing(self):
        handler = SpeakerCreatedHandler()

        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 0
        mock_summary.counters.properties_set = 0
        mock_summary.counters.relationships_created = 0
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SpeakerCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={
                "speaker_id": str(uuid.uuid4()),
                "handle": "SPEAKER_A",
                "display_name": "Speaker A",
                "provisional": True,
            },
        )

        with pytest.raises(ReferentNotReadyError):
            await handler.apply(mock_tx, event)

    async def test_speaker_created_no_raise_when_interview_present(self):
        handler = SpeakerCreatedHandler()

        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 1
        mock_summary.counters.properties_set = 6
        mock_summary.counters.relationships_created = 1
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SpeakerCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={
                "speaker_id": str(uuid.uuid4()),
                "handle": "SPEAKER_A",
                "display_name": "Speaker A",
                "provisional": True,
            },
        )

        # Should not raise
        await handler.apply(mock_tx, event)

        mock_tx.run.assert_called_once()
        mock_result.consume.assert_called_once()

    async def test_speaker_created_no_raise_on_idempotent_replay(self):
        """Idempotent replay: Interview present, Speaker already MERGEd
        (nodes_created == 0) but the unconditional SET still fires
        (properties_set > 0) -> must not raise."""
        handler = SpeakerCreatedHandler()

        mock_result = AsyncMock()
        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 0
        mock_summary.counters.properties_set = 6
        mock_summary.counters.relationships_created = 0
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SpeakerCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "speaker_id": str(uuid.uuid4()),
                "handle": "SPEAKER_A",
                "display_name": "Speaker A",
                "provisional": True,
            },
        )

        # Should not raise
        await handler.apply(mock_tx, event)


@pytest.mark.asyncio
class TestSentenceCreatedReferentGuard:
    """SentenceCreated's SET is version-gated, so write counters alone
    cannot distinguish 'Interview missing' from 'idempotent same-version
    replay' (both yield all-zero counters). The handler must instead check
    row presence from a RETURN that is unconditional on the version gate."""

    async def test_sentence_created_raises_when_interview_missing(self):
        handler = SentenceCreatedHandler()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)  # no row -> MATCH found nothing
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 0
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={
                "interview_id": str(uuid.uuid4()),
                "index": 0,
                "text": "Orphaned sentence.",
                "speaker": "Speaker A",
            },
        )

        with pytest.raises(ReferentNotReadyError):
            await handler.apply(mock_tx, event)

    async def test_sentence_created_no_raise_when_interview_present(self):
        handler = SentenceCreatedHandler()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"interview_matched": "some-interview-id"})
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={
                "interview_id": str(uuid.uuid4()),
                "index": 0,
                "text": "A sentence.",
                "speaker": "Speaker A",
            },
        )

        # Should not raise
        await handler.apply(mock_tx, event)

        query = mock_tx.run.call_args[0][0]
        assert "RETURN" in query
        assert "interview_matched" in query

    async def test_sentence_created_no_raise_on_idempotent_same_version_replay(self):
        """THE critical trap: Interview present (row returned) but the
        version-gated SET does not fire because the same version was
        already applied -> all write counters are 0. Must NOT raise."""
        handler = SentenceCreatedHandler()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"interview_matched": "some-interview-id"})
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 0
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=1,
            data={
                "interview_id": str(uuid.uuid4()),
                "index": 0,
                "text": "Same version replay.",
                "speaker": "Speaker A",
            },
        )

        # Should not raise (this is the false-positive trap the guard must avoid)
        await handler.apply(mock_tx, event)

    async def test_sentence_created_no_raise_on_newer_version_reapply(self):
        handler = SentenceCreatedHandler()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"interview_matched": "some-interview-id"})
        mock_summary = MagicMock()
        mock_summary.counters.properties_set = 5
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock(return_value=mock_result)

        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=2,
            data={
                "interview_id": str(uuid.uuid4()),
                "index": 0,
                "text": "Newer version reapply.",
                "speaker": "Speaker A",
            },
        )

        # Should not raise
        await handler.apply(mock_tx, event)
