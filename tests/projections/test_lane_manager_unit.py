"""
Unit tests for LaneManager with mocked dependencies.

Tests partitioning logic, queue management, and concurrent event processing
without requiring actual EventStoreDB or Neo4j.
"""

import asyncio
import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.lane_manager import Lane, LaneManager


@pytest.mark.asyncio
class TestLanePartitioning:
    """Test lane partitioning logic."""

    def test_consistent_hashing(self):
        """Test that same interview_id always maps to same lane."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        interview_id = str(uuid.uuid4())

        # Get lane 10 times for same interview_id
        lanes = [manager.get_lane_for_interview(interview_id) for _ in range(10)]

        # All should be the same lane
        assert all(lane == lanes[0] for lane in lanes)

    def test_different_interviews_can_map_to_different_lanes(self):
        """Test that different interview_ids can map to different lanes."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        # Generate 100 different interview IDs
        interview_ids = [str(uuid.uuid4()) for _ in range(100)]

        # Get lanes for each
        lane_ids = [manager.get_lane_for_interview(iid).lane_id for iid in interview_ids]

        # Should have some distribution (not all in same lane)
        unique_lanes = set(lane_ids)
        assert len(unique_lanes) > 1, "All interviews mapped to same lane - hash distribution broken"

    def test_lane_distribution(self):
        """Test that interviews are reasonably distributed across lanes."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        # Generate 1000 interview IDs
        interview_ids = [str(uuid.uuid4()) for _ in range(1000)]

        # Count how many go to each lane
        lane_counts = {}
        for iid in interview_ids:
            lane_id = manager.get_lane_for_interview(iid).lane_id
            lane_counts[lane_id] = lane_counts.get(lane_id, 0) + 1

        # All lanes should be used
        assert len(lane_counts) == 12, f"Not all lanes used: {lane_counts}"

        # Check distribution is reasonable (within 50% of average)
        avg = 1000 / 12  # ~83
        for lane_id, count in lane_counts.items():
            assert count > avg * 0.5, f"Lane {lane_id} underutilized: {count} (expected ~{avg})"
            assert count < avg * 1.5, f"Lane {lane_id} overutilized: {count} (expected ~{avg})"

    def test_hash_matches_expected_algorithm(self):
        """Test that hash algorithm matches what we expect."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        interview_id = "test-interview-123"

        # Calculate expected lane using same algorithm
        hash_value = int(hashlib.md5(interview_id.encode()).hexdigest(), 16)
        expected_lane_id = hash_value % 12

        # Get actual lane
        actual_lane = manager.get_lane_for_interview(interview_id)

        assert actual_lane.lane_id == expected_lane_id


@pytest.mark.asyncio
class TestLaneManager:
    """Test LaneManager orchestration."""

    async def test_start_all_lanes(self):
        """Test that starting manager starts all lanes."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=3)

        await manager.start()

        # All lanes should be running
        for lane in manager.lanes:
            assert lane.is_running is True

        await manager.stop()

    async def test_stop_all_lanes(self):
        """Test that stopping manager stops all lanes."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=3)

        await manager.start()
        await manager.stop()

        # All lanes should be stopped
        for lane in manager.lanes:
            assert lane.is_running is False

    async def test_route_event_to_correct_lane(self):
        """Test that events are routed to the correct lane."""
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock()

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        manager = LaneManager(handler_registry, lane_count=12)

        await manager.start()

        try:
            # Create event
            interview_id = str(uuid.uuid4())
            event = EventEnvelope(
                event_type="SentenceCreated",
                aggregate_type=AggregateType.SENTENCE,
                aggregate_id=str(uuid.uuid4()),
                version=0,
                data={"interview_id": interview_id, "text": "Test"},
            )

            # Determine expected lane
            expected_lane = manager.get_lane_for_interview(interview_id)

            # Route event
            checkpoint_called = False

            async def checkpoint_callback():
                nonlocal checkpoint_called
                checkpoint_called = True

            await manager.route_event(event, checkpoint_callback)

            # Give lane time to process
            await asyncio.sleep(0.1)

            # Expected lane should have processed it
            assert expected_lane.events_processed >= 1

        finally:
            await manager.stop()

    async def test_extract_interview_id_from_sentence_event(self):
        """Test extracting interview_id from Sentence event data."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        interview_id = str(uuid.uuid4())
        event = EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={"interview_id": interview_id, "text": "Test"},
        )

        extracted_id = manager._extract_interview_id(event)
        assert extracted_id == interview_id

    async def test_extract_interview_id_from_interview_event(self):
        """Test extracting interview_id from Interview event (aggregate_id)."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        interview_id = str(uuid.uuid4())
        event = EventEnvelope(
            event_type="InterviewCreated",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=interview_id,
            version=0,
            data={"title": "Test Interview"},
        )

        extracted_id = manager._extract_interview_id(event)
        assert extracted_id == interview_id

    async def test_get_status(self):
        """Test getting status of all lanes."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=3)

        await manager.start()

        status = manager.get_status()

        assert status["lane_count"] == 3
        assert len(status["lanes"]) == 3
        assert "total_events_processed" in status
        assert "total_events_failed" in status

        await manager.stop()


@pytest.mark.asyncio
class TestLane:
    """Test individual Lane behavior."""

    async def test_lane_processes_events_in_order(self):
        """Test that lane processes events in the order they're enqueued."""
        # Mock handler that records event order
        processed_events = []

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(
            side_effect=lambda event, lane_id: processed_events.append(event.aggregate_id)
        )

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry)
        await lane.start()

        try:
            # Enqueue 5 events
            event_ids = [str(uuid.uuid4()) for _ in range(5)]
            for event_id in event_ids:
                event = EventEnvelope(
                    event_type="SentenceCreated",
                    aggregate_type=AggregateType.SENTENCE,
                    aggregate_id=event_id,
                    version=0,
                    data={"interview_id": "test", "text": "Test"},
                )
                await lane.enqueue(event, AsyncMock())

            # Wait for processing
            await asyncio.sleep(0.2)

            # Events should be processed in order
            assert processed_events == event_ids

        finally:
            await lane.stop()

    async def test_lane_continues_after_handler_error(self):
        """Test that lane continues processing after a handler error."""
        call_count = 0

        async def failing_handler(event, lane_id):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated error")

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(side_effect=failing_handler)

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry)
        await lane.start()

        try:
            # Enqueue 3 events
            for i in range(3):
                event = EventEnvelope(
                    event_type="SentenceCreated",
                    aggregate_type=AggregateType.SENTENCE,
                    aggregate_id=str(uuid.uuid4()),
                    version=0,
                    data={"interview_id": "test", "text": f"Test {i}"},
                )
                await lane.enqueue(event, AsyncMock())

            # Wait for processing
            await asyncio.sleep(0.2)

            # All 3 should have been attempted (2nd failed, but 3rd still processed)
            assert call_count == 3
            assert lane.events_failed == 1
            assert lane.events_processed == 2

        finally:
            await lane.stop()

    async def test_lane_status(self):
        """Test getting lane status."""
        handler_registry = MagicMock()
        lane = Lane(lane_id=5, handler_registry=handler_registry)

        await lane.start()

        status = lane.get_status()

        assert status["id"] == 5
        assert status["is_running"] is True
        assert status["queue_depth"] == 0
        assert status["events_processed"] == 0
        assert status["events_failed"] == 0

        await lane.stop()

    async def test_checkpoint_called_after_successful_processing(self):
        """Test that checkpoint callback is called after successful processing."""
        checkpoint_called = False

        async def checkpoint_callback():
            nonlocal checkpoint_called
            checkpoint_called = True

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock()

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry)
        await lane.start()

        try:
            event = EventEnvelope(
                event_type="SentenceCreated",
                aggregate_type=AggregateType.SENTENCE,
                aggregate_id=str(uuid.uuid4()),
                version=0,
                data={"interview_id": "test", "text": "Test"},
            )

            await lane.enqueue(event, checkpoint_callback)

            # Wait for processing
            await asyncio.sleep(0.1)

            assert checkpoint_called is True

        finally:
            await lane.stop()

    async def test_checkpoint_called_even_after_handler_failure(self):
        """Test that checkpoint is called even if handler fails (to move past bad event)."""
        checkpoint_called = False

        async def checkpoint_callback():
            nonlocal checkpoint_called
            checkpoint_called = True

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(side_effect=Exception("Simulated error"))

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry)
        await lane.start()

        try:
            event = EventEnvelope(
                event_type="SentenceCreated",
                aggregate_type=AggregateType.SENTENCE,
                aggregate_id=str(uuid.uuid4()),
                version=0,
                data={"interview_id": "test", "text": "Test"},
            )

            await lane.enqueue(event, checkpoint_callback)

            # Wait for processing
            await asyncio.sleep(0.1)

            # Checkpoint should still be called to move past the bad event
            assert checkpoint_called is True

        finally:
            await lane.stop()
