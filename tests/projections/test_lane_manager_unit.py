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
from src.projections.reorder_buffer import WatermarkTracker


@pytest.mark.asyncio
class TestLanePartitioning:
    """Test lane partitioning logic."""

    async def test_consistent_hashing(self):
        """Test that same interview_id always maps to same lane."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=12)

        interview_id = str(uuid.uuid4())

        # Get lane 10 times for same interview_id
        lanes = [manager.get_lane_for_interview(interview_id) for _ in range(10)]

        # All should be the same lane
        assert all(lane == lanes[0] for lane in lanes)

    async def test_different_interviews_can_map_to_different_lanes(self):
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

    async def test_lane_distribution(self):
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

    async def test_hash_matches_expected_algorithm(self):
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

        # max_hold_s tiny: this test cares about routing correctness, not
        # commit_position ordering, and the event has no commit_position
        # (None), so the default 250ms hold would make the 0.1s sleep below
        # flaky/failing -- force near-immediate release instead.
        manager = LaneManager(handler_registry, lane_count=12, max_hold_s=0.0)

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

    async def test_project_events_lane_key_is_aggregate_id(self):
        """Project events route by aggregate_id — one lane per project, never dropped."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=4)
        event = MagicMock()
        event.aggregate_type = "Project"
        event.aggregate_id = "abc-123"
        assert manager._extract_interview_id(event) == "abc-123"

    async def test_segment_identified_lane_key_is_aggregate_id(self):
        """SegmentIdentified (Interview stream, M4.5c) lane-keys by aggregate_id."""
        handler_registry = MagicMock()
        manager = LaneManager(handler_registry, lane_count=4)
        interview_id = str(uuid.uuid4())
        event = EventEnvelope(
            event_type="SegmentIdentified",
            aggregate_type=AggregateType.INTERVIEW,
            aggregate_id=interview_id,
            version=0,
            data={
                "segment_id": str(uuid.uuid4()),
                "topic": "onboarding",
                "start_index": 0,
                "end_index": 2,
                "confidence": 0.9,
            },
        )
        assert manager._extract_interview_id(event) == interview_id

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

        # max_hold_s=0.0: these events carry no commit_position (None, all
        # tied at "+infinity"), so release order falls back to FIFO/insertion
        # order (the buffer's stable tiebreak) -- force near-immediate
        # release so the test's real-time sleep stays fast and reliable.
        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
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

        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
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

        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
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

        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
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

    async def test_sync_checkpoint_callback_is_called_without_raising(self):
        """Regression (defect 1): SubscriptionManager passes a SYNC checkpoint
        callback (e.g. `lambda event_id=...: subscription.ack(event_id)`),
        which returns None. `_process_event` must call it and must not treat
        the None return value as awaitable (previously raised
        `TypeError: object NoneType can't be used in 'await' expression`,
        swallowed by the lane's broad exception handler, so acks silently
        never happened)."""
        checkpoint_calls = []

        def sync_checkpoint_callback():
            checkpoint_calls.append(1)

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock()

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
        await lane.start()

        try:
            event = EventEnvelope(
                event_type="SentenceCreated",
                aggregate_type=AggregateType.SENTENCE,
                aggregate_id=str(uuid.uuid4()),
                version=0,
                data={"interview_id": "test", "text": "Test"},
            )

            await lane.enqueue(event, sync_checkpoint_callback)

            # Wait for processing
            await asyncio.sleep(0.1)

            assert checkpoint_calls == [1]
            # No error should have been recorded (the old bug raised a
            # TypeError, caught by the broad handler and counted as failed).
            assert lane.events_failed == 0
            assert lane.events_processed == 1

        finally:
            await lane.stop()

    async def test_async_checkpoint_callback_still_awaited(self):
        """Regression: async checkpoint callbacks (the old contract, still
        used by existing tests/callers) must continue to be awaited."""
        checkpoint_called = False

        async def async_checkpoint_callback():
            nonlocal checkpoint_called
            await asyncio.sleep(0)  # prove it actually runs as a coroutine
            checkpoint_called = True

        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock()

        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=0.0)
        await lane.start()

        try:
            event = EventEnvelope(
                event_type="SentenceCreated",
                aggregate_type=AggregateType.SENTENCE,
                aggregate_id=str(uuid.uuid4()),
                version=0,
                data={"interview_id": "test", "text": "Test"},
            )

            await lane.enqueue(event, async_checkpoint_callback)

            # Wait for processing
            await asyncio.sleep(0.1)

            assert checkpoint_called is True
            assert lane.events_failed == 0
            assert lane.events_processed == 1

        finally:
            await lane.stop()


@pytest.mark.asyncio
class TestLaneReorderBuffer:
    """
    Integration-of-units: Lane wired with a ReorderBuffer (+ shared/fake
    WatermarkTracker) releases buffered, out-of-order arrivals in ascending
    commit_position order, and defers the checkpoint (ack) until each entry
    is actually released -- never merely on enqueue/buffering.
    """

    @staticmethod
    def _make_event(commit_position):
        return EventEnvelope(
            event_type="SentenceCreated",
            aggregate_type=AggregateType.SENTENCE,
            aggregate_id=str(uuid.uuid4()),
            version=0,
            data={"interview_id": "test-interview", "text": "Test"},
            commit_position=commit_position,
        )

    async def test_out_of_order_arrivals_processed_in_commit_position_order(self):
        """Events enqueued out of order (3, 1, 2) are handled in ascending
        commit_position order (1, 2, 3), via the max_hold aging path (no
        watermark tracker registered -> low_watermark() is always None)."""
        processed = []
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(
            side_effect=lambda event, lane_id: processed.append(event.commit_position)
        )
        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        acked = []

        def make_cb(cp):
            def cb():
                acked.append(cp)

            return cb

        max_hold_s = 0.05
        lane = Lane(lane_id=0, handler_registry=handler_registry, max_hold_s=max_hold_s)

        # Enqueue out of commit_position order, BEFORE starting the lane, so
        # all three are already queued when the processing loop begins --
        # they land in the buffer together rather than racing the loop.
        for cp in (3, 1, 2):
            await lane.enqueue(self._make_event(cp), make_cb(cp))

        await lane.start()
        try:
            # Not yet aged (or watermark-gated) -- nothing released.
            assert processed == []
            assert acked == []

            # Wait past max_hold_s (generous slack for scheduling) so the
            # aged-out flush releases everything, in ascending order.
            await asyncio.sleep(max_hold_s + 0.2)

            assert processed == [1, 2, 3]
            assert acked == [1, 2, 3]
        finally:
            await lane.stop()

    async def test_checkpoint_deferred_until_watermark_releases_entry(self):
        """Deferred ack: a buffered event's checkpoint_callback must not fire
        merely because it was enqueued -- only once it's actually released
        (here, gated by the watermark rather than max_hold aging)."""
        processed = []
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(
            side_effect=lambda event, lane_id: processed.append(event.commit_position)
        )
        handler_registry = MagicMock()
        handler_registry.get_handler = MagicMock(return_value=mock_handler)

        acked = []

        def make_cb(cp):
            def cb():
                acked.append(cp)

            return cb

        class ManualWatermark:
            """Minimal low_watermark()-duck-typed stub, manually controlled
            by the test. WatermarkTracker's own per-subscription bookkeeping
            is covered by test_reorder_buffer.py -- this isolates Lane's
            wiring to "whatever object provides low_watermark()"."""

            def __init__(self):
                self.value = None

            def low_watermark(self):
                return self.value

        watermark = ManualWatermark()
        # max_hold_s intentionally large: only the watermark gate should be
        # able to release the buffered entry within this test's run time.
        lane = Lane(lane_id=0, handler_registry=handler_registry, watermark_tracker=watermark, max_hold_s=10.0)

        await lane.enqueue(self._make_event(5), make_cb(5))
        await lane.start()
        try:
            await asyncio.sleep(0.1)
            # Buffered, watermark still None -- not releasable, not acked.
            assert processed == []
            assert acked == []

            # Advance the watermark past the buffered entry's position, and
            # enqueue a fresh (not-yet-releasable) event so the loop's
            # blocked queue.get() wakes immediately and re-drains, instead
            # of waiting on the bounded (<=1s) periodic re-check.
            watermark.value = 5
            await lane.enqueue(self._make_event(6), make_cb(6))

            await asyncio.sleep(0.1)

            # Only the newly-releasable entry (commit_position 5) fired --
            # commit_position 6 is still buffered (watermark hasn't reached
            # it) and its callback has NOT been invoked.
            assert processed == [5]
            assert acked == [5]
            assert len(lane._buffer) == 1
        finally:
            await lane.stop()
