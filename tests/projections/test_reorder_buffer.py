"""
Pure unit tests for WatermarkTracker and ReorderBuffer.

No ESDB/Neo4j/asyncio event loop needed -- all timing is driven through an
injected fake clock, never a real sleep.
"""

from src.projections.reorder_buffer import ReorderBuffer, WatermarkTracker


class FakeClock:
    """Deterministic, manually-advanced monotonic clock for tests."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class TestWatermarkTracker:
    def test_low_watermark_none_when_nothing_registered(self):
        tracker = WatermarkTracker()
        assert tracker.low_watermark() is None

    def test_low_watermark_none_until_all_registered_subs_have_recorded(self):
        tracker = WatermarkTracker()
        tracker.register("interview")
        tracker.register("sentence")

        # Neither sub has recorded yet.
        assert tracker.low_watermark() is None

        tracker.record("interview", 10)
        # "sentence" is registered but hasn't recorded -- still None.
        assert tracker.low_watermark() is None

        tracker.record("sentence", 3)
        # Both registered subs have now recorded at least once.
        assert tracker.low_watermark() == 3

    def test_low_watermark_is_min_across_registered_subs(self):
        tracker = WatermarkTracker()
        tracker.register("interview")
        tracker.register("sentence")
        tracker.register("project")

        tracker.record("interview", 100)
        tracker.record("sentence", 5)
        tracker.record("project", 50)

        assert tracker.low_watermark() == 5

    def test_record_keeps_the_max_per_subscription(self):
        tracker = WatermarkTracker()
        tracker.register("interview")

        tracker.record("interview", 5)
        tracker.record("interview", 3)  # out-of-order delivery, lower position
        tracker.record("interview", 9)

        assert tracker.low_watermark() == 9

    def test_unregistered_subscription_does_not_gate_watermark(self):
        """Only REGISTERED subs count; recording for one registered sub is
        enough once that's the only one registered."""
        tracker = WatermarkTracker()
        tracker.register("interview")
        tracker.record("interview", 7)

        assert tracker.low_watermark() == 7


class TestReorderBufferOrdering:
    def test_pop_ready_releases_in_ascending_commit_position_order(self):
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)

        buffer.add(3, "event-3", lambda: None)
        buffer.add(1, "event-1", lambda: None)
        buffer.add(2, "event-2", lambda: None)

        released = buffer.pop_ready(watermark=3, max_hold_s=10.0)

        assert [e.commit_position for e in released] == [1, 2, 3]
        assert [e.event for e in released] == ["event-1", "event-2", "event-3"]
        assert len(buffer) == 0

    def test_watermark_gate_blocks_until_watermark_reaches_commit_position(self):
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)
        buffer.add(5, "event-5", lambda: None)

        # Watermark below the head's commit_position -- not releasable yet.
        assert buffer.pop_ready(watermark=4, max_hold_s=10.0) == []
        assert len(buffer) == 1

        # Watermark advances to (or past) the head's commit_position.
        released = buffer.pop_ready(watermark=5, max_hold_s=10.0)
        assert [e.commit_position for e in released] == [5]
        assert len(buffer) == 0

    def test_max_hold_flush_when_watermark_stuck(self):
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)
        buffer.add(5, "event-5", lambda: None)

        max_hold_s = 0.25

        # Watermark never advances (e.g. an idle subscription). Not yet aged.
        assert buffer.pop_ready(watermark=None, max_hold_s=max_hold_s) == []
        assert buffer.next_deadline(max_hold_s) == 0.0 + max_hold_s

        clock.advance(max_hold_s)  # exactly at the deadline

        released = buffer.pop_ready(watermark=None, max_hold_s=max_hold_s)
        assert [e.commit_position for e in released] == [5]
        assert len(buffer) == 0
        assert buffer.next_deadline(max_hold_s) is None  # buffer now empty

    def test_ordering_preserved_under_max_hold_low_head_blocks_higher_aged_entry(self):
        """A not-yet-releasable low-commit-position head must block release
        of a higher-position entry even if that higher entry has itself
        aged past max_hold_s -- never release out of order."""
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)
        max_hold_s = 0.25

        # High-position entry added first; ages past max_hold_s.
        buffer.add(100, "event-100", lambda: None)
        clock.advance(max_hold_s)

        # Low-position entry added late -- freshly enqueued, not aged.
        buffer.add(1, "event-1", lambda: None)

        # Head is now commit_position=1 (lower sorts first), enqueued at the
        # current clock time -- not releasable by watermark (None) or age (0).
        released = buffer.pop_ready(watermark=None, max_hold_s=max_hold_s)
        assert released == []
        assert len(buffer) == 2

    def test_none_commit_position_sorts_as_positive_infinity(self):
        """Events without a commit_position (rare defensive case) order after
        positioned events, and only ever release via max_hold."""
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)
        max_hold_s = 0.25

        buffer.add(None, "event-none", lambda: None)
        buffer.add(50, "event-50", lambda: None)

        # Positioned event (50) is head, releasable via watermark.
        released = buffer.pop_ready(watermark=50, max_hold_s=max_hold_s)
        assert [e.event for e in released] == ["event-50"]
        assert len(buffer) == 1

        # The None-position entry never satisfies the watermark gate --
        # only max_hold releases it.
        assert buffer.pop_ready(watermark=99999, max_hold_s=max_hold_s) == []
        clock.advance(max_hold_s)
        released = buffer.pop_ready(watermark=99999, max_hold_s=max_hold_s)
        assert [e.event for e in released] == ["event-none"]


class TestReorderBufferDeferredAck:
    def test_pop_ready_returns_entries_with_callback_but_never_invokes_it(self):
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)

        calls = []
        buffer.add(1, "event-1", lambda: calls.append("ack-1"))

        released = buffer.pop_ready(watermark=1, max_hold_s=10.0)

        assert len(released) == 1
        assert released[0].checkpoint_callback is not None
        assert calls == []  # buffer itself never calls it

        # The caller is expected to invoke it after processing.
        released[0].checkpoint_callback()
        assert calls == ["ack-1"]


class TestReorderBufferNextDeadline:
    def test_next_deadline_none_when_empty(self):
        buffer = ReorderBuffer(clock=FakeClock())
        assert buffer.next_deadline(max_hold_s=0.25) is None

    def test_next_deadline_tracks_the_head_not_later_entries(self):
        clock = FakeClock()
        buffer = ReorderBuffer(clock=clock)
        max_hold_s = 0.25

        buffer.add(10, "first", lambda: None)
        clock.advance(0.1)
        buffer.add(1, "second", lambda: None)  # becomes head (lower commit_position)

        # Head is "second", enqueued at t=0.1 -- deadline is 0.1 + max_hold_s,
        # not 0.0 + max_hold_s (which would be "first"'s deadline).
        assert buffer.next_deadline(max_hold_s) == 0.1 + max_hold_s
