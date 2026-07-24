"""
Per-lane commit-position reordering.

EventStoreDB delivers events to a lane via three independent persistent
subscriptions (interview/sentence/project category streams). Delivery order
across those subscriptions is arrival order, not causal order, so a lane can
see e.g. an UtteranceIdentified event before the SpeakerCreated event it
depends on even though the store's `commit_position` (a global, monotonic
position in the whole event store) says the speaker event happened first.

`WatermarkTracker` tracks, per subscription, the highest `commit_position`
delivered so far; its `low_watermark()` is the minimum across all registered
subscriptions -- the point below which no subscription can possibly still be
holding an undelivered earlier event. `ReorderBuffer` holds a lane's
in-flight events ordered by `commit_position` and only releases (in strictly
ascending order) once an entry is provably safe to release: either the
watermark has passed it, or it has been held long enough (`max_hold_s`) that
we give up waiting (an idle/stalled subscription must not stall the lane
forever; a genuinely late lower-position event is then the
ReferentNotReadyError backstop's job).
"""

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class WatermarkTracker:
    """
    Tracks per-subscription high-water commit positions and exposes the
    minimum across all registered subscriptions (the "low watermark").
    """

    def __init__(self) -> None:
        self._high_water: Dict[str, int] = {}
        self._registered: Set[str] = set()

    def register(self, subscription_name: str) -> None:
        """Declare a subscription that will eventually call `record()`."""
        self._registered.add(subscription_name)

    def record(self, subscription_name: str, commit_position: int) -> None:
        """Record a delivered commit_position; high-water is monotonic (max)."""
        current = self._high_water.get(subscription_name)
        self._high_water[subscription_name] = commit_position if current is None else max(current, commit_position)

    def low_watermark(self) -> Optional[int]:
        """
        Minimum high-water commit_position across all REGISTERED subscriptions.

        Returns None if no subscription is registered, or if any registered
        subscription has not yet delivered (and thus recorded) at least one
        event -- until then we don't know a safe lower bound.
        """
        if not self._registered:
            return None
        values = []
        for name in self._registered:
            if name not in self._high_water:
                return None
            values.append(self._high_water[name])
        return min(values)


@dataclass
class ReorderEntry:
    """A single buffered (event, checkpoint_callback) pair awaiting release."""

    commit_position: Optional[int]
    event: Any
    checkpoint_callback: Callable
    enqueue_time: float


class ReorderBuffer:
    """
    Holds a lane's in-flight events ordered by `commit_position` ascending
    and releases them, strictly in that order, once safe (see module
    docstring for the release algorithm).
    """

    def __init__(self, clock=None):
        """
        Args:
            clock: zero-arg callable returning monotonic seconds. Defaults to
                asyncio's event-loop clock (NOT wall-clock time.time/Date.now)
                so it agrees with `asyncio.wait_for` timeouts used by the
                Lane processing loop.
        """
        if clock is None:
            import asyncio

            clock = lambda: asyncio.get_event_loop().time()  # noqa: E731
        self._clock = clock
        # Heap items: (sort_key, sequence, ReorderEntry). `commit_position is
        # None` sorts as +infinity (positioned events always order first);
        # `sequence` is a tiebreaker so heapq never has to compare
        # ReorderEntry/event objects (which are not orderable).
        self._heap: List[Tuple[float, int, ReorderEntry]] = []
        self._seq = itertools.count()

    def add(self, commit_position: Optional[int], event: Any, checkpoint_callback: Callable) -> None:
        """Buffer an event; enqueue time is captured now via the injected clock."""
        sort_key = commit_position if commit_position is not None else math.inf
        entry = ReorderEntry(
            commit_position=commit_position,
            event=event,
            checkpoint_callback=checkpoint_callback,
            enqueue_time=self._clock(),
        )
        heapq.heappush(self._heap, (sort_key, next(self._seq), entry))

    def _head_releasable(self, watermark: Optional[int], max_hold_s: float) -> bool:
        _, _, entry = self._heap[0]
        if watermark is not None and entry.commit_position is not None and entry.commit_position <= watermark:
            return True
        if self._clock() - entry.enqueue_time >= max_hold_s:
            return True
        return False

    def pop_ready(self, watermark: Optional[int], max_hold_s: float) -> List[ReorderEntry]:
        """
        Release entries in ascending commit_position order, stopping at the
        first head that is not yet releasable (never skips ahead -- see
        module docstring).
        """
        released: List[ReorderEntry] = []
        while self._heap and self._head_releasable(watermark, max_hold_s):
            _, _, entry = heapq.heappop(self._heap)
            released.append(entry)
        return released

    def next_deadline(self, max_hold_s: float) -> Optional[float]:
        """Clock-time at which the current head becomes max_hold-releasable, or None if empty."""
        if not self._heap:
            return None
        _, _, entry = self._heap[0]
        return entry.enqueue_time + max_hold_s

    def __len__(self) -> int:
        return len(self._heap)
