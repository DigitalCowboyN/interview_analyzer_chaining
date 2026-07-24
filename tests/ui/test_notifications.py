"""NotificationHub + scope mapping (M5.1 Task 3) and EsdbWatcher (M5.1 Task 4):
pure in-process pub/sub core for the SSE bridge, plus the ESDB catch-up
watcher that feeds it. `scope_notifications` is the single translation point
from core-domain streams to the thin surface-tag contract the browser sees —
no event types, stream names, or ESDB concepts leak past it.

The EsdbWatcher tests below use a fake client (queue-driven sync iterator,
no real ESDB) that mirrors the sentinel-pull idiom verified against
esdbclient's actual CatchupSubscription in
src/projections/subscription_manager.py: stop() on a real esdbclient
subscription cancels its gRPC call, which makes a blocked next() raise
StopIteration (see esdbclient/streams.py's ReadResponse/CatchupSubscription) —
FakeSubscription.stop() reproduces exactly that.
"""

import asyncio
import json
import queue as thread_queue
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from src.ui.notifications import (
    EsdbWatcher,
    Notification,
    NotificationHub,
    get_live_feed,
    scope_notifications,
)

IID = "iv-1"
PID = "proj-1"


# ---------------------------------------------------------------------------
# scope_notifications: pure mapping, no hub involved
# ---------------------------------------------------------------------------


def test_sentence_stream_maps_to_transcript_notification():
    result = scope_notifications(f"Sentence-{IID}", {"interview_id": IID})
    assert result == [Notification("transcript", interview_id=IID)]


def test_sentence_stream_missing_interview_id_returns_empty():
    assert scope_notifications(f"Sentence-{IID}", {}) == []


def test_interview_stream_maps_to_transcript_notification_using_stream_suffix():
    result = scope_notifications(f"Interview-{IID}", {})
    assert result == [Notification("transcript", interview_id=IID)]


def test_bare_interview_stream_with_empty_suffix_returns_empty():
    # Malformed stream name "Interview-" (no id) must not produce a
    # Notification with interview_id="" -- same falsy-id guard as the
    # Sentence-/Project- branches.
    assert scope_notifications("Interview-", {}) == []


def test_interview_stream_with_project_id_adds_interviews_notification():
    result = scope_notifications(f"Interview-{IID}", {"project_id": PID})
    assert result == [
        Notification("transcript", interview_id=IID),
        Notification("interviews", project_id=PID),
    ]


def test_project_stream_maps_to_project_notification():
    result = scope_notifications(f"Project-{PID}", {"project_id": PID})
    assert result == [Notification("project", project_id=PID)]


def test_project_stream_missing_project_id_returns_empty():
    assert scope_notifications(f"Project-{PID}", {}) == []


def test_unknown_stream_returns_empty():
    assert scope_notifications("SomeOtherStream-abc", {"interview_id": IID}) == []


# ---------------------------------------------------------------------------
# NotificationHub: fan-out, matching, resync, unsubscribe, drop-oldest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_transcript_notification_reaches_only_matching_subscriber():
    hub = NotificationHub()
    sub_a = hub.subscribe(interview_id=IID, project_id=None)
    sub_b = hub.subscribe(interview_id="iv-other", project_id=None)

    hub.publish(Notification("transcript", interview_id=IID))

    assert sub_a.queue.qsize() == 1
    assert sub_a.queue.get_nowait() == Notification("transcript", interview_id=IID)
    assert sub_b.queue.qsize() == 0


@pytest.mark.asyncio
async def test_publish_project_notification_reaches_only_matching_subscriber():
    hub = NotificationHub()
    sub_a = hub.subscribe(interview_id=None, project_id=PID)
    sub_b = hub.subscribe(interview_id=None, project_id="proj-other")

    hub.publish(Notification("project", project_id=PID))

    assert sub_a.queue.qsize() == 1
    assert sub_b.queue.qsize() == 0


@pytest.mark.asyncio
async def test_publish_interviews_notification_matches_on_project_id():
    hub = NotificationHub()
    sub_a = hub.subscribe(interview_id=None, project_id=PID)
    sub_b = hub.subscribe(interview_id=None, project_id="proj-other")

    hub.publish(Notification("interviews", project_id=PID))

    assert sub_a.queue.qsize() == 1
    assert sub_b.queue.qsize() == 0


@pytest.mark.asyncio
async def test_broadcast_resync_reaches_all_subscribers_regardless_of_scope():
    hub = NotificationHub()
    sub_a = hub.subscribe(interview_id=IID, project_id=None)
    sub_b = hub.subscribe(interview_id=None, project_id=PID)

    hub.broadcast_resync()

    assert sub_a.queue.get_nowait() == Notification("resync")
    assert sub_b.queue.get_nowait() == Notification("resync")


@pytest.mark.asyncio
async def test_close_unregisters_subscriber_and_decrements_count():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)
    assert hub.subscriber_count == 1

    sub.close()

    assert hub.subscriber_count == 0
    hub.publish(Notification("transcript", interview_id=IID))
    assert sub.queue.qsize() == 0


@pytest.mark.asyncio
async def test_subscriber_count_tracks_multiple_subscribers():
    hub = NotificationHub()
    assert hub.subscriber_count == 0
    sub_a = hub.subscribe(interview_id=IID, project_id=None)
    sub_b = hub.subscribe(interview_id=None, project_id=PID)
    assert hub.subscriber_count == 2
    sub_a.close()
    assert hub.subscriber_count == 1
    sub_b.close()
    assert hub.subscriber_count == 0


@pytest.mark.asyncio
async def test_full_queue_drops_oldest_without_raising():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    # Fill the bounded queue (maxsize 64) then publish one more — must not raise.
    for _ in range(64):
        hub.publish(Notification("transcript", interview_id=IID))
    assert sub.queue.qsize() == 64

    # Mark the very first item distinctly so we can prove it was dropped.
    # Drain and re-fill with a distinguishable sentinel as the oldest entry.
    while not sub.queue.empty():
        sub.queue.get_nowait()

    sentinel_oldest = Notification("transcript", interview_id="sentinel-oldest")
    sub.queue.put_nowait(sentinel_oldest)
    for i in range(63):
        sub.queue.put_nowait(Notification("transcript", interview_id=f"filler-{i}"))
    assert sub.queue.full()

    # Must carry the subscriber's interview_id so hub.publish's matching
    # actually routes it to this subscriber (unmatched notifications are
    # never enqueued at all, so they wouldn't exercise drop-oldest).
    newest = Notification("transcript", interview_id=IID)
    hub.publish(newest)
    assert sub.queue.qsize() == 64
    # Oldest (sentinel) must have been dropped, newest must be present.
    remaining = []
    while not sub.queue.empty():
        remaining.append(sub.queue.get_nowait())
    assert sentinel_oldest not in remaining
    assert newest in remaining
    assert remaining[-1] == newest


# ---------------------------------------------------------------------------
# EsdbWatcher: fake queue-driven client, no real ESDB
# ---------------------------------------------------------------------------


@dataclass
class FakeRecordedEvent:
    """Stands in for esdbclient.RecordedEvent: only the two fields the
    watcher actually reads (the resolved stream name and raw data bytes)."""

    stream_name: str
    data: bytes


class FakeSubscription:
    """Sync iterator standing in for esdbclient's CatchupSubscription: pulls
    from a thread-safe queue.Queue so `asyncio.to_thread(next, iterator, ...)`
    behaves like the real blocking-network-read idiom without any network —
    a blocked get() runs on a worker thread and doesn't stall the event loop.
    stop() unblocks a pending get() by raising StopIteration, mirroring how
    esdbclient's real subscription.stop() makes the next blocked next() call
    raise StopIteration (verified in esdbclient/streams.py)."""

    def __init__(self):
        self._queue: "thread_queue.Queue" = thread_queue.Queue()
        self.stopped = False

    def push_event(self, event: FakeRecordedEvent) -> None:
        self._queue.put(event)

    def push_exception(self, exc: BaseException) -> None:
        self._queue.put(exc)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self) -> None:
        self.stopped = True
        self._queue.put(StopIteration())


class FakeEventStoreDBClient:
    """Stands in for esdbclient.EventStoreDBClient: hands out queued
    FakeSubscription objects per stream_name, or a fresh empty one (which
    just blocks, like a quiet real stream) if none was queued for that
    stream. Records every handed-out subscription so tests can assert on
    `.stopped` after EsdbWatcher.stop()."""

    def __init__(self):
        self.subscribe_calls: list = []
        self.handed_out: list = []
        self._queued: dict = {}

    def queue_subscription(self, stream_name: str, subscription: FakeSubscription) -> None:
        self._queued.setdefault(stream_name, []).append(subscription)

    def subscribe_to_stream(self, stream_name: str, *, from_end: bool, resolve_links: bool):
        assert from_end is True
        assert resolve_links is True
        self.subscribe_calls.append(stream_name)
        pending = self._queued.get(stream_name)
        subscription = pending.pop(0) if pending else FakeSubscription()
        self.handed_out.append(subscription)
        return subscription


class FakeEventStore:
    """Stands in for src.events.store.EventStoreClient: exposes the same
    `get_client()` async context manager the watcher relies on."""

    def __init__(self, client: FakeEventStoreDBClient):
        self._client = client

    @asynccontextmanager
    async def get_client(self):
        yield self._client


SENTENCE_STREAM = "$ce-Sentence"
ALL_WATCHED_STREAMS = ("$ce-Interview", "$ce-Sentence", "$ce-Project")


def make_watcher(client: FakeEventStoreDBClient, hub: NotificationHub) -> EsdbWatcher:
    # backoff_seconds=(0, 0): no meaningful delay, so a subscription-error
    # test doesn't need a real wait or to monkeypatch asyncio.sleep.
    return EsdbWatcher(hub, event_store=FakeEventStore(client), backoff_seconds=(0, 0))


@pytest.mark.asyncio
async def test_watcher_publishes_notifications_from_subscribed_events():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    client = FakeEventStoreDBClient()
    sentence_sub = FakeSubscription()
    sentence_sub.push_event(FakeRecordedEvent(f"Sentence-{IID}", json.dumps({"interview_id": IID}).encode()))
    client.queue_subscription(SENTENCE_STREAM, sentence_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("transcript", interview_id=IID)
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_skips_malformed_event_and_keeps_looping():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    client = FakeEventStoreDBClient()
    sentence_sub = FakeSubscription()
    sentence_sub.push_event(FakeRecordedEvent(f"Sentence-{IID}", b"not-json"))
    sentence_sub.push_event(FakeRecordedEvent(f"Sentence-{IID}", json.dumps({"interview_id": IID}).encode()))
    client.queue_subscription(SENTENCE_STREAM, sentence_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("transcript", interview_id=IID)
        # The malformed event ahead of it produced nothing -- if it had
        # crashed the loop instead of being skipped, this valid event
        # would never have been reached at all.
        assert sub.queue.qsize() == 0
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_skips_valid_json_non_dict_payload_and_keeps_looping():
    # Valid JSON that isn't the dict shape scope_notifications expects (a
    # bare list here) must be treated as malformed too -- not left to raise
    # out of _handle_event and get mistaken for a subscription failure.
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    client = FakeEventStoreDBClient()
    sentence_sub = FakeSubscription()
    sentence_sub.push_event(FakeRecordedEvent(f"Sentence-{IID}", json.dumps([1, 2, 3]).encode()))
    sentence_sub.push_event(FakeRecordedEvent(f"Sentence-{IID}", json.dumps({"interview_id": IID}).encode()))
    client.queue_subscription(SENTENCE_STREAM, sentence_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("transcript", interview_id=IID)
        assert sub.queue.qsize() == 0
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_resubscribes_and_broadcasts_resync_after_subscription_error():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    client = FakeEventStoreDBClient()
    broken_sub = FakeSubscription()
    broken_sub.push_exception(RuntimeError("connection dropped"))
    client.queue_subscription(SENTENCE_STREAM, broken_sub)
    healthy_sub = FakeSubscription()  # resubscribe target; left quiet after that
    client.queue_subscription(SENTENCE_STREAM, healthy_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        # broadcast_resync reaches every subscriber unconditionally, so
        # receiving it here proves the watcher survived the exception.
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("resync")
        assert client.subscribe_calls.count(SENTENCE_STREAM) == 2
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_ensure_started_is_idempotent_under_concurrent_calls():
    hub = NotificationHub()
    client = FakeEventStoreDBClient()
    watcher = make_watcher(client, hub)

    await asyncio.gather(watcher.ensure_started(), watcher.ensure_started())

    # Exactly one subscribe per watched stream -- a second, non-idempotent
    # ensure_started would have doubled this to 6.
    assert sorted(client.subscribe_calls) == sorted(ALL_WATCHED_STREAMS)
    await watcher.stop()


@pytest.mark.asyncio
async def test_stop_cancels_tasks_and_stops_active_subscriptions():
    hub = NotificationHub()
    client = FakeEventStoreDBClient()
    watcher = make_watcher(client, hub)

    await watcher.ensure_started()
    tasks = list(watcher._tasks.values())
    assert tasks and all(not task.done() for task in tasks)

    await watcher.stop()

    assert all(task.done() for task in tasks)
    assert watcher._tasks == {}
    assert watcher._active_subscriptions == {}
    assert all(sub.stopped for sub in client.handed_out)


@pytest.mark.asyncio
async def test_watcher_restartable_after_cancelled_stop():
    # stop() runs in the SSE generator's finally, where a client disconnect
    # can deliver CancelledError at stop()'s first suspension point. State
    # must already be detached by then (synchronously), so a later
    # ensure_started starts fresh tasks instead of no-oping forever against
    # the dead ones (the "bricked watcher" failure mode).
    hub = NotificationHub()
    client = FakeEventStoreDBClient()
    watcher = make_watcher(client, hub)

    await watcher.ensure_started()
    first_tasks = list(watcher._tasks.values())
    await asyncio.sleep(0)  # one loop turn: watch tasks reach their blocking pull
    assert len(client.subscribe_calls) == 3

    stop_task = asyncio.create_task(watcher.stop())
    # One more turn: stop() runs its synchronous section (detach state, stop
    # subscriptions, cancel tasks) and parks at its first `await task`.
    # sleep(0) is a bare yield, not a timed sleep.
    await asyncio.sleep(0)
    stop_task.cancel()
    try:
        await stop_task
    except asyncio.CancelledError:
        # Whether the cancellation propagates or the interrupted await
        # resolves as the (already-cancelled) watch task finishing is a
        # scheduler detail -- the invariants below hold either way, and
        # they are what "not bricked" means.
        pass

    # The interrupted stop already detached state and cancelled the tasks
    # (all synchronously, before its first await).
    assert watcher._tasks == {}
    assert watcher._active_subscriptions == {}
    # Drain the cancelled watch tasks deterministically (pristine teardown).
    await asyncio.gather(*first_tasks, return_exceptions=True)

    # A fresh start works: three new subscriptions, three live tasks -- no
    # zombie no-op against the dead ones.
    await watcher.ensure_started()
    await asyncio.sleep(0)  # let the fresh watch tasks reach their pull
    assert len(client.subscribe_calls) == 6
    assert sorted(watcher._tasks) == sorted(ALL_WATCHED_STREAMS)
    assert all(not task.done() for task in watcher._tasks.values())
    await watcher.stop()


@pytest.mark.asyncio
async def test_stop_survives_subscription_stop_raising():
    # Hardening: one subscription raising from .stop() must not abort
    # shutdown -- remaining subscriptions still stopped, tasks still
    # cancelled, state still cleared.
    hub = NotificationHub()
    client = FakeEventStoreDBClient()
    watcher = make_watcher(client, hub)

    await watcher.ensure_started()
    await asyncio.sleep(0)  # let watch tasks register their subscriptions
    assert len(watcher._active_subscriptions) == 3

    first_sub = watcher._active_subscriptions["$ce-Interview"]
    original_stop = first_sub.stop
    call_state = {"raised": False}

    def raising_stop():
        # Terminate the stream (so its worker thread unblocks and the watch
        # task can actually finish -- keeping the test deterministic), then
        # raise: models a gRPC teardown error surfacing from .stop().
        original_stop()
        call_state["raised"] = True
        raise RuntimeError("gRPC teardown failed")

    first_sub.stop = raising_stop

    await watcher.stop()

    assert call_state["raised"] is True
    assert watcher._tasks == {}
    assert watcher._active_subscriptions == {}
    # The OTHER two subscriptions were still stopped despite the raise.
    others = [s for s in client.handed_out if s is not first_sub]
    assert others and all(s.stopped for s in others)


@pytest.mark.asyncio
async def test_get_live_feed_returns_same_singleton_pair_across_calls(monkeypatch):
    import src.ui.notifications as notifications_module

    monkeypatch.setattr(notifications_module, "_hub", None)
    monkeypatch.setattr(notifications_module, "_watcher", None)

    hub_a, watcher_a = get_live_feed()
    hub_b, watcher_b = get_live_feed()

    assert hub_a is hub_b
    assert watcher_a is watcher_b
    assert isinstance(hub_a, NotificationHub)
    assert isinstance(watcher_a, EsdbWatcher)
