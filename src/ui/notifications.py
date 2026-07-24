"""UI notifications (M5.1): pub/sub core for the SSE bridge.

Pure in-process building blocks — no ESDB, no FastAPI, no framework
coupling — mirroring the reader idiom (plain classes/functions, unit
testable in isolation). `scope_notifications` is the single translation
point from core-domain streams (Sentence-*, Interview-*, Project-*) to the
thin surface-tag contract the browser sees: `{surface, interview_id?,
project_id?}`. The browser must never learn about event types, stream
names, or ESDB concepts — everything upstream of this module speaks
core-domain, everything downstream speaks only surface tags.

Task 4 adds `EsdbWatcher`: it bridges ESDB catch-up subscriptions on the
three notification-relevant category streams to the `NotificationHub`
above, and `get_live_feed()`, a module-level lazy singleton pair the SSE
route (src/api/routers/ui.py) uses for its lazy start/stop lifecycle.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.events.store import EventStoreClient, get_event_store_client

logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 64


@dataclass(frozen=True)
class Notification:
    """Thin surface tag delivered to a subscriber's queue. `surface` is one
    of "transcript" | "interviews" | "project" | "resync"; the id fields are
    the only scoping the browser gets — no stream names, no event types."""

    surface: str
    interview_id: Optional[str] = None
    project_id: Optional[str] = None


def scope_notifications(stream_name: str, payload: dict) -> List[Notification]:
    """Map one core-domain stream event to zero or more surface notifications.

    Pure function: no I/O, no hub access. Returns [] when the stream isn't
    one we notify on, or when a required id is missing from the payload
    (log-and-skip for the missing-key case is the caller's job, not ours).
    """
    if stream_name.startswith("Sentence-"):
        interview_id = payload.get("interview_id")
        if not interview_id:
            return []
        return [Notification("transcript", interview_id=interview_id)]

    if stream_name.startswith("Interview-"):
        # The interview id is the stream-name suffix itself (verified fact),
        # so this notification never depends on payload contents. Guard the
        # degenerate bare "Interview-" case (empty suffix) the same way the
        # sibling branches guard their required id -- a malformed stream name
        # must never produce a Notification with interview_id="".
        interview_id = stream_name[len("Interview-") :]
        if not interview_id:
            return []
        notifications = [Notification("transcript", interview_id=interview_id)]
        project_id = payload.get("project_id")
        if project_id:
            notifications.append(Notification("interviews", project_id=project_id))
        return notifications

    if stream_name.startswith("Project-"):
        project_id = payload.get("project_id")
        if not project_id:
            return []
        return [Notification("project", project_id=project_id)]

    return []


class Subscription:
    """One subscriber's handle: a bounded queue to read from, and `close()`
    to unregister from the hub. Callers (SSE endpoint handlers, Task 4/5)
    read `.queue` and call `.close()` when the connection ends."""

    def __init__(self, hub: "NotificationHub", interview_id: Optional[str], project_id: Optional[str]):
        self._hub = hub
        self.interview_id = interview_id
        self.project_id = project_id
        self.queue: "asyncio.Queue[Notification]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    def close(self) -> None:
        self._hub._unregister(self)


class NotificationHub:
    """In-process pub/sub: one hub per process, subscribers register a scope
    (interview_id and/or project_id), `publish` fans out to whichever
    subscribers match. `resync` reaches everyone unconditionally."""

    def __init__(self):
        self._subscriptions: List[Subscription] = []

    def subscribe(self, interview_id: Optional[str], project_id: Optional[str]) -> Subscription:
        subscription = Subscription(self, interview_id=interview_id, project_id=project_id)
        self._subscriptions.append(subscription)
        return subscription

    def _unregister(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    def publish(self, notification: Notification) -> None:
        for subscription in self._subscriptions:
            if self._matches(notification, subscription):
                self._enqueue(subscription.queue, notification)

    def broadcast_resync(self) -> None:
        resync = Notification("resync")
        for subscription in self._subscriptions:
            self._enqueue(subscription.queue, resync)

    @staticmethod
    def _matches(notification: Notification, subscription: Subscription) -> bool:
        if notification.surface == "transcript":
            return notification.interview_id == subscription.interview_id
        if notification.surface in ("interviews", "project"):
            return notification.project_id == subscription.project_id
        return False

    @staticmethod
    def _enqueue(queue: "asyncio.Queue[Notification]", notification: Notification) -> None:
        """Bounded queue, never blocks the publisher. On overflow, drop the
        oldest item and retry — a slow browser self-heals via the next
        notification rather than stalling the watcher that feeds every
        subscriber."""
        try:
            queue.put_nowait(notification)
        except asyncio.QueueFull:
            queue.get_nowait()
            queue.put_nowait(notification)


def format_sse_event(notification: Notification) -> str:
    """Render one Notification as an SSE `data:` frame containing only its
    non-None fields — e.g. a "transcript" notification never carries a
    stray `"project_id": null`. Pure formatting, no framework coupling; the
    SSE route (src/api/routers/ui.py) is the only caller."""
    fields = {key: value for key, value in asdict(notification).items() if value is not None}
    return f"data: {json.dumps(fields)}\n\n"


# ---------------------------------------------------------------------------
# EsdbWatcher: bridges ESDB catch-up subscriptions to the NotificationHub.
# ---------------------------------------------------------------------------

# Category streams the watcher subscribes to; scope_notifications maps events
# from each back to the surface tags subscribers care about.
_WATCHED_STREAMS: Tuple[str, ...] = ("$ce-Interview", "$ce-Sentence", "$ce-Project")

# Sentinel returned by asyncio.to_thread(next, iterator, _SUBSCRIPTION_ENDED)
# when the underlying (blocking, sync) esdbclient iterator has nothing more
# to give without blocking further -- mirrors
# src/projections/subscription_manager.py's identical sentinel idiom.
_SUBSCRIPTION_ENDED = object()


class EsdbWatcher:
    """Bridges three ESDB catch-up subscriptions ($ce-Interview, $ce-Sentence,
    $ce-Project) to a NotificationHub. Lazy lifecycle by design: the SSE route
    calls `ensure_started()` on first connect and `stop()` once the last
    subscriber disconnects (`hub.subscriber_count == 0`) -- there is no
    lifespan/startup coupling.

    Mirrors the M4.7 sentinel-pull idiom from
    `subscription_manager.py::_run_subscription`: sync esdbclient iterators
    are consumed via `await asyncio.to_thread(next, iterator, sentinel)` so a
    blocking network read never starves the event loop. Subscriptions are
    catch-up (`from_end=True, resolve_links=True`), not persistent -- no
    consumer groups, no acking.
    """

    def __init__(
        self,
        hub: NotificationHub,
        event_store: Optional[EventStoreClient] = None,
        backoff_seconds: Sequence[float] = (1, 2, 5, 10),
    ):
        self._hub = hub
        self._event_store = event_store or get_event_store_client()
        self._backoff_seconds: Tuple[float, ...] = tuple(backoff_seconds)
        self._tasks: Dict[str, asyncio.Task] = {}
        # Active esdbclient subscription object per stream name, for as long
        # as that stream's outer loop iteration is connected. Held so stop()
        # can call .stop() on it directly -- required to unblock a worker
        # thread parked in asyncio.to_thread(next, ...); see stop()'s
        # docstring (same rationale as SubscriptionManager.stop()).
        self._active_subscriptions: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        """Idempotent: spawns one watch task per category stream on the first
        call. Concurrent callers are serialized by the lock so exactly one
        set of tasks is ever created, even if two SSE connections race to
        start the watcher at once.

        Liveness check, not just presence check: done tasks (e.g. left
        behind by a stop() that was itself cancelled mid-await, or torn
        down by a stop() that raced this call to the lock) are pruned
        first, so a dead watcher restarts instead of no-oping forever
        against zombie entries."""
        async with self._lock:
            self._tasks = {name: task for name, task in self._tasks.items() if not task.done()}
            for stream_name in _WATCHED_STREAMS:
                if stream_name not in self._tasks:
                    self._tasks[stream_name] = asyncio.create_task(self._watch(stream_name))

    async def stop(self) -> None:
        """Stop every watch task cleanly. Cancellation-safe by construction:
        stop() runs in the SSE generator's `finally`, where a real client
        disconnect can deliver CancelledError at any suspension point -- so
        everything that matters (detaching state, stopping subscriptions,
        cancelling tasks) happens synchronously BEFORE the first await. A
        stop interrupted mid-await then leaves the watcher restartable
        (dicts already cleared, tasks already cancelled; ensure_started's
        liveness prune covers any residue) instead of bricked with stale,
        dead tasks that make ensure_started no-op forever.

        Each task spends most of its life blocked inside
        `asyncio.to_thread(next, iterator, ...)`, parked on the esdbclient
        subscription's blocking network read. Cancelling the asyncio.Task
        alone does not unblock that worker thread -- the cancellation is
        only delivered the next time the coroutine resumes on the event
        loop, which won't happen until the blocking next() call itself
        returns. So, mirroring SubscriptionManager.stop(), call `.stop()` on
        each active esdbclient subscription object first: that cancels its
        underlying gRPC stream, which makes the blocked next() call return
        (and the pending task cancellation get delivered) promptly.
        """
        async with self._lock:
            tasks = list(self._tasks.values())
            subscriptions = list(self._active_subscriptions.values())
            # Detach state before any await (see docstring): an interrupted
            # stop must never leave stale entries for ensure_started to
            # mistake for a live watcher.
            self._tasks.clear()
            self._active_subscriptions.clear()

            for subscription in subscriptions:
                # Per-item guard: one subscription raising from .stop() must
                # not abort shutdown and leave the rest unstopped / the
                # tasks below uncancelled.
                try:
                    subscription.stop()
                except Exception:
                    logger.warning("EsdbWatcher: error stopping a subscription during shutdown", exc_info=True)

            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    if not task.done():
                        # Our own cancellation interrupted the wait (the
                        # awaited task hasn't finished yet) -- propagate it.
                        # State is already consistent, and the task will
                        # still finish on the loop: it was cancelled above.
                        raise

    async def _watch(self, stream_name: str) -> None:
        """One category stream's subscribe/consume/reconnect loop.

        On any subscription exception: close the iterator (finally, below),
        back off through `self._backoff_seconds` (repeating the last value
        for subsequent failures), resubscribe from_end=True, and only once
        that resubscribe succeeds, `hub.broadcast_resync()` -- clients then
        refetch whatever they missed while this stream's watcher was down.
        """
        attempt = 0
        needs_resync = False
        while True:
            try:
                async with self._event_store.get_client() as client:
                    subscription = client.subscribe_to_stream(stream_name, from_end=True, resolve_links=True)
                    self._active_subscriptions[stream_name] = subscription

                    if needs_resync:
                        self._hub.broadcast_resync()
                        needs_resync = False
                    attempt = 0

                    iterator = iter(subscription)
                    while True:
                        event = await asyncio.to_thread(next, iterator, _SUBSCRIPTION_ENDED)
                        if event is _SUBSCRIPTION_ENDED:
                            break
                        self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("EsdbWatcher: subscription to '%s' failed", stream_name, exc_info=True)
                needs_resync = True
                delay = self._backoff_seconds[min(attempt, len(self._backoff_seconds) - 1)]
                attempt += 1
                await asyncio.sleep(delay)
            finally:
                # Stale finallys must not evict a successor's live subscription.
                if self._active_subscriptions.get(stream_name) is subscription:
                    self._active_subscriptions.pop(stream_name, None)
                    subscription.stop()

    def _handle_event(self, event: Any) -> None:
        """Decode one resolved event and publish its mapped notifications.
        Malformed `event.data` (unparseable JSON, or valid JSON that isn't
        the dict-shaped payload scope_notifications expects, e.g. a bare
        list or number) is logged at debug and skipped -- the loop keeps
        running rather than killing the whole subscription over one bad
        payload. Decoding and mapping are wrapped in the same try so a
        shape surprise can't escape as an uncaught exception and get
        mistaken for a subscription failure (which would trigger an
        unwarranted backoff/resubscribe/resync)."""
        try:
            payload = json.loads(event.data)
            # Use the RESOLVED event's stream name (link-resolved semantics,
            # per resolve_links=True) -- see scope_notifications' docstring
            # for why this is the mapping's only input besides the payload.
            notifications = scope_notifications(event.stream_name, payload)
        except Exception:
            logger.debug("EsdbWatcher: skipping malformed event on stream '%s'", event.stream_name)
            return

        for notification in notifications:
            self._hub.publish(notification)


_hub: Optional[NotificationHub] = None
_watcher: Optional[EsdbWatcher] = None


def get_live_feed() -> Tuple[NotificationHub, EsdbWatcher]:
    """Module-level lazy singleton pair: one NotificationHub and one
    EsdbWatcher per process, created on first access. The SSE route's only
    coupling point to this module's process-wide state."""
    global _hub, _watcher
    if _hub is None:
        _hub = NotificationHub()
    if _watcher is None:
        _watcher = EsdbWatcher(_hub)
    return _hub, _watcher
