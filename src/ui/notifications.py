"""UI notifications (M5.1): pub/sub core for the SSE bridge.

Pure in-process building blocks — no ESDB, no FastAPI, no framework
coupling — mirroring the reader idiom (plain classes/functions, unit
testable in isolation). `scope_notifications` is the single translation
point from core-domain streams (Sentence-*, Interview-*, Project-*) to the
thin surface-tag contract the browser sees: `{surface, interview_id?,
project_id?}`. The browser must never learn about event types, stream
names, or ESDB concepts — everything upstream of this module speaks
core-domain, everything downstream speaks only surface tags.

Task 4 (ESDB watcher) reads from this module and calls `hub.publish(...)`
for each event it observes, via `scope_notifications`; the watcher itself
does not live here yet — this module stays pure Python until then.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

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
        # so this notification never depends on payload contents.
        interview_id = stream_name[len("Interview-") :]
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


# --- Task 4 adds the ESDB watcher below this line (subscribes to $all,
# translates events via scope_notifications, publishes to a module-level
# NotificationHub instance). Nothing ESDB-related lives above it. ---
