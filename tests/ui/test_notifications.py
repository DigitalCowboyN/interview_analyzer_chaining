"""NotificationHub + scope mapping (M5.1 Task 3): pure in-process pub/sub core
for the SSE bridge. `scope_notifications` is the single translation point from
core-domain streams to the thin surface-tag contract the browser sees — no
event types, stream names, or ESDB concepts leak past it.
"""

import pytest

from src.ui.notifications import Notification, NotificationHub, scope_notifications

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
