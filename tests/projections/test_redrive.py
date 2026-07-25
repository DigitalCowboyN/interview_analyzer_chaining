"""tests/projections/test_redrive.py

Unit tests for the parked-event redrive CLI (src/projections/redrive.py).

M4.9 Task 2 made parking append-only; M4.9 Task 4 added a reorder buffer so
parking should now be rare. This module is the manual recovery path: replay
parked events through the real handler registry once their referents have
landed. Handlers MERGE (idempotent), so:
  - a parked event whose referent now exists applies cleanly -> "redriven"
  - a parked event whose referent is still missing raises
    ReferentNotReadyError again -> "still_parked" (run must NOT crash)
  - a parked event type with no registered handler -> "no_handler"
  - redriving an already-applied event a second time is a safe no-op that
    still counts as "redriven" (MERGE idempotency)

Fakes (no live EventStoreDB/Neo4j): FakeParkedEventsManager returns canned
ParkedEvent lists per aggregate type; FakeRegistry/FakeHandler script the
per-event-type handler outcome.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from src.events.envelope import Actor, ActorType, EventEnvelope
from src.projections.handlers.speaker_handlers import ReferentNotReadyError
from src.projections.parked_events import ParkedEvent
from src.projections.redrive import (
    KNOWN_AGGREGATE_TYPES,
    _resolve_aggregate_types,
    redrive,
    redrive_aggregate,
)


def _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence") -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=str(uuid.uuid4()),
        version=1,
        data={"test": "data"},
        actor=Actor(actor_type=ActorType.SYSTEM),
        correlation_id=str(uuid.uuid4()),
    )


def _parked(event: EventEnvelope) -> ParkedEvent:
    return ParkedEvent(
        original_event=event,
        error_message="referent not ready",
        error_type="ReferentNotReadyError",
        retry_count=5,
        parked_at=datetime.now(timezone.utc),
        lane_id=0,
    )


class FakeParkedEventsManager:
    """Canned parked-event lists per aggregate type, mirroring the real
    ParkedEventsManager.get_parked_events(aggregate_type, max_count) signature."""

    def __init__(self, events_by_aggregate):
        self._events_by_aggregate = events_by_aggregate

    async def get_parked_events(self, aggregate_type, max_count=None):
        events = self._events_by_aggregate.get(aggregate_type, [])
        if max_count is not None:
            return events[:max_count]
        return list(events)


class FakeHandler:
    """Handler stub whose .handle() outcome is scripted per call (in order;
    the last scripted outcome repeats once exhausted)."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def handle(self, event):
        self.calls += 1
        outcome = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        if outcome == "ok":
            return
        raise outcome


class FakeRegistry:
    """Mirrors HandlerRegistry.get_handler(event_type) -> Optional[handler]."""

    def __init__(self, handlers_by_event_type):
        self._handlers = handlers_by_event_type

    def get_handler(self, event_type):
        return self._handlers.get(event_type)


@pytest.mark.asyncio
async def test_redrive_applies_event_whose_referent_now_exists():
    event = _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence")
    manager = FakeParkedEventsManager({"Sentence": [_parked(event)]})
    handler = FakeHandler(["ok"])
    registry = FakeRegistry({"SpeakerAttributed": handler})

    counts = await redrive_aggregate("Sentence", manager, registry)

    assert counts == {"redriven": 1, "still_parked": 0, "no_handler": 0, "failed": 0}
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_redrive_reports_still_parked_on_referent_not_ready_and_does_not_crash():
    event = _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence")
    manager = FakeParkedEventsManager({"Sentence": [_parked(event)]})
    handler = FakeHandler([ReferentNotReadyError("referent still missing")])
    registry = FakeRegistry({"SpeakerAttributed": handler})

    counts = await redrive_aggregate("Sentence", manager, registry)

    assert counts == {"redriven": 0, "still_parked": 1, "no_handler": 0, "failed": 0}


@pytest.mark.asyncio
async def test_redrive_counts_unexpected_handler_error_as_failed_and_continues():
    """A parked event whose handler raises a non-ReferentNotReadyError error
    is counted as `failed` and does NOT abort the rest of the run (final
    review Minor #4)."""
    bad_event = _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence")
    good_event = _envelope(event_type="SpeakerReattributed", aggregate_type="Sentence")
    manager = FakeParkedEventsManager({"Sentence": [_parked(bad_event), _parked(good_event)]})
    registry = FakeRegistry(
        {
            "SpeakerAttributed": FakeHandler([RuntimeError("boom")]),
            "SpeakerReattributed": FakeHandler(["ok"]),
        }
    )

    counts = await redrive_aggregate("Sentence", manager, registry)

    # The failing event is counted, and the following event still redrove.
    assert counts == {"redriven": 1, "still_parked": 0, "no_handler": 0, "failed": 1}


@pytest.mark.asyncio
async def test_redrive_reports_no_handler_when_none_registered():
    event = _envelope(event_type="SomeRetiredEvent", aggregate_type="Interview")
    manager = FakeParkedEventsManager({"Interview": [_parked(event)]})
    registry = FakeRegistry({})  # nothing registered for this event type

    counts = await redrive_aggregate("Interview", manager, registry)

    assert counts == {"redriven": 0, "still_parked": 0, "no_handler": 1, "failed": 0}


@pytest.mark.asyncio
async def test_redrive_is_idempotent_on_already_applied_event():
    """MERGE semantics: redriving an already-applied event a second time is
    a safe no-op that still counts as redriven (no exception, no double-count
    weirdness)."""
    event = _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence")
    manager = FakeParkedEventsManager({"Sentence": [_parked(event)]})
    handler = FakeHandler(["ok", "ok"])
    registry = FakeRegistry({"SpeakerAttributed": handler})

    first = await redrive_aggregate("Sentence", manager, registry)
    second = await redrive_aggregate("Sentence", manager, registry)

    assert first == {"redriven": 1, "still_parked": 0, "no_handler": 0, "failed": 0}
    assert second == {"redriven": 1, "still_parked": 0, "no_handler": 0, "failed": 0}
    assert handler.calls == 2


@pytest.mark.asyncio
async def test_redrive_summarizes_across_aggregate_types_with_correct_json_shape():
    ok_event = _envelope(event_type="SpeakerAttributed", aggregate_type="Sentence")
    stuck_event = _envelope(event_type="SpeakerReattributed", aggregate_type="Sentence")
    orphan_event = _envelope(event_type="LegacyEvent", aggregate_type="Interview")

    manager = FakeParkedEventsManager(
        {
            "Sentence": [_parked(ok_event), _parked(stuck_event)],
            "Interview": [_parked(orphan_event)],
        }
    )
    registry = FakeRegistry(
        {
            "SpeakerAttributed": FakeHandler(["ok"]),
            "SpeakerReattributed": FakeHandler([ReferentNotReadyError("nope")]),
        }
    )

    summary = await redrive(["Sentence", "Interview"], manager, registry)

    assert summary["redriven"] == 1
    assert summary["still_parked"] == 1
    assert summary["no_handler"] == 1
    assert summary["by_aggregate"] == {
        "Sentence": {"redriven": 1, "still_parked": 1, "no_handler": 0, "failed": 0},
        "Interview": {"redriven": 0, "still_parked": 0, "no_handler": 1, "failed": 0},
    }

    # Must be dumpable as a single JSON line (seed_smoke.py / migrate_shim_drop.py convention).
    line = json.dumps(summary)
    assert "\n" not in line
    assert json.loads(line) == summary


def test_resolve_aggregate_types_defaults_to_known_types_for_all_or_empty():
    assert _resolve_aggregate_types(["all"]) == KNOWN_AGGREGATE_TYPES
    assert _resolve_aggregate_types([]) == KNOWN_AGGREGATE_TYPES


def test_resolve_aggregate_types_passes_through_explicit_selection():
    assert _resolve_aggregate_types(["Interview"]) == ["Interview"]
    assert _resolve_aggregate_types(["Interview", "Project"]) == ["Interview", "Project"]
