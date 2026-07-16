"""Segment events on the Interview aggregate: emit, replay, redraw discipline."""

import uuid

import pytest

from src.events.aggregates import Interview
from src.events.envelope import Actor, ActorType
from src.events.interview_events import segment_id_for

ACTOR = Actor(actor_type=ActorType.SYSTEM, user_id="enrichment")


def _interview() -> Interview:
    interview = Interview(str(uuid.uuid4()))
    interview.create(title="t", source="s", language="en", actor=ACTOR)
    return interview


def test_segment_id_for_is_deterministic_uuid5():
    iid = "abc"
    expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "abc:segment:0"))
    assert segment_id_for(iid, 0) == expected
    assert segment_id_for(iid, 1) != expected


def test_record_segment_emits_event_and_updates_state():
    interview = _interview()
    sid = segment_id_for(interview.aggregate_id, 0)
    event = interview.record_segment(sid, "Vendor choice", 0, 2, 0.9, actor=ACTOR)
    assert event.event_type == "SegmentIdentified"
    assert event.data == {
        "segment_id": sid, "topic": "Vendor choice",
        "start_index": 0, "end_index": 2, "confidence": 0.9,
    }
    assert interview.segments[sid] == {
        "topic": "Vendor choice", "start_index": 0, "end_index": 2, "removed": False,
    }


def test_record_segment_guards():
    interview = _interview()
    sid = segment_id_for(interview.aggregate_id, 0)
    with pytest.raises(ValueError):  # inverted range
        interview.record_segment(sid, "t", 3, 1, 0.5, actor=ACTOR)
    interview.record_segment(sid, "t", 0, 1, 0.5, actor=ACTOR)
    with pytest.raises(ValueError):  # live duplicate
        interview.record_segment(sid, "t", 0, 1, 0.5, actor=ACTOR)


def test_remove_then_reidentify_supports_redraw():
    interview = _interview()
    sid = segment_id_for(interview.aggregate_id, 0)
    interview.record_segment(sid, "old", 0, 1, 0.5, actor=ACTOR)
    event = interview.remove_segment(sid, reason="wrong split", actor=ACTOR)
    assert event.event_type == "SegmentRemoved"
    assert event.data == {"segment_id": sid, "reason": "wrong split"}
    assert interview.segments[sid]["removed"] is True
    # redraw: re-identifying a removed segment id is allowed
    interview.record_segment(sid, "new", 0, 2, 0.7, actor=ACTOR)
    assert interview.segments[sid] == {
        "topic": "new", "start_index": 0, "end_index": 2, "removed": False,
    }


def test_remove_segment_guards():
    interview = _interview()
    with pytest.raises(ValueError):  # unknown
        interview.remove_segment("nope", actor=ACTOR)
    sid = segment_id_for(interview.aggregate_id, 0)
    interview.record_segment(sid, "t", 0, 0, 0.5, actor=ACTOR)
    interview.remove_segment(sid, actor=ACTOR)
    with pytest.raises(ValueError):  # already removed
        interview.remove_segment(sid, actor=ACTOR)


def test_replay_rebuilds_segments():
    interview = _interview()
    sid = segment_id_for(interview.aggregate_id, 0)
    interview.record_segment(sid, "t", 0, 1, 0.5, actor=ACTOR)
    interview.remove_segment(sid, actor=ACTOR)
    events = interview.get_uncommitted_events()

    replayed = Interview(interview.aggregate_id)
    replayed.load_from_history(events)
    assert replayed.segments[sid]["removed"] is True
    assert replayed.version == interview.version
