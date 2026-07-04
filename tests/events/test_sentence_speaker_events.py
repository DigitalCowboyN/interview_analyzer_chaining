import pytest

from src.events.aggregates import Sentence


def make_sentence() -> Sentence:
    s = Sentence("11111111-1111-1111-1111-111111111111")
    s.create(
        interview_id="22222222-2222-2222-2222-222222222222",
        index=0,
        text="Can you hear me?",
        start_char=10,
        end_char=26,
    )
    return s


def test_create_stores_offsets():
    s = make_sentence()
    assert s.start_char == 10
    assert s.end_char == 26
    event = s.get_uncommitted_events()[0]
    assert event.data["start_char"] == 10
    assert event.data["end_char"] == 26


def test_attribute_speaker_sets_state_and_event():
    s = make_sentence()
    event = s.attribute_speaker(
        speaker_id="33333333-3333-3333-3333-333333333333", confidence=0.72, method="inference"
    )
    assert event.event_type == "SpeakerAttributed"
    assert s.speaker_id == "33333333-3333-3333-3333-333333333333"
    assert s.speaker_confidence == 0.72
    assert s.speaker_locked is False


def test_reattribute_speaker_locks_against_system_overwrite():
    s = make_sentence()
    s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.72, "inference")
    event = s.reattribute_speaker("44444444-4444-4444-4444-444444444444")
    assert event.event_type == "SpeakerReattributed"
    assert event.data["old_speaker_id"] == "33333333-3333-3333-3333-333333333333"
    assert s.speaker_locked is True
    with pytest.raises(ValueError, match="locked"):
        s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.9, "inference")


def test_attribute_speaker_requires_created_sentence():
    s = Sentence("11111111-1111-1111-1111-111111111111")
    with pytest.raises(ValueError):
        s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.5, "inference")
