import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
U2 = "66666666-6666-6666-6666-666666666666"
FRAGS = ["77777777-7777-7777-7777-777777777771", "77777777-7777-7777-7777-777777777772"]


def make_interview_with_speaker() -> Interview:
    i = Interview(IID)
    i.create(title="test.txt", source="data/input/test.txt")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    return i


def test_identify_utterance_records_fragments_in_order():
    i = make_interview_with_speaker()
    event = i.identify_utterance(U1, SP1, FRAGS, confidence=0.75)
    assert event.event_type == "UtteranceIdentified"
    assert i.utterances[U1]["fragment_ids"] == FRAGS


def test_identify_utterance_requires_known_speaker():
    i = make_interview_with_speaker()
    with pytest.raises(ValueError, match="Unknown speaker"):
        i.identify_utterance(U1, "99999999-9999-9999-9999-999999999999", FRAGS, 0.5)


def test_identify_utterance_requires_fragments():
    i = make_interview_with_speaker()
    with pytest.raises(ValueError, match="at least one fragment"):
        i.identify_utterance(U1, SP1, [], 0.5)


def test_record_interruption_requires_both_utterances():
    i = make_interview_with_speaker()
    i.identify_utterance(U1, SP1, FRAGS, 0.75)
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_interruption(U2, U1, FRAGS[0])


def test_record_interruption_and_remove_stitch():
    i = make_interview_with_speaker()
    i.identify_utterance(U1, SP1, [FRAGS[0]], 0.75)
    i.identify_utterance(U2, SP1, [FRAGS[1]], 0.6)
    event = i.record_interruption(U2, U1, FRAGS[1])
    assert event.event_type == "InterruptionRecorded"
    removed = i.remove_stitch(U2, reason="not actually a continuation")
    assert removed.event_type == "StitchRemoved"
    assert i.utterances[U2]["removed"] is True


def test_replay_reconstructs_utterance_state():
    source = make_interview_with_speaker()
    source.identify_utterance(U1, SP1, FRAGS, 0.75)
    source.remove_stitch(U1, reason="wrong")
    history = source.get_uncommitted_events()

    replayed = Interview(IID)
    replayed.load_from_history(history)

    assert replayed.utterances[U1]["fragment_ids"] == FRAGS
    assert replayed.utterances[U1]["removed"] is True
    assert replayed.speakers[SP1]["handle"] == "S1"
    assert replayed.version == source.version


def test_remove_stitch_twice_raises():
    i = make_interview_with_speaker()
    i.identify_utterance(U1, SP1, FRAGS, 0.75)
    i.remove_stitch(U1, reason="wrong")
    with pytest.raises(ValueError, match="already removed"):
        i.remove_stitch(U1, reason="second attempt")
