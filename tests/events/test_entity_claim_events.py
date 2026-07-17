import pytest

from src.events.aggregates import Interview, Fragment

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
F1 = "77777777-7777-7777-7777-777777777771"
CLAIM = "88888888-8888-8888-8888-888888888881"

ENTITY = {"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}


def make_sentence():
    s = Fragment(F1)
    s.create(interview_id=IID, index=0, text="Use Neo4j here.")
    return s


def make_interview():
    i = Interview(IID)
    i.create(title="t.txt", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.identify_utterance(U1, SP1, [F1], 0.9)
    return i


def test_record_entities_event_and_state():
    s = make_sentence()
    event = s.record_entities([ENTITY], model="haiku", provider="anthropic")
    assert event.event_type == "EntitiesExtracted"
    assert event.data["interview_id"] == IID  # lane routing key
    assert event.data["entities"] == [ENTITY]
    assert s.entities == [ENTITY]


def test_record_entities_requires_created_sentence():
    s = Fragment(F1)
    with pytest.raises(ValueError, match="created"):
        s.record_entities([ENTITY], model="haiku", provider="anthropic")


def test_record_claim_derives_speaker_and_guards():
    i = make_interview()
    event = i.record_claim(CLAIM, U1, "We will ship Friday", "commitment", 0.8, "haiku", "anthropic")
    assert event.event_type == "ClaimExtracted"
    assert event.data["speaker_id"] == SP1
    assert event.data["kind"] == "commitment"
    with pytest.raises(ValueError, match="already recorded"):
        i.record_claim(CLAIM, U1, "dup", "assertion", 0.5, "haiku", "anthropic")
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_claim(
            "99999999-9999-9999-9999-999999999999", "no-such", "x", "assertion", 0.5, "haiku", "anthropic"
        )


def test_record_claim_rejects_removed_utterance():
    i = make_interview()
    i.remove_stitch(U1, reason="wrong")
    with pytest.raises(ValueError, match="removed"):
        i.record_claim(CLAIM, U1, "x", "assertion", 0.5, "haiku", "anthropic")


def test_claim_replay_reconstructs():
    i = make_interview()
    i.record_claim(CLAIM, U1, "We will ship Friday", "commitment", 0.8, "haiku", "anthropic")
    replayed = Interview(IID)
    replayed.load_from_history(i.get_uncommitted_events())
    assert CLAIM in replayed.claims
    assert replayed.claims[CLAIM]["kind"] == "commitment"


def test_entities_replay_reconstructs():
    s = make_sentence()
    s.record_entities([ENTITY], model="haiku", provider="anthropic")
    replayed = Fragment(F1)
    replayed.load_from_history(s.get_uncommitted_events())
    assert replayed.entities == [ENTITY]
