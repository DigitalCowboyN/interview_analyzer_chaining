import pytest

from src.events.aggregates import Interview, Fragment

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
F1 = "77777777-7777-7777-7777-777777777771"


def test_fragment_embedding_event_carries_lane_key():
    s = Fragment(F1)
    s.create(interview_id=IID, index=0, text="Hi.")
    event = s.record_embedding(model="text-embedding-3-small", dim=3, vector_b64="AAAA")
    assert event.event_type == "EmbeddingGenerated"
    assert event.data["interview_id"] == IID
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 3


def test_fragment_embedding_requires_created():
    s = Fragment(F1)
    with pytest.raises(ValueError, match="created"):
        s.record_embedding(model="m", dim=3, vector_b64="AAAA")


def test_utterance_embedding_guards_unknown_utterance():
    i = Interview(IID)
    i.create(title="t", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.identify_utterance(U1, SP1, [F1], 0.9)
    event = i.record_utterance_embedding(U1, model="m", dim=3, vector_b64="AAAA")
    assert event.event_type == "UtteranceEmbeddingGenerated"
    assert i.utterance_embeddings[U1] == {"model": "m", "dim": 3}
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_utterance_embedding("no-such", model="m", dim=3, vector_b64="AAAA")


def test_embedding_replay_reconstructs():
    s = Fragment(F1)
    s.create(interview_id=IID, index=0, text="Hi.")
    s.record_embedding(model="text-embedding-3-small", dim=3, vector_b64="AAAA")
    replayed = Fragment(F1)
    replayed.load_from_history(s.get_uncommitted_events())
    assert replayed.embedding_model == "text-embedding-3-small"
    assert replayed.embedding_dim == 3
