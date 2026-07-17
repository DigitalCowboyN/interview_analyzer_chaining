"""
Shim-drop test for the Sentence -> Fragment code-surface rename (M4.8).

Wire format is frozen: AggregateType.SENTENCE, the "Sentence" value, and
event type strings (SentenceCreated, etc.) do not change. The deprecated
code-surface aliases (Sentence, SentenceRepository, get_sentence_repository,
create_sentence_repository) are gone; Fragment/FragmentRepository/
get_fragment_repository/create_fragment_repository are the only names.
"""

import src.events.aggregates
import src.events.repository


def test_deprecated_aliases_are_gone():
    assert not hasattr(src.events.aggregates, "Sentence")
    assert not hasattr(src.events.repository, "SentenceRepository")
    assert not hasattr(src.events.repository, "get_sentence_repository")

    factory = src.events.repository.RepositoryFactory
    assert not hasattr(factory, "create_sentence_repository")


def test_new_names_are_importable():
    from src.events.aggregates import Fragment
    from src.events.repository import FragmentRepository, get_fragment_repository

    assert Fragment is not None
    assert FragmentRepository is not None
    assert get_fragment_repository is not None


def test_wire_format_frozen():
    from src.events.aggregates import Fragment

    # wire format frozen: the aggregate still stamps "Sentence"
    f = Fragment("77777777-7777-7777-7777-777777777771")
    f.create(interview_id="22222222-2222-2222-2222-222222222222", index=0, text="Hi.")
    event = f.get_uncommitted_events()[0]
    assert event.aggregate_type == "Sentence"
    assert event.event_type == "SentenceCreated"
