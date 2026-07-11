"""
Alias test for the Sentence -> Fragment code-surface rename (Task 8).

Wire format is frozen: AggregateType.SENTENCE, the "Sentence" value, and
event type strings (SentenceCreated, etc.) do not change. Only the code
surface (class/function names) is renamed, with deprecated aliases kept
for backward compatibility.
"""


def test_fragment_rename_aliases():
    from src.events.aggregates import Fragment, Sentence
    from src.events.repository import get_fragment_repository, get_sentence_repository

    assert Sentence is Fragment
    assert get_sentence_repository is get_fragment_repository
    # wire format frozen: the aggregate still stamps "Sentence"
    f = Fragment("77777777-7777-7777-7777-777777777771")
    f.create(interview_id="22222222-2222-2222-2222-222222222222", index=0, text="Hi.")
    event = f.get_uncommitted_events()[0]
    assert event.aggregate_type == "Sentence"
    assert event.event_type == "SentenceCreated"
