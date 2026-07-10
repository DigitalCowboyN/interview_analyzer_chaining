import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_interview():
    i = Interview(IID)
    i.create(title="t", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    i.apply_lens("meeting_minutes", 1)
    return i


def record(i, item_id=ITEM, **over):
    kwargs = dict(
        lens="meeting_minutes",
        lens_version=1,
        node_type="Decision",
        item_id=item_id,
        fields={"text": "Go with X", "made_by": "S1"},
        supporting_fragment_ids=["77777777-7777-7777-7777-777777777771"],
        speaker_links=[{"relationship": "DECIDED_BY", "speaker_id": SP1}],
        confidence=0.9,
        model="haiku",
        provider="anthropic",
    )
    kwargs.update(over)
    return i.record_lens_extraction(**kwargs)


def test_apply_lens_records_run():
    i = make_interview()
    assert i.lens_runs == {"meeting_minutes": 1}


def test_apply_lens_rejects_version_downgrade():
    i = make_interview()
    i.apply_lens("meeting_minutes", 3)
    with pytest.raises(ValueError, match="version"):
        i.apply_lens("meeting_minutes", 2)


def test_record_requires_matching_lens_run():
    i = Interview(IID)
    i.create(title="t", source="s")
    with pytest.raises(ValueError, match="LensApplied"):
        record(i)


def test_record_and_replay():
    i = make_interview()
    event = record(i)
    assert event.event_type == "LensExtractionGenerated"
    assert i.lens_items[ITEM]["node_type"] == "Decision"
    replayed = Interview(IID)
    replayed.load_from_history(i.get_uncommitted_events())
    assert replayed.lens_items[ITEM]["lens_version"] == 1
    assert replayed.lens_runs == {"meeting_minutes": 1}


def test_duplicate_item_raises():
    i = make_interview()
    record(i)
    with pytest.raises(ValueError, match="already recorded"):
        record(i)


def test_override_locks_item_against_rerecord():
    i = make_interview()
    record(i)
    event = i.override_lens_extraction(ITEM, {"text": "Go with Y"}, note="fixed wording")
    assert event.event_type == "LensExtractionOverridden"
    assert i.lens_items[ITEM]["locked"] is True
    i.apply_lens("meeting_minutes", 2)
    with pytest.raises(ValueError, match="locked"):
        record(i, lens_version=2)


def test_override_unknown_item_raises():
    i = make_interview()
    with pytest.raises(ValueError, match="Unknown lens item"):
        i.override_lens_extraction("no-such", {"text": "x"})
