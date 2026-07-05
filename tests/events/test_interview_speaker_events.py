import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"


def make_interview() -> Interview:
    interview = Interview(IID)
    interview.create(title="test.txt", source="data/input/test.txt")
    return interview


def test_add_speaker_records_provisional_speaker():
    i = make_interview()
    event = i.add_speaker(SP1, handle="S1", display_name="S1", provisional=True,
                          confidence=0.8, method="inference")
    assert event.event_type == "SpeakerCreated"
    assert i.speakers[SP1]["handle"] == "S1"
    assert i.speakers[SP1]["provisional"] is True


def test_duplicate_speaker_id_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    with pytest.raises(ValueError, match="already exists"):
        i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")


def test_rename_speaker_clears_provisional_flag():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    event = i.rename_speaker(SP1, "Dana the ECU owner")
    assert event.event_type == "SpeakerRenamed"
    assert i.speakers[SP1]["display_name"] == "Dana the ECU owner"
    assert i.speakers[SP1]["provisional"] is False


def test_merge_speakers_marks_merged_into():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.add_speaker(SP2, "S2", "S2", True, 0.6, "inference")
    event = i.merge_speakers(surviving_speaker_id=SP1, merged_speaker_id=SP2)
    assert event.event_type == "SpeakerMerged"
    assert i.speakers[SP2]["merged_into"] == SP1


def test_merge_into_self_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    with pytest.raises(ValueError, match="itself"):
        i.merge_speakers(SP1, SP1)


def test_rename_unknown_speaker_rejected():
    i = make_interview()
    with pytest.raises(ValueError, match="Unknown speaker"):
        i.rename_speaker(SP1, "Nobody")


def test_rename_already_merged_speaker_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.add_speaker(SP2, "S2", "S2", True, 0.6, "inference")
    i.merge_speakers(surviving_speaker_id=SP1, merged_speaker_id=SP2)
    with pytest.raises(ValueError, match="already been merged"):
        i.rename_speaker(SP2, "Retired Speaker")


def test_merge_with_already_merged_speaker_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.add_speaker(SP2, "S2", "S2", True, 0.6, "inference")
    sp3 = "55555555-5555-5555-5555-555555555550"
    i.add_speaker(sp3, "S3", "S3", True, 0.7, "inference")
    i.merge_speakers(surviving_speaker_id=SP1, merged_speaker_id=SP2)
    with pytest.raises(ValueError, match="already been merged"):
        i.merge_speakers(surviving_speaker_id=SP2, merged_speaker_id=sp3)
