import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import SpecOutcome
from src.events.aggregates import Interview, Sentence
from src.lens.engine import LensEngine

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"


def build_world():
    interview = Interview(IID)
    interview.create(title="t", source="s", metadata={"fragment_count": 2})
    interview.add_speaker(SP1, "S1", "Alice", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(2)]
    sentences = {}
    for i, fid in enumerate(f_ids):
        s = Sentence(fid)
        s.create(interview_id=IID, index=i, text=f"Fragment {i}.")
        s.attribute_speaker(SP1, 0.9, "inference")
        s.mark_events_as_committed()
        sentences[fid] = s
    interview.identify_utterance(U1, SP1, f_ids, 0.9)
    interview.mark_events_as_committed()
    return interview, sentences


def outcome_for(spec_name):
    canned = {
        "objectives": {"objectives": [{"text": "Ship the ECU tool", "confidence": 0.9}]},
        "decisions": {"decisions": [{"text": "Go with X", "made_by": "Alice", "confidence": 0.9}]},
        "action_items": {
            "action_items": [{"text": "Draft the doc", "owner": "SELF", "due": None, "confidence": 0.8}]
        },
        "followups": {"followups": []},
    }
    return SpecOutcome(data=canned[spec_name], provider="anthropic", model="haiku")


def patch_engine(interview, sentences):
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    executor = MagicMock()
    executor.run_spec_on_text = AsyncMock(side_effect=lambda spec, text, ctx=None: outcome_for(spec.name))
    return (
        patch("src.lens.engine.get_interview_repository", return_value=interview_repo),
        patch("src.lens.engine.get_sentence_repository", return_value=sentence_repo),
        patch.object(LensEngine, "_build_executor", return_value=executor),
        executor,
    )


@pytest.mark.asyncio
async def test_apply_extracts_items_with_links_and_grounding():
    interview, sentences = build_world()
    p1, p2, p3, executor = patch_engine(interview, sentences)
    with p1, p2, p3:
        result = await LensEngine().apply(IID, "meeting_minutes")

    assert result.items_extracted == 3  # objective + decision + action item
    assert result.items_skipped_existing == 0
    assert interview.lens_runs == {"meeting_minutes": 1}
    decision = next(v for v in interview.lens_items.values() if v["node_type"] == "Decision")
    assert decision["lens"] == "meeting_minutes"
    types = {v["node_type"] for v in interview.lens_items.values()}
    assert types == {"Objective", "Decision", "ActionItem"}  # empty followups -> no item


@pytest.mark.asyncio
async def test_owner_self_resolves_to_utterance_speaker_and_alice_by_display_name():
    interview, sentences = build_world()
    captured = []
    original = interview.record_lens_extraction

    def spy(*a, **k):
        captured.append(k)
        return original(*a, **k)

    interview.record_lens_extraction = spy
    p1, p2, p3, _ = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")

    by_type = {k["node_type"]: k for k in captured}
    assert by_type["Decision"]["speaker_links"] == [
        {"relationship": "DECIDED_BY", "speaker_id": SP1}  # "Alice" display-name match
    ]
    assert by_type["ActionItem"]["speaker_links"] == [
        {"relationship": "OWNED_BY", "speaker_id": SP1}  # SELF -> utterance speaker
    ]
    assert by_type["Decision"]["supporting_fragment_ids"]  # utterance fragments
    assert by_type["Objective"]["supporting_fragment_ids"] == []  # document scope


@pytest.mark.asyncio
async def test_second_run_same_version_is_idempotent():
    interview, sentences = build_world()
    p1, p2, p3, executor = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")
        result2 = await LensEngine().apply(IID, "meeting_minutes")

    assert result2.items_extracted == 0
    assert result2.items_skipped_existing == 3
    assert len([i for i in interview.lens_items]) == 3  # no duplicates


@pytest.mark.asyncio
async def test_locked_items_survive_forced_rerun():
    interview, sentences = build_world()
    p1, p2, p3, _ = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")
        locked_id = next(
            iid for iid, v in interview.lens_items.items() if v["node_type"] == "Decision"
        )
        interview.override_lens_extraction(locked_id, {"text": "Go with Y"})
        interview.mark_events_as_committed()
        result = await LensEngine().apply(IID, "meeting_minutes", force=True)

    assert result.items_skipped_locked >= 1
    assert interview.lens_items[locked_id]["locked"] is True
