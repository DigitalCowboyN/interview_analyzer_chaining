from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator, _match_participant

FM_TEXT = """---
title: Q3 Vendor Selection
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
---
Alice: We will go with vendor X for the telemetry pipeline work.
Bob Reyes: Sounds good to me, let us proceed with it.
"""


def test_match_participant_full_first_and_ambiguous():
    participants = ["Alice Johnson", "Bob Reyes", "Alice Smith"]
    assert _match_participant("Bob Reyes", participants) == "Bob Reyes"
    assert _match_participant("bob", participants) == "Bob Reyes"      # unique first name
    assert _match_participant("Alice", participants) is None           # ambiguous
    assert _match_participant("Carol", participants) is None           # unlisted


@pytest.mark.asyncio
async def test_labeled_ingest_seeds_confirmed_speakers(tmp_path):
    input_file = tmp_path / "m.txt"
    input_file.write_text(FM_TEXT)
    saved = {}

    async def fake_save(agg, **k):
        saved["interview"] = agg
        agg.mark_events_as_committed()

    interview_repo = MagicMock(save=AsyncMock(side_effect=fake_save))
    sentence_repo = MagicMock(save=AsyncMock())
    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.ingestion.orchestrator.get_sentence_repository", return_value=sentence_repo):
        orch = IngestionOrchestrator(project_id="p", map_dir=tmp_path / "maps")
        await orch.ingest_file(input_file)

    interview = saved["interview"]
    assert interview.title == "Q3 Vendor Selection"
    assert interview.metadata["front_matter"]["participants"] == ["Alice Johnson", "Bob Reyes"]
    by_handle = {info["handle"]: info for info in interview.speakers.values()}
    assert by_handle["Alice"]["display_name"] == "Alice Johnson"   # first-name seed
    assert by_handle["Bob Reyes"]["display_name"] == "Bob Reyes"   # full-name seed
    assert by_handle["Alice"]["provisional"] is False


@pytest.mark.asyncio
async def test_inference_prompt_receives_participants_hint():
    from src.ingestion.models import RawFragment
    from src.ingestion.speaker_inference import SpeakerInferenceService

    service = SpeakerInferenceService()
    captured = {}

    async def fake_call(prompt, schema=None):
        captured["prompt"] = prompt
        return {"assignments": []}

    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(side_effect=fake_call)
        frags = [RawFragment(text="Hello there.", start_char=0, end_char=12, sequence_order=0)]
        await service.infer(frags, participants=["Alice Johnson", "Bob Reyes"])
    assert "Known participants: Alice Johnson, Bob Reyes." in captured["prompt"]
