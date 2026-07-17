import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator

LABELED = """Alice: Hi, thanks for joining today.
Bob: Happy to be here.
Alice: Let's get started.
"""


@pytest.mark.asyncio
async def test_labeled_ingestion_emits_events_and_writes_map(tmp_path: Path):
    input_file = tmp_path / "meeting.txt"
    input_file.write_text(LABELED)
    map_dir = tmp_path / "maps"

    saved_aggregates = []

    async def capture_save(aggregate, expected_version=None):
        saved_aggregates.append(aggregate)
        aggregate.mark_events_as_committed()

    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=capture_save)

    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=mock_repo), \
         patch("src.ingestion.orchestrator.get_fragment_repository", return_value=mock_repo):
        orchestrator = IngestionOrchestrator(project_id="proj-1", map_dir=map_dir)
        result = await orchestrator.ingest_file(input_file)

    assert result.fragment_count == 3
    assert result.speaker_count == 2
    assert result.utterance_count == 3  # baseline: three speaker turns
    assert result.interruption_count == 0
    assert result.low_confidence_count == 0  # parsed labels are confidence 1.0

    map_file = map_dir / "meeting_map.jsonl"
    assert map_file.exists()
    entries = [json.loads(line) for line in map_file.read_text().splitlines()]
    assert len(entries) == 3
    for entry in entries:
        assert LABELED[entry["start_char"]:entry["end_char"]] == entry["sentence"]
        assert entry["speaker_id"] is not None
        assert entry["utterance_id"] is not None
        assert 0.0 <= entry["speaker_confidence"] <= 1.0


@pytest.mark.asyncio
async def test_flat_ingestion_uses_inference_and_counts_low_confidence(tmp_path: Path):
    flat = "Well, hey, how are you doing? Are you able to hear me? Yep."
    input_file = tmp_path / "raw.txt"
    input_file.write_text(flat)

    async def capture_save(aggregate, expected_version=None):
        aggregate.mark_events_as_committed()

    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=capture_save)

    inference_response = {
        "assignments": [
            {"index": 0, "speaker": "S1", "confidence": 0.9},
            {"index": 1, "speaker": "S1", "confidence": 0.5},
            {"index": 2, "speaker": "S2", "confidence": 0.9},
        ]
    }
    stitch_response = {"utterances": [], "interruptions": []}

    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=mock_repo), \
         patch("src.ingestion.orchestrator.get_fragment_repository", return_value=mock_repo), \
         patch("src.ingestion.speaker_inference.agent") as sp_agent, \
         patch("src.ingestion.stitcher.agent") as st_agent:
        sp_agent.call_model = AsyncMock(return_value=inference_response)
        st_agent.call_model = AsyncMock(return_value=stitch_response)
        orchestrator = IngestionOrchestrator(project_id="proj-1", map_dir=tmp_path / "maps")
        result = await orchestrator.ingest_file(input_file)

    assert result.fragment_count == 3
    assert result.speaker_count == 2
    assert result.utterance_count == 2  # baseline: S1,S1 then S2
    assert result.low_confidence_count == 1  # the 0.5 assignment
    assert sp_agent.call_model.await_count == 1
    assert st_agent.call_model.await_count == 1

    map_file = tmp_path / "maps" / "raw_map.jsonl"
    assert map_file.exists()
    entries = [json.loads(line) for line in map_file.read_text().splitlines()]
    assert len(entries) == 3
    for entry in entries:
        assert flat[entry["start_char"]:entry["end_char"]] == entry["sentence"]
        assert entry["speaker_id"] is not None
        assert entry["utterance_id"] is not None
