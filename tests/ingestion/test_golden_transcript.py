"""Golden crosstalk transcript regression test.

The fixture is a verbatim excerpt of a real messy Zoom transcript
(data/input/GMT20231026-210203_Recording.txt): one continuous string, no
speaker labels, rapid overlapping exchanges. The LLM window response is
recorded in the expected-JSON (hand-verified), so this test is deterministic
and runs in CI without API keys while exercising the full Layer 1 flow.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


@pytest.mark.asyncio
async def test_crosstalk_excerpt_end_to_end(tmp_path: Path):
    source = (GOLDEN_DIR / "crosstalk_excerpt.txt").read_text()
    expected_doc = json.loads((GOLDEN_DIR / "crosstalk_expected.json").read_text())
    exp = expected_doc["expected"]

    input_file = tmp_path / "crosstalk.txt"
    input_file.write_text(source)

    async def fake_save(aggregate, expected_version=None):
        aggregate.mark_events_as_committed()

    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=fake_save)

    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=mock_repo), \
         patch("src.ingestion.orchestrator.get_sentence_repository", return_value=mock_repo), \
         patch("src.ingestion.speaker_inference.agent") as sp_agent, \
         patch("src.ingestion.stitcher.agent") as st_agent:
        sp_agent.call_model = AsyncMock(return_value=expected_doc["speaker_window_response"])
        st_agent.call_model = AsyncMock(return_value=expected_doc["stitch_window_response"])
        orchestrator = IngestionOrchestrator(project_id="golden", map_dir=tmp_path / "maps")
        result = await orchestrator.ingest_file(input_file)

    assert result.fragment_count == exp["fragment_count"]
    assert result.speaker_count == exp["speaker_count"]
    assert result.utterance_count == exp["utterance_count"]
    assert result.utterance_count >= exp["min_utterances"]
    assert result.low_confidence_count == exp["low_confidence_count"]

    # The map grounds every fragment: offsets must recover the exact text.
    map_file = tmp_path / "maps" / "crosstalk_map.jsonl"
    entries = [json.loads(line) for line in map_file.read_text().splitlines()]
    assert len(entries) == result.fragment_count
    for entry in entries:
        assert source[entry["start_char"]:entry["end_char"]] == entry["sentence"]
        if exp["every_fragment_attributed"]:
            assert entry["speaker_id"] is not None
            assert entry["utterance_id"] is not None
        assert 0.0 <= entry["speaker_confidence"] <= 1.0
