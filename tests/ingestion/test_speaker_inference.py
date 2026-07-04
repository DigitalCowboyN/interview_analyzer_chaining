from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import SpeakerInferenceService


def frags(n: int):
    return [
        RawFragment(text=f"Fragment {i}.", start_char=i * 20, end_char=i * 20 + 11, sequence_order=i)
        for i in range(n)
    ]


def window_response(indices, speakers, confidence=0.9):
    return {
        "assignments": [
            {"index": i, "speaker": s, "confidence": confidence} for i, s in zip(indices, speakers)
        ]
    }


@pytest.mark.asyncio
async def test_single_window_assigns_all_fragments():
    service = SpeakerInferenceService(window_size=10, overlap=2)
    fragments = frags(4)
    mock_response = window_response([0, 1, 2, 3], ["S1", "S2", "S1", "S2"])
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=mock_response)
        result = await service.infer(fragments)
    assert result.handles == ["S1", "S2"]
    assert [a.handle for a in result.assignments] == ["S1", "S2", "S1", "S2"]
    assert all(0.0 <= a.confidence <= 1.0 for a in result.assignments)


@pytest.mark.asyncio
async def test_overlapping_windows_reconcile_handles():
    # Window 1 covers 0-3, window 2 covers 2-5 (overlap=2).
    # Window 2 calls the same people "A" and "B"; overlap voting must map them
    # back to the global S1/S2 handles.
    service = SpeakerInferenceService(window_size=4, overlap=2)
    fragments = frags(6)
    responses = [
        window_response([0, 1, 2, 3], ["S1", "S2", "S1", "S2"]),
        window_response([0, 1, 2, 3], ["A", "B", "A", "B"]),  # local indices of frags 2-5
    ]
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(side_effect=responses)
        result = await service.infer(fragments)
    assert result.handles == ["S1", "S2"]
    assert [a.handle for a in result.assignments] == ["S1", "S2", "S1", "S2", "S1", "S2"]


@pytest.mark.asyncio
async def test_missing_assignment_gets_zero_confidence_unknown():
    # LLM omitted fragment 1; service must still return one assignment per fragment.
    service = SpeakerInferenceService(window_size=10, overlap=2)
    fragments = frags(3)
    mock_response = window_response([0, 2], ["S1", "S1"])
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=mock_response)
        result = await service.infer(fragments)
    assert len(result.assignments) == 3
    assert result.assignments[1].confidence == 0.0


@pytest.mark.asyncio
async def test_invalid_window_response_skipped_not_fatal():
    service = SpeakerInferenceService(window_size=10, overlap=2)
    fragments = frags(2)
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value={"nonsense": True})
        result = await service.infer(fragments)
    assert len(result.assignments) == 2
    assert all(a.confidence == 0.0 for a in result.assignments)


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError, match="overlap"):
        SpeakerInferenceService(window_size=10, overlap=10)
