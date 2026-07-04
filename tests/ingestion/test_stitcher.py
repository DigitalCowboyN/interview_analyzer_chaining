from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import FragmentSpeaker
from src.ingestion.stitcher import Stitcher


def make_inputs(handles):
    fragments = [
        RawFragment(text=f"Fragment {i}.", start_char=i * 20, end_char=i * 20 + 11, sequence_order=i)
        for i in range(len(handles))
    ]
    assignments = [
        FragmentSpeaker(sequence_order=i, handle=h, confidence=0.9) for i, h in enumerate(handles)
    ]
    return fragments, assignments


@pytest.mark.asyncio
async def test_baseline_groups_consecutive_same_speaker():
    fragments, assignments = make_inputs(["S1", "S1", "S2", "S1"])
    stitcher = Stitcher()
    empty = {"utterances": [], "interruptions": []}
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=empty)
        result = await stitcher.stitch(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0, 1], [2], [3]]


@pytest.mark.asyncio
async def test_llm_merge_stitches_interrupted_utterance():
    # S1 speaks (0), S2 interrupts (1), S1 resumes the same thought (2).
    fragments, assignments = make_inputs(["S1", "S2", "S1"])
    llm_response = {
        "utterances": [
            {"speaker": "S1", "fragment_indices": [0, 2], "confidence": 0.8},
            {"speaker": "S2", "fragment_indices": [1], "confidence": 0.9},
        ],
        "interruptions": [{"interrupting": 1, "interrupted": 0, "at_index": 1}],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=llm_response)
        result = await stitcher.stitch(fragments, assignments)
    s1_utterances = [u for u in result.utterances if u.handle == "S1"]
    assert len(s1_utterances) == 1
    assert s1_utterances[0].sequence_orders == [0, 2]
    assert len(result.interruptions) == 1
    assert result.interruptions[0].at_sequence_order == 1


@pytest.mark.asyncio
async def test_invalid_llm_proposal_falls_back_to_baseline():
    # Proposal mixes two speakers into one utterance -> must be dropped.
    fragments, assignments = make_inputs(["S1", "S2"])
    bad_response = {
        "utterances": [{"speaker": "S1", "fragment_indices": [0, 1], "confidence": 0.8}],
        "interruptions": [],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=bad_response)
        result = await stitcher.stitch(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0], [1]]  # baseline preserved


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_baseline():
    fragments, assignments = make_inputs(["S1", "S2", "S2"])
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(side_effect=RuntimeError("api down"))
        result = await stitcher.stitch(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0], [1, 2]]
    assert result.interruptions == []


def test_baseline_method_needs_no_llm():
    fragments, assignments = make_inputs(["S1", "S1", "S2"])
    stitcher = Stitcher()
    result = stitcher.baseline(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0, 1], [2]]
    assert all(u.confidence == 1.0 for u in result.utterances)
    assert result.interruptions == []


@pytest.mark.asyncio
async def test_interruption_ordinals_remap_after_sort():
    # LLM returns proposals in reverse order; after the merged list is sorted
    # by first sequence order, interruption ordinals must point at the same
    # utterances, not the same positions.
    fragments, assignments = make_inputs(["S1", "S2", "S1"])
    llm_response = {
        "utterances": [
            {"speaker": "S2", "fragment_indices": [1], "confidence": 0.9},
            {"speaker": "S1", "fragment_indices": [0, 2], "confidence": 0.8},
        ],
        "interruptions": [{"interrupting": 0, "interrupted": 1, "at_index": 1}],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=llm_response)
        result = await stitcher.stitch(fragments, assignments)
    assert [u.handle for u in result.utterances] == ["S1", "S2"]
    intr = result.interruptions[0]
    assert result.utterances[intr.interrupting_ordinal].handle == "S2"
    assert result.utterances[intr.interrupted_ordinal].handle == "S1"
    assert intr.at_sequence_order == 1


@pytest.mark.asyncio
async def test_interruption_referencing_dropped_proposal_is_dropped():
    fragments, assignments = make_inputs(["S1", "S2"])
    llm_response = {
        "utterances": [
            {"speaker": "S1", "fragment_indices": [0, 1], "confidence": 0.8},  # invalid: mixed
            {"speaker": "S2", "fragment_indices": [1], "confidence": 0.9},
        ],
        "interruptions": [{"interrupting": 1, "interrupted": 0, "at_index": 1}],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=llm_response)
        result = await stitcher.stitch(fragments, assignments)
    assert result.interruptions == []
