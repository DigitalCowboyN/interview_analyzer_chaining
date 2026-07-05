import pytest
from pydantic import ValidationError

from src.models.ingestion_responses import (
    SpeakerWindowResponse,
    StitchWindowResponse,
)


def test_speaker_window_response_validates():
    resp = SpeakerWindowResponse.model_validate(
        {"assignments": [{"index": 0, "speaker": "S1", "confidence": 0.9}]}
    )
    assert resp.assignments[0].speaker == "S1"


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SpeakerWindowResponse.model_validate(
            {"assignments": [{"index": 0, "speaker": "S1", "confidence": 1.5}]}
        )


def test_stitch_window_response_validates():
    resp = StitchWindowResponse.model_validate(
        {
            "utterances": [{"speaker": "S1", "fragment_indices": [0, 2], "confidence": 0.7}],
            "interruptions": [{"interrupting": 1, "interrupted": 0, "at_index": 1}],
        }
    )
    assert resp.utterances[0].fragment_indices == [0, 2]
    assert resp.interruptions[0].at_index == 1


def test_prompts_file_has_required_keys():
    from src.utils.helpers import load_yaml

    prompts = load_yaml("prompts/ingestion_prompts.yaml")
    assert "{fragments}" in prompts["speaker_window"]["prompt"]
    assert "{fragments}" in prompts["stitch_window"]["prompt"]


def test_negative_confidence_rejected():
    with pytest.raises(ValidationError):
        SpeakerWindowResponse.model_validate(
            {"assignments": [{"index": 0, "speaker": "S1", "confidence": -0.1}]}
        )


def test_empty_fragment_indices_rejected():
    with pytest.raises(ValidationError):
        StitchWindowResponse.model_validate(
            {"utterances": [{"speaker": "S1", "fragment_indices": [], "confidence": 0.5}]}
        )
