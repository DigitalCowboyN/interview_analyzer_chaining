"""topic_segments extractor declaration: registry entry, prompt, strict model."""

import pytest
from pydantic import ValidationError

from src.enrichment.registry import ExtractorRegistry
from src.models.extractor_responses import SegmentsResult
from src.utils.helpers import load_yaml


def test_registry_loads_topic_segments_as_document_scope():
    specs = ExtractorRegistry.load("config/extractors.yaml")
    spec = next(s for s in specs if s.name == "topic_segments")
    assert spec.scope == "document"
    assert spec.response_model == "SegmentsResult"
    assert spec.prompt_key == "topic_segments"
    assert spec.resolve_model() is SegmentsResult


def test_prompt_formats_with_sentence_placeholder_only():
    prompts = load_yaml("prompts/core_extractors.yaml")
    template = prompts["topic_segments"]["prompt"]
    rendered = template.format(sentence="[0] [S1]: Hello")
    assert "[0] [S1]: Hello" in rendered
    assert '"segments"' in rendered


def test_segments_result_valid_and_strict():
    result = SegmentsResult.model_validate(
        {"segments": [{"topic": "Vendors", "start_index": 0, "end_index": 2, "confidence": 0.9}]}
    )
    assert result.segments[0].topic == "Vendors"

    with pytest.raises(ValidationError):
        SegmentsResult.model_validate(
            {
                "segments": [
                    {
                        "topic": "x",
                        "start_index": -1,
                        "end_index": 0,
                        "confidence": 0.5,
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):  # extra="forbid"
        SegmentsResult.model_validate({"segments": [], "extra_key": 1})
