import pytest
from pydantic import ValidationError

from src.models.extractor_responses import (
    ClaimsResult,
    EntityMentionsResult,
    PurposeResult,
)


def test_purpose_result_validates_confidence_range():
    PurposeResult.model_validate({"purpose": "Query", "confidence": 0.5})
    with pytest.raises(ValidationError):
        PurposeResult.model_validate({"purpose": "Query", "confidence": 1.5})
    with pytest.raises(ValidationError):
        PurposeResult.model_validate({"purpose": "Query", "confidence": -0.1})


def test_entity_mentions_span_validated():
    ok = EntityMentionsResult.model_validate(
        {"entities": [{"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}]}
    )
    assert ok.entities[0].entity_type == "product"
    with pytest.raises(ValidationError):
        EntityMentionsResult.model_validate(
            {"entities": [{"text": "x", "entity_type": "product", "start": 9, "end": 4, "confidence": 0.9}]}
        )


def test_empty_entities_accepted():
    assert EntityMentionsResult.model_validate({"entities": []}).entities == []


def test_claim_kind_restricted():
    ClaimsResult.model_validate(
        {"claims": [{"text": "We will ship Friday", "kind": "commitment", "confidence": 0.8}]}
    )
    with pytest.raises(ValidationError):
        ClaimsResult.model_validate({"claims": [{"text": "x", "kind": "vibe", "confidence": 0.8}]})


def test_schema_export_is_json_schema():
    schema = PurposeResult.model_json_schema()
    assert schema["type"] == "object"
    assert "purpose" in schema["properties"]
