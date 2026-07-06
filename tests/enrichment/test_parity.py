"""Parity: the registry path produces every dimension the legacy pipeline did."""

LEGACY_DIMENSIONS = {
    "function_type",
    "structure_type",
    "purpose",
    "topic_level_1",
    "topic_level_3",
}


def test_registry_covers_all_legacy_dimensions():
    from src.enrichment.registry import ExtractorRegistry

    names = {s.name for s in ExtractorRegistry.load("config/extractors.yaml")}
    assert LEGACY_DIMENSIONS <= names
    assert {"overall_keywords", "domain_keywords"} <= names


def test_analysis_event_shape_superset_of_legacy():
    from src.events.sentence_events import AnalysisGeneratedData

    fields = set(AnalysisGeneratedData.model_fields)
    legacy = {"model", "version", "classification", "keywords", "topics",
              "domain_keywords", "confidence", "raw_ref"}
    assert legacy <= fields
