from src.enrichment.syntax_check import syntax_flags


def test_agreement_returns_empty():
    assert syntax_flags("Can you hear me?", "interrogative", "simple") == {}


def test_function_disagreement_flagged():
    flags = syntax_flags("Can you hear me?", "declarative", "simple")
    assert "function_type_disagreement" in flags
    assert "interrogative" in flags["function_type_disagreement"]


def test_structure_disagreement_flagged():
    flags = syntax_flags(
        "I stayed home because it rained, and Bob left early.", "declarative", "simple"
    )
    assert "structure_type_disagreement" in flags


def test_empty_labels_not_flagged():
    assert syntax_flags("Hello there.", "", "") == {}


def test_case_insensitive_agreement():
    assert syntax_flags("Close the door!", "Exclamatory", "Simple") == {}
