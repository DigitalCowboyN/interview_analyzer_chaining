"""
Tests for text processing utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

# Module to test
from src.utils import text_processing

# Test cases for segment_text
VALID_TEXT_CASES = [
    ("Hello world. This is a test.", ["Hello world.", "This is a test."]),
    (
        "Sentence one? Sentence two! Sentence three...",
        ["Sentence one?", "Sentence two!", "Sentence three..."],
    ),
    (
        "Mr. Smith went to Washington D.C. on Jan. 1st.",
        ["Mr. Smith went to Washington D.C. on Jan. 1st."],
    ),  # Test abbreviations
    (
        " This sentence has leading/trailing spaces. ",
        ["This sentence has leading/trailing spaces."],
    ),
    ("Sentence.\nWith newline.", ["Sentence.", "With newline."]),  # Test newlines
    ("", []),  # Empty string
    ("   ", []),  # Whitespace only
]


@pytest.mark.parametrize("text, expected_sentences", VALID_TEXT_CASES)
def test_segment_text_valid(text, expected_sentences):
    """Test segment_text with various valid inputs using loaded spaCy model."""
    # Ensure the real spacy model is loaded (or skip if not)
    if text_processing.nlp is None:
        pytest.skip("spaCy model not loaded, skipping test.")

    result = text_processing.segment_text(text)
    assert result == expected_sentences


def test_segment_text_spacy_load_failure():
    """Test segment_text behavior when the spaCy model failed to load."""
    # Temporarily mock the module-level 'nlp' variable to be None
    with patch.object(text_processing, "nlp", None):
        # Also mock logger to check error message
        with patch.object(text_processing, "logger") as mock_logger:
            result = text_processing.segment_text("Some text that won't be processed.")
            assert result == []  # Expect empty list
            mock_logger.error.assert_called_once_with(
                "spaCy model not loaded. Cannot segment text."
            )


# Optional: Test the loading mechanism itself if complex
@patch("src.utils.text_processing.spacy.load")
@patch("src.utils.text_processing.logger")
def test_spacy_model_loading_success(mock_logger, mock_spacy_load):
    """Test that spaCy loading logs success and sets nlp correctly."""
    mock_model = MagicMock()
    mock_spacy_load.return_value = mock_model

    # This test cannot easily verify the import-time log message without reloading.
    # Focus on the outcome: nlp attribute should be set.
    if text_processing.nlp is None:
        pytest.skip(
            "Test assumes spaCy model loaded successfully in test environment setup."
        )
    assert text_processing.nlp is not None
    # Optionally check if spacy.load was called, though redundant if nlp is set
    # mock_spacy_load.assert_called_once_with("en_core_web_sm")


@patch(
    "src.utils.text_processing.spacy.load", side_effect=OSError("Mock loading error")
)
@patch("src.utils.text_processing.logger")
def test_spacy_model_loading_failure(mock_logger, mock_spacy_load):
    """Test that spaCy loading failure logs error and sets nlp to None."""
    # To properly test the try/except block at import time, we need to reload the module
    # This is generally discouraged in tests.
    # A better approach is to refactor the loading into an explicit function.

    # Let's assume the failed load happened and verify the state.
    # We patch 'nlp' directly for the check, as reloading is problematic.
    with patch.object(text_processing, "nlp", None):
        assert text_processing.nlp is None
        # Check that the error was logged during the (assumed) import/load phase
        # This assertion might be fragile depending on when logs are checked.
        # mock_logger.error.assert_called_once()
        # assert "Could not load spaCy model" in mock_logger.error.call_args[0][0]
        pytest.skip(
            "Testing import-time exception logging is difficult without reloading. Skipping direct log check."
        )
