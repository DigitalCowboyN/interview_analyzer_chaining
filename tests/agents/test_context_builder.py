"""
Tests for src/agents/context_builder.py
"""

import pytest

from src.agents.context_builder import ContextBuilder

# --- Fixtures ---


@pytest.fixture
def mock_config_valid():
    """Provides a valid mock config dictionary for ContextBuilder tests."""
    return {
        "preprocessing": {
            "context_windows": {
                "immediate": 0,  # Window of 0 means just the target sentence
                "narrow": 1,    # Target + 1 before/after
                "broad": 2      # Target + 2 before/after
            }
        }
        # Other config sections not needed by ContextBuilder
    }


@pytest.fixture
def mock_config_missing_preprocessing():
    """Config missing the 'preprocessing' section."""
    return {}


@pytest.fixture
def mock_config_missing_windows():
    """Config missing the 'context_windows' sub-section."""
    return {"preprocessing": {}}


@pytest.fixture
def sample_sentences():
    """Provides a sample list of sentences for testing."""
    return [
        "Sentence 0.",
        "Sentence 1.",
        "Sentence 2.",
        "Sentence 3.",
        "Sentence 4.",
    ]

# --- Tests for __init__ ---


def test_init_success(mock_config_valid):
    """Test successful initialization with valid config."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    assert builder.context_windows == mock_config_valid["preprocessing"]["context_windows"]


def test_init_missing_preprocessing(mock_config_missing_preprocessing, caplog):
    """Test initialization when 'preprocessing' key is missing."""
    builder = ContextBuilder(config_dict=mock_config_missing_preprocessing)
    assert builder.context_windows == {}  # Should default to empty
    # Check for the WARNING log about empty windows
    assert "Context windows are empty. Check config ['preprocessing']['context_windows']." in caplog.text
    # Ensure the old error log is NOT present
    assert "Config key missing for context_windows: 'preprocessing'" not in caplog.text


def test_init_missing_windows(mock_config_missing_windows, caplog):
    """Test initialization when 'context_windows' key is missing."""
    builder = ContextBuilder(config_dict=mock_config_missing_windows)
    assert builder.context_windows == {}  # Should default to empty
    # Check for the WARNING log about empty windows
    assert "Context windows are empty. Check config ['preprocessing']['context_windows']." in caplog.text
    # Ensure the old error log is NOT present
    assert "Config key missing for context_windows: 'context_windows'" not in caplog.text

# --- Tests for build_context ---


def test_build_context_middle(sample_sentences, mock_config_valid):
    """Test building context for a sentence in the middle of the list."""
    builder = ContextBuilder(config_dict=mock_config_valid)

    # Window size 1 (narrow)
    context = builder.build_context(sample_sentences, idx=2, window_size=1)
    expected = (
        "Sentence 1.\n"
        ">>> TARGET: Sentence 2. <<<\n"
        "Sentence 3."
    )
    assert context == expected

    # Window size 2 (broad)
    context = builder.build_context(sample_sentences, idx=2, window_size=2)
    expected = (
        "Sentence 0.\n"
        "Sentence 1.\n"
        ">>> TARGET: Sentence 2. <<<\n"
        "Sentence 3.\n"
        "Sentence 4."
    )
    assert context == expected


def test_build_context_start(sample_sentences, mock_config_valid):
    """Test building context for the first sentence."""
    builder = ContextBuilder(config_dict=mock_config_valid)

    # Window size 1 (narrow)
    context = builder.build_context(sample_sentences, idx=0, window_size=1)
    expected = (
        ">>> TARGET: Sentence 0. <<<\n"
        "Sentence 1."
    )
    assert context == expected


def test_build_context_end(sample_sentences, mock_config_valid):
    """Test building context for the last sentence."""
    builder = ContextBuilder(config_dict=mock_config_valid)

    # Window size 1 (narrow)
    context = builder.build_context(sample_sentences, idx=4, window_size=1)
    expected = (
        "Sentence 3.\n"
        ">>> TARGET: Sentence 4. <<<"
    )
    assert context == expected

    # Window size 2 (broad) - should clip at the end
    context = builder.build_context(sample_sentences, idx=4, window_size=2)
    expected = (
        "Sentence 2.\n"
        "Sentence 3.\n"
        ">>> TARGET: Sentence 4. <<<"
    )
    assert context == expected


def test_build_context_window_zero(sample_sentences, mock_config_valid):
    """Test building context with window size 0."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    context = builder.build_context(sample_sentences, idx=2, window_size=0)
    expected = ">>> TARGET: Sentence 2. <<<"
    assert context == expected


def test_build_context_large_window(sample_sentences, mock_config_valid):
    """Test window size larger than list boundaries."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    context = builder.build_context(sample_sentences, idx=2, window_size=10)  # Window > list size
    expected = (
        "Sentence 0.\n"
        "Sentence 1.\n"
        ">>> TARGET: Sentence 2. <<<\n"
        "Sentence 3.\n"
        "Sentence 4."
    )
    assert context == expected  # Should be clipped to list boundaries


def test_build_context_invalid_index(sample_sentences, mock_config_valid):
    """Test building context with an out-of-bounds index."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    assert builder.build_context(sample_sentences, idx=-1, window_size=1) == ""
    assert builder.build_context(sample_sentences, idx=len(sample_sentences), window_size=1) == ""


def test_build_context_empty_list(mock_config_valid):
    """Test building context with an empty sentence list."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    assert builder.build_context([], idx=0, window_size=1) == ""

# --- Tests for build_all_contexts ---


def test_build_all_contexts_success(sample_sentences, mock_config_valid):
    """Test building all contexts for a list of sentences."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    all_contexts = builder.build_all_contexts(sample_sentences)

    assert len(all_contexts) == len(sample_sentences)  # Should have entry for each sentence

    # Check context for sentence 2 (as an example)
    assert 2 in all_contexts
    context_s2 = all_contexts[2]
    assert "immediate" in context_s2
    assert "narrow" in context_s2
    assert "broad" in context_s2

    # Verify immediate context (window 0)
    assert context_s2["immediate"] == ">>> TARGET: Sentence 2. <<<"

    # Verify narrow context (window 1)
    expected_narrow = (
        "Sentence 1.\n"
        ">>> TARGET: Sentence 2. <<<\n"
        "Sentence 3."
    )
    assert context_s2["narrow"] == expected_narrow

    # Verify broad context (window 2)
    expected_broad = (
        "Sentence 0.\n"
        "Sentence 1.\n"
        ">>> TARGET: Sentence 2. <<<\n"
        "Sentence 3.\n"
        "Sentence 4."
    )
    assert context_s2["broad"] == expected_broad

    # Check context for first sentence (index 0)
    assert 0 in all_contexts
    context_s0 = all_contexts[0]
    expected_narrow_s0 = (
        ">>> TARGET: Sentence 0. <<<\n"
        "Sentence 1."
    )
    assert context_s0["narrow"] == expected_narrow_s0


def test_build_all_contexts_empty_input(mock_config_valid):
    """Test build_all_contexts returns empty dict for empty sentence list."""
    builder = ContextBuilder(config_dict=mock_config_valid)
    assert builder.build_all_contexts([]) == {}


def test_build_all_contexts_uses_config_windows(sample_sentences):
    """Test that build_all_contexts uses the specific windows from config."""
    custom_config = {
        "preprocessing": {
            "context_windows": {
                "tiny": 0,
                "medium": 1
                # Only these two windows defined
            }
        }
    }
    builder = ContextBuilder(config_dict=custom_config)
    all_contexts = builder.build_all_contexts(sample_sentences)

    assert len(all_contexts) == len(sample_sentences)

    # Check an arbitrary sentence's contexts
    assert 1 in all_contexts
    context_s1 = all_contexts[1]

    # Should have keys from custom_config, not the default fixture
    assert "tiny" in context_s1
    assert "medium" in context_s1
    assert "immediate" not in context_s1  # From default fixture
    assert "narrow" not in context_s1    # From default fixture
    assert "broad" not in context_s1     # From default fixture

    # Check content
    assert context_s1["tiny"] == ">>> TARGET: Sentence 1. <<<"
    expected_medium = (
        "Sentence 0.\n"
        ">>> TARGET: Sentence 1. <<<\n"
        "Sentence 2."
    )
    assert context_s1["medium"] == expected_medium

# Potential test for build_sentence_context if needed,
# but focusing on the primary build_all_contexts method first.
# def test_build_sentence_context_variant(...)
