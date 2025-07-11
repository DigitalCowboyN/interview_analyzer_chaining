"""
test_context_builder.py

This module contains unit tests for the `ContextBuilder` class from
`src.agents.context_builder.py`, focusing on its textual context generation capabilities.

The tests cover:
    - Building textual context strings around a target sentence with varying window sizes.
    - Correct handling of sentence boundaries (start/end of the list).
    - Correct formatting, including newline separators and target sentence markers.
    - Building contexts for all sentences and verifying the output structure.
    - Edge cases like zero window size and empty input lists.

Note: Tests related to `build_embedding_context` are currently commented out,
      mirroring the status of the corresponding method in the source code.

Usage:
    Run the tests using pytest:
        pytest tests/test_context_builder.py
"""

# import numpy as np  # Only needed for embedding context tests, which are currently commented out
import pytest

from src.agents.context_builder import ContextBuilder
from src.config import config  # To access configured embedding dimension


@pytest.fixture
def sentences():
    """
    Pytest fixture providing a sample list of sentences for testing.

    Returns:
        List[str]: A list of sample sentences.
    """
    # Sample sentences for testing
    return [
        "Sentence 0.",  # idx 0
        "Sentence 1.",  # idx 1
        "Sentence 2 is the target.",  # idx 2
        "Sentence 3.",  # idx 3
        "Sentence 4.",  # idx 4
    ]


@pytest.fixture
def builder():
    """
    Pytest fixture providing a `ContextBuilder` instance for testing.

    Returns:
        ContextBuilder: A new instance of the context builder.
    """
    # Fixture to provide a ContextBuilder instance
    return ContextBuilder()


def test_build_context_basic(builder, sentences):
    """
    Test `build_context` with a window size of 1 around a middle sentence.

    Verifies that the correct preceding and succeeding sentences are included,
    separated by newlines, and the target sentence is correctly marked.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    context = builder.build_context(sentences, idx=2, window_size=1)
    # Use triple quotes for correct multiline string with actual newlines
    expected = """Sentence 1.
>>> TARGET: Sentence 2 is the target. <<<
Sentence 3."""
    assert context == expected


def test_build_context_start(builder, sentences):
    """
    Test `build_context` correctly handles the first sentence (index 0).

    Verifies that only the target sentence and the succeeding sentence (within the window)
    are included, with the target correctly marked.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    context = builder.build_context(sentences, idx=0, window_size=1)
    # Use triple quotes
    expected = """>>> TARGET: Sentence 0. <<<
Sentence 1."""
    assert context == expected


def test_build_context_end(builder, sentences):
    """
    Test `build_context` correctly handles the last sentence.

    Verifies that only the target sentence and the preceding sentence (within the window)
    are included, with the target correctly marked.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    context = builder.build_context(sentences, idx=4, window_size=1)
    # Use triple quotes
    expected = """Sentence 3.
>>> TARGET: Sentence 4. <<<"""
    assert context == expected


def test_build_context_zero_window(builder, sentences):
    """
    Test `build_context` with window_size=0.

    Verifies that only the marked target sentence is returned when the window size is zero.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    context = builder.build_context(sentences, idx=2, window_size=0)
    expected = ">>> TARGET: Sentence 2 is the target. <<<"
    assert context == expected


def test_build_context_large_window(builder, sentences):
    """
    Test `build_context` with a window size larger than list boundaries.

    Verifies that the context includes all available sentences up to the list start/end
    when the window size exceeds the number of available preceding/succeeding sentences.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    context = builder.build_context(sentences, idx=2, window_size=10)
    # Use triple quotes
    expected = """Sentence 0.
Sentence 1.
>>> TARGET: Sentence 2 is the target. <<<
Sentence 3.
Sentence 4."""
    assert context == expected


def test_build_context_invalid_idx(builder, sentences):
    """
    Test `build_context` returns an empty string for invalid indices.

    Verifies that providing an index less than 0 or greater than/equal to the list length
    results in an empty string return value.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    assert builder.build_context(sentences, idx=-1, window_size=1) == ""
    assert builder.build_context(sentences, idx=len(sentences), window_size=1) == ""


def test_build_context_empty_list(builder):
    """
    Test `build_context` returns an empty string when given an empty list.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
    """
    assert builder.build_context([], idx=0, window_size=1) == ""


# --- Embedding Context Tests --- (Commented out as build_embedding_context is commented out)
# def test_build_embedding_context_basic(builder, sentences):
#     """
#     Test embedding context generation, ensuring target is excluded.
#     Window size 1 around idx 2 -> uses sentences 1 and 3 for embedding.
#     """
#     # Temporarily un-comment the embedder init in ContextBuilder for this test if needed,
#     # or mock the embedder object if testing ContextBuilder logic independently
#     if not hasattr(builder, 'embedder'):
#         pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=1)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert isinstance(embedding_context, np.ndarray)
#     assert embedding_context.shape == (expected_dim,)
#     # Check it's not zero vector (assuming sents 1 & 3 have non-zero avg embedding)
#     assert np.any(embedding_context != 0)
#
# def test_build_embedding_context_zero_window(builder, sentences):
#     """
#     Test embedding context with window_size=0. No surrounding sentences, expect zero vector.
#     """
#     if not hasattr(builder, 'embedder'):
#          pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=0)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert isinstance(embedding_context, np.ndarray)
#     assert embedding_context.shape == (expected_dim,)
#     assert np.all(embedding_context == 0) # Should be zero vector
#
# def test_build_embedding_context_start(builder, sentences):
#     """
#     Test embedding context at start. window=1, idx=0 -> uses sentence 1 only.
#     """
#     if not hasattr(builder, 'embedder'):
#          pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context(sentences, idx=0, window_size=1)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert isinstance(embedding_context, np.ndarray)
#     assert embedding_context.shape == (expected_dim,)
#     assert np.any(embedding_context != 0)
#
# def test_build_embedding_context_end(builder, sentences):
#     """
#     Test embedding context at end. window=1, idx=4 -> uses sentence 3 only.
#     """
#     if not hasattr(builder, 'embedder'):
#          pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context(sentences, idx=4, window_size=1)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert isinstance(embedding_context, np.ndarray)
#     assert embedding_context.shape == (expected_dim,)
#     assert np.any(embedding_context != 0)
#
# def test_build_embedding_context_empty_list(builder):
#     """Test embedding context with empty list."""
#     if not hasattr(builder, 'embedder'):
#          pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context([], idx=0, window_size=1)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert np.all(embedding_context == 0)
#     assert embedding_context.shape == (expected_dim,)
#
# def test_build_embedding_context_invalid_idx(builder, sentences):
#     """Test embedding context with invalid index."""
#     if not hasattr(builder, 'embedder'):
#          pytest.skip("Skipping embedding test as embedder is commented out in ContextBuilder")
#     embedding_context = builder.build_embedding_context(sentences, idx=-1, window_size=1)
#     expected_dim = builder.embedder.get_sentence_embedding_dimension()
#     assert np.all(embedding_context == 0)
#     assert embedding_context.shape == (expected_dim,)

# --- build_all_contexts Tests ---


def test_build_all_contexts_structure(builder, sentences):
    """
    Test the structure and content returned by `build_all_contexts`.

    Verifies that:
    - The outer dictionary has keys corresponding to each sentence index.
    - Each inner dictionary contains keys for all configured context window types.
    - Each context string value is a string and contains the correct target marker.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
        sentences: Fixture providing the sample list of sentences.
    """
    contexts = builder.build_all_contexts(sentences)
    assert len(contexts) == len(sentences)
    expected_keys = list(config["preprocessing"]["context_windows"].keys())

    for idx in range(len(sentences)):
        assert idx in contexts
        assert isinstance(contexts[idx], dict)
        # Check all expected keys are present and in the same order (Python 3.7+ dicts preserve order)
        assert list(contexts[idx].keys()) == expected_keys
        for key in expected_keys:
            assert isinstance(contexts[idx][key], str)
            # Check that the target sentence is marked within each context string
            # Ensure sentence text exists before checking substring
            if sentences[idx]:
                assert f">>> TARGET: {sentences[idx]} <<<" in contexts[idx][key]


def test_build_all_contexts_empty_list(builder):
    """
    Test `build_all_contexts` returns an empty dictionary for an empty input list.

    Args:
        builder: Fixture providing a `ContextBuilder` instance.
    """
    contexts = builder.build_all_contexts([])
    assert contexts == {}
