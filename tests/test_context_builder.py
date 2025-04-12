"""
test_context_builder.py

This module contains unit tests for the ContextBuilder class, which is responsible for
building both textual and embedding-based contexts for sentences. The tests cover:
    - Building a context string with newline separators and a marked target sentence.
    - Generating an embedding context vector (excluding the target).
    - Building contexts for all sentences and verifying the structure.
    - Edge cases: context at start/end of list, zero window size, empty input list.

Usage:
    Run the tests using pytest:
        pytest tests/test_context_builder.py

Modifications:
    - If the context window configuration changes, update relevant tests if needed.
    - If the target marker format changes, update test_build_context assertions.
"""

import pytest
import numpy as np
from src.agents.context_builder import ContextBuilder
from src.config import config # To access configured embedding dimension

@pytest.fixture
def sentences():
    # Sample sentences for testing
    return [
        "Sentence 0.", # idx 0
        "Sentence 1.", # idx 1
        "Sentence 2 is the target.", # idx 2
        "Sentence 3.", # idx 3
        "Sentence 4."  # idx 4
    ]

@pytest.fixture
def builder():
    # Fixture to provide a ContextBuilder instance
    return ContextBuilder()

def test_build_context_basic(builder, sentences):
    """
    Test build_context with a window size of 1 around a middle sentence.
    """
    context = builder.build_context(sentences, idx=2, window_size=1)
    # Use triple quotes for correct multiline string with actual newlines
    expected = """Sentence 1.
>>> TARGET: Sentence 2 is the target. <<<
Sentence 3."""
    assert context == expected

def test_build_context_start(builder, sentences):
    """
    Test build_context for the first sentence (index 0).
    """
    context = builder.build_context(sentences, idx=0, window_size=1)
    # Use triple quotes
    expected = """>>> TARGET: Sentence 0. <<<
Sentence 1."""
    assert context == expected

def test_build_context_end(builder, sentences):
    """
    Test build_context for the last sentence (index 4).
    """
    context = builder.build_context(sentences, idx=4, window_size=1)
    # Use triple quotes
    expected = """Sentence 3.
>>> TARGET: Sentence 4. <<<"""
    assert context == expected

def test_build_context_zero_window(builder, sentences):
    """
    Test build_context with window_size=0. Should only return the marked target.
    """
    context = builder.build_context(sentences, idx=2, window_size=0)
    expected = ">>> TARGET: Sentence 2 is the target. <<<"
    assert context == expected

def test_build_context_large_window(builder, sentences):
    """
    Test build_context with a window size larger than the list boundaries.
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
    """Test build_context with an invalid index."""
    assert builder.build_context(sentences, idx=-1, window_size=1) == ""
    assert builder.build_context(sentences, idx=len(sentences), window_size=1) == ""

def test_build_context_empty_list(builder):
    """Test build_context with an empty sentences list."""
    assert builder.build_context([], idx=0, window_size=1) == ""

# --- Embedding Context Tests ---

def test_build_embedding_context_basic(builder, sentences):
    """
    Test embedding context generation, ensuring target is excluded.
    Window size 1 around idx 2 -> uses sentences 1 and 3 for embedding.
    """
    embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert isinstance(embedding_context, np.ndarray)
    assert embedding_context.shape == (expected_dim,)
    # Check it's not zero vector (assuming sents 1 & 3 have non-zero avg embedding)
    assert np.any(embedding_context != 0)

def test_build_embedding_context_zero_window(builder, sentences):
    """
    Test embedding context with window_size=0. No surrounding sentences, expect zero vector.
    """
    embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=0)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert isinstance(embedding_context, np.ndarray)
    assert embedding_context.shape == (expected_dim,)
    assert np.all(embedding_context == 0) # Should be zero vector

def test_build_embedding_context_start(builder, sentences):
    """
    Test embedding context at start. window=1, idx=0 -> uses sentence 1 only.
    """
    embedding_context = builder.build_embedding_context(sentences, idx=0, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert isinstance(embedding_context, np.ndarray)
    assert embedding_context.shape == (expected_dim,)
    assert np.any(embedding_context != 0) 

def test_build_embedding_context_end(builder, sentences):
    """
    Test embedding context at end. window=1, idx=4 -> uses sentence 3 only.
    """
    embedding_context = builder.build_embedding_context(sentences, idx=4, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert isinstance(embedding_context, np.ndarray)
    assert embedding_context.shape == (expected_dim,)
    assert np.any(embedding_context != 0)

def test_build_embedding_context_empty_list(builder):
    """Test embedding context with empty list."""
    embedding_context = builder.build_embedding_context([], idx=0, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert np.all(embedding_context == 0)
    assert embedding_context.shape == (expected_dim,)

def test_build_embedding_context_invalid_idx(builder, sentences):
    """Test embedding context with invalid index."""
    embedding_context = builder.build_embedding_context(sentences, idx=-1, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert np.all(embedding_context == 0)
    assert embedding_context.shape == (expected_dim,)

# --- build_all_contexts Tests ---

def test_build_all_contexts_structure(builder, sentences):
    """
    Test the structure returned by build_all_contexts.
    Checks number of entries and keys for each entry.
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
    Test build_all_contexts with an empty list. Expect empty dictionary.
    """
    contexts = builder.build_all_contexts([])
    assert contexts == {}
