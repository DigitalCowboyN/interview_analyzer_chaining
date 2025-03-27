"""
test_context_builder.py

This module contains unit tests for the ContextBuilder class, which is responsible for
building both textual and embedding-based contexts for sentences. The tests cover:
    - Building a context string from a list of sentences.
    - Generating an embedding context vector with the correct dimension.
    - Building contexts for all sentences and ensuring the returned structure is correct.
    - Edge cases such as no available context sentences and an empty input list.

Usage:
    Run the tests using pytest:
        pytest tests/test_context_builder.py

Modifications:
    - If the context window configuration changes, update the tests accordingly.
    - If the set of context keys changes, update the assertions in test_build_all_contexts.
"""

import pytest
import numpy as np
from src.agents.context_builder import ContextBuilder

@pytest.fixture
def sentences():
    return [
        "First sentence.",
        "Second sentence about testing.",
        "Third sentence provides context.",
        "Fourth sentence enhances detail.",
        "Fifth and final sentence."
    ]

def test_build_context(sentences):
    """
    Test the construction of immediate context around a given sentence.

    Verifies that build_context correctly constructs a context string that includes
    the specified number of surrounding sentences, excluding the target sentence.
    
    Expected behavior: For index 2 with window_size=1, it should combine sentence 1 and 3.
    """
    builder = ContextBuilder()
    context = builder.build_context(sentences, idx=2, window_size=1)
    expected = "Second sentence about testing. Fourth sentence enhances detail."
    assert context == expected

def test_build_embedding_context(sentences):
    """
    Test the generation of embedding-based context for a given sentence.

    Verifies that build_embedding_context returns an embedding vector with the correct dimension
    and that the vector is non-zero when context sentences are available.
    """
    builder = ContextBuilder()
    embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=1)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert embedding_context.shape[0] == expected_dim
    # Ensure the vector is not all zeros (indicating that context was computed)
    assert np.any(embedding_context != 0)

def test_build_embedding_context_empty(sentences):
    """
    Test build_embedding_context when no context sentences are available.

    For instance, if there is only one sentence or window_size=0, then no context is available,
    and the function should return a zero vector.
    """
    builder = ContextBuilder()
    # Use window_size 0 so that for any idx, the context list is empty.
    zero_vector = builder.build_embedding_context(sentences, idx=2, window_size=0)
    expected_dim = builder.embedder.get_sentence_embedding_dimension()
    assert zero_vector.shape[0] == expected_dim
    # The returned vector should be all zeros.
    assert np.all(zero_vector == 0)

def test_build_all_contexts(sentences):
    """
    Test the construction of contexts for all sentences.

    Verifies that build_all_contexts returns a dictionary mapping each sentence index to its
    corresponding context, and that each context dictionary contains the expected keys.
    
    Expected keys: "structure", "immediate", "observer", "broader", "overall".
    """
    builder = ContextBuilder()
    contexts = builder.build_all_contexts(sentences)
    assert len(contexts) == len(sentences)
    for idx, ctx in contexts.items():
        for key in ["structure", "immediate", "observer", "broader", "overall"]:
            assert key in ctx, f"Context missing key: {key}"
            assert isinstance(ctx[key], str)

def test_build_all_contexts_empty():
    """
    Test build_all_contexts with an empty list of sentences.

    Verifies that if an empty list is provided, build_all_contexts returns an empty dictionary.
    """
    builder = ContextBuilder()
    contexts = builder.build_all_contexts([])
    assert contexts == {}
