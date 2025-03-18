# tests/test_context_builder.py
import pytest
import pytest
import asyncio
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
    builder = ContextBuilder()

    context = builder.build_context(sentences, idx=2, window_size=1)

    assert context == "Second sentence about testing. Fourth sentence enhances detail."


def test_build_embedding_context(sentences):
    builder = ContextBuilder()

    embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=1)

    assert embedding_context.shape[0] == builder.embedder.get_sentence_embedding_dimension()
    assert embedding_context.any()  # not a zero vector


def test_build_all_contexts(sentences):
    builder = ContextBuilder()

    contexts = builder.build_all_contexts(sentences)

    assert len(contexts) == len(sentences)
    for idx, ctx in contexts.items():
        assert "immediate" in ctx
        assert "observer" in ctx
        assert isinstance(ctx["immediate"], str)
