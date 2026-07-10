import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.enrichment.embedder import decode_vector, encode_vector, get_embedder


def test_vector_roundtrip():
    vec = [0.1, -2.5, 3.25]
    decoded = decode_vector(encode_vector(vec))
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(vec, decoded))


def test_get_embedder_openai_pinned():
    cfg = {
        "embeddings": {
            "provider": "openai",
            "openai": {"model_name": "text-embedding-3-small", "embedding_dim": 1536},
        },
        "openai": {"api_key": "sk-test"},
    }
    embedder = get_embedder(cfg)
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dim == 1536


def test_get_embedder_local_uses_existing_embedding_block():
    cfg = {
        "embeddings": {"provider": "local"},
        "embedding": {"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384},
    }
    embedder = get_embedder(cfg)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.dim == 384


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="embeddings.provider"):
        get_embedder({"embeddings": {"provider": "mystery"}})


@pytest.mark.asyncio
async def test_openai_embedder_calls_api():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 3
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    vectors = await embedder.embed(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_embedder_rejects_wrong_dim():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 4  # API returns 3 -> mismatch
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    with pytest.raises(ValueError, match="dim"):
        await embedder.embed(["hello"])


@pytest.mark.asyncio
async def test_openai_embedder_passes_dimensions_param():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 3
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    await embedder.embed(["hello"])
    assert embedder.client.embeddings.create.call_args.kwargs["dimensions"] == 3
