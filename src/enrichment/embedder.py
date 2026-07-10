"""Embedding providers (config-pinned — never per-call failover).

Vectors from different models are incomparable; the provider is pinned in
config, every vector is tagged {model, dim}, and switching means re-running
the embedding extractor via event replay.
"""

import asyncio
import base64
import struct
from typing import Any, Dict, List, Optional, Protocol

from src.utils.logger import get_logger

logger = get_logger()


def encode_vector(vec: List[float]) -> str:
    """Base64 of little-endian float32 — compact, replay-safe event payload."""
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")


def decode_vector(s: str) -> List[float]:
    raw = base64.b64decode(s.encode("ascii"))
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _validate_dims(vectors: List[List[float]], dim: int) -> None:
    """A misconfigured embedding_dim would silently create a wrong-dim index."""
    for v in vectors:
        if len(v) != dim:
            raise ValueError(f"Embedder returned dim {len(v)}, configured dim {dim}")


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed(self, texts: List[str]) -> List[List[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model_name: str, dim: int, api_key: str):
        from openai import AsyncOpenAI

        self.model_name = model_name
        self.dim = dim
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model_name, input=texts, dimensions=self.dim
        )
        vectors = [item.embedding for item in response.data]
        _validate_dims(vectors, self.dim)
        return vectors


class LocalEmbedder:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None  # lazy: sentence-transformers import is heavy

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = await asyncio.to_thread(self._load)
        raw = await asyncio.to_thread(model.encode, texts)
        vectors = [list(map(float, v)) for v in raw]
        _validate_dims(vectors, self.dim)
        return vectors


def get_embedder(config_dict: Optional[Dict[str, Any]] = None) -> Embedder:
    """Build the config-pinned embedder (no failover — model space matters)."""
    from src.config import config as global_config

    cfg = config_dict if config_dict is not None else global_config
    embeddings_cfg = cfg.get("embeddings", {})
    provider = embeddings_cfg.get("provider", "openai")
    if provider == "openai":
        ocfg = embeddings_cfg.get("openai", {})
        return OpenAIEmbedder(
            model_name=ocfg.get("model_name", "text-embedding-3-small"),
            dim=ocfg.get("embedding_dim", 1536),
            api_key=cfg.get("openai", {}).get("api_key", ""),
        )
    if provider == "local":
        lcfg = cfg.get("embedding", {})  # reuse the pre-existing block
        return LocalEmbedder(
            model_name=lcfg.get("model_name", "all-MiniLM-L6-v2"),
            dim=lcfg.get("embedding_dim", 384),
        )
    raise ValueError(f"Unknown embeddings.provider: {provider!r} (openai|local)")
