"""Projection handlers for embedding events (Layer 2).

Decodes the inline vector and writes it as a node property, tagged with the
model. One Neo4j vector index per model (created lazily, once per handler
instance). No cross-model comparison — the provider is config-pinned.
"""

import logging
import re

from src.enrichment.embedder import decode_vector
from src.events.envelope import EventEnvelope
from src.utils.neo4j_driver import Neo4jConnectionManager

from .base_handler import BaseProjectionHandler
from .speaker_handlers import _raise_if_no_writes

logger = logging.getLogger(__name__)


def _sanitize(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", model)


async def _ensure_vector_index(ensured: set, label: str, prefix: str, model: str, dim: int):
    """Create a per-model vector index in AUTO-COMMIT.

    Neo4j forbids schema DDL inside a data-write transaction, so the index is
    created on its own session — never on the event's `tx`. Idempotent
    (IF NOT EXISTS); tracked per model so a handler serving multiple models
    (e.g. during replay of a re-embedding) creates every needed index.
    """
    if model in ensured:
        return
    name = f"{prefix}_{_sanitize(model)}"
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=dim,
        )
    ensured.add(model)


class EmbeddingGeneratedHandler(BaseProjectionHandler):
    """Writes a fragment embedding onto its Sentence node."""

    def __init__(self, parked_events_manager=None):
        super().__init__(parked_events_manager)
        self._ensured_models: set = set()

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        await _ensure_vector_index(
            self._ensured_models, "Sentence", "fragment_embedding", data["model"], data["dim"]
        )
        vector = decode_vector(data["vector_b64"])
        query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        SET s.embedding = $vector, s.embedding_model = $model, s.embedding_dim = $dim
        """
        result = await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            vector=vector,
            model=data["model"],
            dim=data["dim"],
        )
        _raise_if_no_writes(await result.consume(), "EmbeddingGenerated", event.aggregate_id)


class UtteranceEmbeddingGeneratedHandler(BaseProjectionHandler):
    """Writes an utterance embedding onto its Utterance node."""

    def __init__(self, parked_events_manager=None):
        super().__init__(parked_events_manager)
        self._ensured_models: set = set()

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        await _ensure_vector_index(
            self._ensured_models, "Utterance", "utterance_embedding", data["model"], data["dim"]
        )
        vector = decode_vector(data["vector_b64"])
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        SET u.embedding = $vector, u.embedding_model = $model, u.embedding_dim = $dim
        """
        result = await tx.run(
            query,
            utterance_id=data["utterance_id"],
            vector=vector,
            model=data["model"],
            dim=data["dim"],
        )
        _raise_if_no_writes(
            await result.consume(), "UtteranceEmbeddingGenerated", data["utterance_id"]
        )
