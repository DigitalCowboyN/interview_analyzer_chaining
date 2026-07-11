"""Generic projection handlers for lens events (Layer 3).

One handler set serves every lens: node labels and relationship names come
from the event payload, validated here as defense-in-depth (emit-time
validation against the lens YAML is primary) before being interpolated into
Cypher. Everything else rides as parameters.
"""

import json
import logging
import re

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler
from .speaker_handlers import _raise_if_no_writes

logger = logging.getLogger(__name__)

_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_RELATIONSHIP_RE = re.compile(r"^[A-Z][A-Z_]*$")

# Properties set explicitly by the handlers; extracted fields may not shadow them.
_RESERVED_PROPS = {
    "item_id",
    "lens",
    "lens_version",
    "node_type",
    "confidence",
    "model",
    "provider",
    "interview_id",
    "locked",
}


def _validate_label(node_type: str) -> str:
    """Node labels are interpolated into Cypher; refuse anything non-identifier."""
    if not _LABEL_RE.match(node_type):
        raise ValueError(f"Invalid node label: {node_type!r}")
    return node_type


def _validate_relationship(relationship: str) -> str:
    if not _RELATIONSHIP_RE.match(relationship):
        raise ValueError(f"Invalid relationship name: {relationship!r}")
    return relationship


def _flatten_fields(fields: dict) -> dict:
    """Extracted fields as node properties: scalars and lists of scalars kept,
    anything else JSON-dumped; reserved property names get a field_ prefix."""
    props = {}
    for key, value in fields.items():
        if key in _RESERVED_PROPS:
            key = f"field_{key}"
        is_scalar = value is None or isinstance(value, (str, int, float, bool))
        is_scalar_list = isinstance(value, list) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        )
        props[key] = value if is_scalar or is_scalar_list else json.dumps(value)
    return props


class LensAppliedHandler(BaseProjectionHandler):
    """Supersession marker: a new lens run deletes the interview+lens's prior
    UNLOCKED items. Locked (human-overridden) items always survive."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (n:LensItem {interview_id: $interview_id, lens: $lens})
        WHERE n.lens_version < $lens_version AND coalesce(n.locked, false) = false
        DETACH DELETE n
        """
        await tx.run(
            query,
            interview_id=event.aggregate_id,
            lens=data["lens"],
            lens_version=data["lens_version"],
        )
        logger.info(
            f"Applied LensApplied {data['lens']} v{data['lens_version']} "
            f"for interview {event.aggregate_id}"
        )


class LensExtractionGeneratedHandler(BaseProjectionHandler):
    """Materializes one lens item as a dual-labeled node (:LensItem:<Label>)
    with SUPPORTED_BY fragment grounding and declarative speaker links."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        label = _validate_label(data["node_type"])
        supporting_ids = data.get("supporting_fragment_ids", [])
        speaker_links = data.get("speaker_links", [])

        # Group speaker links by (validated) relationship name; each group
        # becomes one interpolated clause, ids ride as parameters.
        rel_ids: dict = {}
        for link in speaker_links:
            rel = _validate_relationship(link["relationship"])
            rel_ids.setdefault(rel, []).append(link["speaker_id"])

        lines = [
            f"MERGE (n:LensItem:{label} {{item_id: $item_id}})",
            "SET n.lens = $lens, n.lens_version = $lens_version,",
            "    n.node_type = $node_type, n.confidence = $confidence,",
            "    n.model = $model, n.provider = $provider,",
            "    n.interview_id = $interview_id, n += $props",
            "WITH n",
            "OPTIONAL MATCH (s:Fragment) WHERE s.aggregate_id IN $supporting_ids",
            "FOREACH (f IN CASE WHEN s IS NULL THEN [] ELSE [s] END |",
            "    MERGE (n)-[:SUPPORTED_BY]->(f))",
            "WITH n, count(DISTINCT s) AS supported",
        ]
        linked_terms = []
        params = {
            "item_id": data["item_id"],
            "lens": data["lens"],
            "lens_version": data["lens_version"],
            "node_type": data["node_type"],
            "confidence": data["confidence"],
            "model": data["model"],
            "provider": data["provider"],
            "interview_id": event.aggregate_id,
            "props": _flatten_fields(data.get("fields", {})),
            "supporting_ids": supporting_ids,
        }
        carried = "supported"
        for i, (rel, ids) in enumerate(rel_ids.items()):
            params[f"speaker_ids_{i}"] = ids
            lines += [
                f"OPTIONAL MATCH (sp{i}:Speaker) WHERE sp{i}.speaker_id IN $speaker_ids_{i}",
                f"FOREACH (x IN CASE WHEN sp{i} IS NULL THEN [] ELSE [sp{i}] END |",
                f"    MERGE (n)-[:{rel}]->(x))",
                f"WITH n, {carried}, count(DISTINCT sp{i}) AS linked_{i}",
            ]
            linked_terms.append(f"linked_{i}")
            carried += f", linked_{i}"
        linked_expr = " + ".join(linked_terms) if linked_terms else "0"
        lines.append(f"RETURN supported, {linked_expr} AS linked")
        query = "\n".join(lines)

        result = await tx.run(query, **params)
        record = await result.single()
        supported = record["supported"] if record is not None else 0
        linked = record["linked"] if record is not None else 0
        if supporting_ids and supported == 0:
            # Fragments arrive on the Sentence subscription; raise so
            # retry/park engages instead of sealing an ungrounded item.
            raise ValueError(
                f"LensExtractionGenerated {data['item_id']}: supporting fragments "
                f"not yet projected ({supporting_ids})"
            )
        if speaker_links and linked == 0:
            raise ValueError(
                f"LensExtractionGenerated {data['item_id']}: speakers "
                f"not yet projected ({[link['speaker_id'] for link in speaker_links]})"
            )
        logger.info(
            f"Applied LensExtractionGenerated {data['item_id']} "
            f"({label}, {supported} supports, {linked} speaker links)"
        )


class LensExtractionOverriddenHandler(BaseProjectionHandler):
    """Human correction: overwrites fields and locks the node against re-runs."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (n:LensItem {item_id: $item_id})
        SET n += $props, n.locked = true,
            n.overridden_at = datetime($occurred_at), n.override_note = $note
        """
        result = await tx.run(
            query,
            item_id=data["item_id"],
            props=_flatten_fields(data.get("fields_overridden", {})),
            occurred_at=event.occurred_at.isoformat(),
            note=data.get("note"),
        )
        _raise_if_no_writes(await result.consume(), "LensExtractionOverridden", data["item_id"])
        logger.info(f"Applied LensExtractionOverridden {data['item_id']}")
