"""One-shot idempotent migration: retire the M4.5a :Sentence shim.

Usage: python -m src.projections.migrate_shim_drop

Run AFTER deploying the M4.8 code (write path is :Fragment-only from that
point). Against a pre-M4.8 graph this:

1. Recreates every stale `fragment_embedding_*` vector index (built on the
   :Sentence label) on :Fragment instead, matching the DDL shape
   `embedding_handlers._ensure_vector_index` now uses for new indexes. This
   runs BEFORE the label strip below: stripping :Sentence first would empty
   the old :Sentence-anchored index, and `IF NOT EXISTS` would mask the gap
   when the lazy ensure later tries to create the (already-named) index on
   :Fragment.
2. Strips the :Sentence label from every node, batched via the new
   (non-deprecated) `CALL (s) { ... } IN TRANSACTIONS OF 1000 ROWS` subquery
   form -- the deprecated `CALL {} IN TRANSACTIONS` form used by the retired
   `migrate_fragment_label` CLI is not used here.
3. Drops the three shim btree indexes (`sentence_sentence_id`,
   `sentence_lookup`, `sentence_sequence`) that only ever served :Sentence
   lookups.

Safe to re-run: a second run finds no stale vector indexes, relabels 0 nodes,
and re-issues the (harmless, IF EXISTS) btree drops.
"""

import asyncio
import json

from src.utils.neo4j_driver import Neo4jConnectionManager

BTREE_SHIM_INDEXES = ["sentence_sentence_id", "sentence_lookup", "sentence_sequence"]

SHOW_INDEXES_QUERY = "SHOW INDEXES"

COUNT_SENTENCE_QUERY = "MATCH (s:Sentence) RETURN count(s)"

STRIP_LABEL_QUERY = """
MATCH (s:Sentence)
CALL (s) {
    REMOVE s:Sentence
} IN TRANSACTIONS OF 1000 ROWS
"""


def _is_stale_fragment_embedding_index(row: dict) -> bool:
    return (
        row["type"] == "VECTOR"
        and row["name"].startswith("fragment_embedding_")
        and row["labelsOrTypes"] == ["Sentence"]
    )


async def _discover_stale_vector_indexes(session) -> list:
    result = await session.run(SHOW_INDEXES_QUERY)
    return [row async for row in result if _is_stale_fragment_embedding_index(row)]


async def _recreate_vector_index_on_fragment(session, row: dict) -> None:
    """Drop the :Sentence-anchored index, recreate it same-named on :Fragment.

    DDL shape mirrors embedding_handlers._ensure_vector_index exactly (label,
    options/similarity function) so the lazy-ensure and migration paths agree.
    """
    name = row["name"]
    prop = row["properties"][0]
    dim = row["options"]["indexConfig"]["vector.dimensions"]
    await session.run(f"DROP INDEX {name} IF EXISTS")
    await session.run(
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        f"FOR (f:Fragment) ON (f.{prop}) "
        "OPTIONS {indexConfig: {"
        "`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
        dim=dim,
    )


async def migrate(session) -> dict:
    stale_indexes = await _discover_stale_vector_indexes(session)
    for row in stale_indexes:
        await _recreate_vector_index_on_fragment(session, row)
    if stale_indexes:
        await session.run("CALL db.awaitIndexes()")

    count_result = await session.run(COUNT_SENTENCE_QUERY)
    count_record = await count_result.single()
    relabeled = count_record["count(s)"] if count_record else 0
    if relabeled:
        await session.run(STRIP_LABEL_QUERY)

    for name in BTREE_SHIM_INDEXES:
        await session.run(f"DROP INDEX {name} IF EXISTS")

    return {
        "relabeled": relabeled,
        "vector_indexes": [row["name"] for row in stale_indexes],
        "btree_dropped": BTREE_SHIM_INDEXES,
    }


async def main() -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        summary = await migrate(session)
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
