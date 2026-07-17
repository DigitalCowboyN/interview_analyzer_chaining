"""Live migration proof (integration): `migrate_shim_drop.migrate` against a
seeded PRE-drop world in the real test Neo4j.

Seeds dual-labeled (:Sentence:Fragment) nodes, a :Sentence-anchored vector
index named `fragment_embedding_testmodel`, and the three shim btree indexes
via DIRECT SESSION WRITES. That is normally forbidden (handlers are the sole
Neo4j writers -- see the Global Constraints in the M4.8 plan) but is
acceptable HERE ONLY: this test impersonates a legacy (pre-M4.8) database
state that no longer exists on the current write path, not a shortcut around
the real write path.

Requires `make test-infra-up`. Shares the test Neo4j with other suites, so
every assertion is scoped to this test's seeded sentence_ids / the
`fragment_embedding_testmodel` index name -- never a bare global count.
"""

import uuid as uuid_mod

import pytest

from src.projections.migrate_shim_drop import migrate
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

SEED_VECTOR = [0.1, 0.2]
INDEX_NAME = "fragment_embedding_testmodel"
BTREE_SHIM_INDEXES = ["sentence_sentence_id", "sentence_lookup", "sentence_sequence"]


async def _seed_pre_drop_world(session, sentence_ids):
    """Impersonate a pre-M4.8 graph: dual-labeled nodes + shim indexes.

    Direct session writes (not the handler write path) -- see module
    docstring for why that's acceptable in this one test.
    """
    for sid in sentence_ids:
        await session.run(
            """
            CREATE (s:Sentence:Fragment {
                sentence_id: $sid, text: $text, embedding_testmodel: $vector
            })
            """,
            sid=sid, text=f"seed sentence {sid}", vector=SEED_VECTOR,
        )

    await session.run(
        f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS "
        "FOR (s:Sentence) ON (s.embedding_testmodel) "
        "OPTIONS {indexConfig: {"
        "`vector.dimensions`: 2, `vector.similarity_function`: 'cosine'}}"
    )
    await session.run("CALL db.awaitIndexes()")

    await session.run(
        "CREATE INDEX sentence_sentence_id IF NOT EXISTS FOR (s:Sentence) ON (s.sentence_id)"
    )
    await session.run(
        "CREATE INDEX sentence_lookup IF NOT EXISTS "
        "FOR (s:Sentence) ON (s.sentence_id, s.event_version)"
    )
    await session.run(
        "CREATE INDEX sentence_sequence IF NOT EXISTS FOR (s:Sentence) ON (s.event_version)"
    )


async def _cleanup(session, sentence_ids):
    await session.run(
        "MATCH (s:Fragment) WHERE s.sentence_id IN $sids DETACH DELETE s",
        sids=sentence_ids,
    )
    await session.run(f"DROP INDEX {INDEX_NAME} IF EXISTS")
    # btree shim indexes are gone permanently post-migration (that's the
    # point) -- drop defensively in case migrate() wasn't reached.
    for name in BTREE_SHIM_INDEXES:
        await session.run(f"DROP INDEX {name} IF EXISTS")


@pytest.mark.asyncio
async def test_migrate_shim_drop_relabels_retargets_index_and_is_idempotent():
    sentence_ids = [str(uuid_mod.uuid4()) for _ in range(3)]

    async with await Neo4jConnectionManager.get_session() as session:
        try:
            await _seed_pre_drop_world(session, sentence_ids)

            # --- First run: migrates the seeded pre-drop world -------------
            summary = await migrate(session)

            assert summary["relabeled"] >= len(sentence_ids)
            assert INDEX_NAME in summary["vector_indexes"]
            assert summary["btree_dropped"] == BTREE_SHIM_INDEXES

            # Seeded nodes lost :Sentence, kept :Fragment. Scoped to seeded
            # ids -- the shared test DB may hold other suites' :Sentence
            # nodes (if any), so we do not assert a global count of 0.
            labels_result = await session.run(
                """
                MATCH (s:Fragment) WHERE s.sentence_id IN $sids
                RETURN count(s) AS total,
                       count(CASE WHEN s:Sentence THEN 1 END) AS still_dual
                """,
                sids=sentence_ids,
            )
            labels_record = await labels_result.single()
            assert labels_record["total"] == len(sentence_ids)
            assert labels_record["still_dual"] == 0

            # SHOW INDEXES: fragment_embedding_testmodel now targets Fragment.
            # Tolerate membership (other suites' fragment_embedding_* indexes
            # may also have been discovered/retargeted by this same run --
            # that's correct migration behavior, not test noise).
            show_result = await session.run("SHOW INDEXES")
            index_rows = {row["name"]: row async for row in show_result}
            assert INDEX_NAME in index_rows
            assert index_rows[INDEX_NAME]["labelsOrTypes"] == ["Fragment"]

            # Repopulation proof: the vector index answers a query for the
            # seeded vector with the seeded node, now reachable via :Fragment.
            query_result = await session.run(
                "CALL db.index.vector.queryNodes($index_name, 1, $vector) "
                "YIELD node RETURN node.sentence_id AS sentence_id",
                index_name=INDEX_NAME, vector=SEED_VECTOR,
            )
            hit = await query_result.single()
            assert hit is not None, "repopulated index returned no hits"
            assert hit["sentence_id"] in sentence_ids

            # --- Second run: idempotent, no more relabeling -----------------
            second_summary = await migrate(session)
            assert second_summary["relabeled"] == 0
        finally:
            await _cleanup(session, sentence_ids)
