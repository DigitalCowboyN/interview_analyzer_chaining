"""tests/projections/test_migrate_shim_drop.py

Pins the statement set/order for the one-shot :Sentence shim-drop migration:
1. Discover stale fragment_embedding_* vector indexes still on :Sentence.
2. Drop + recreate each on :Fragment (same DDL shape as the lazy ensure).
3. db.awaitIndexes() so the recreated indexes are usable before the label
   strip empties the old :Sentence-anchored ones.
4. Strip the :Sentence label in batches (new CALL (s) {...} subquery form).
5. Drop the three shim btree indexes.
"""

import pytest

from src.projections.migrate_shim_drop import migrate
from tests.projections.conftest import FakeSession

BTREE_SHIM_INDEXES = ["sentence_sentence_id", "sentence_lookup", "sentence_sequence"]


def _index_row(name="fragment_embedding_testmodel", dim=768):
    return {
        "name": name,
        "type": "VECTOR",
        "labelsOrTypes": ["Sentence"],
        "properties": ["embedding_testmodel"],
        "options": {
            "indexConfig": {
                "vector.dimensions": dim,
                "vector.similarity_function": "cosine",
            }
        },
    }


@pytest.mark.asyncio
async def test_discovers_only_stale_sentence_fragment_embedding_vector_indexes():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [
                _index_row("fragment_embedding_testmodel"),
                # not a vector index -> excluded
                {
                    "name": "fragment_sentence_id",
                    "type": "RANGE",
                    "labelsOrTypes": ["Fragment"],
                    "properties": ["sentence_id"],
                    "options": {},
                },
                # vector but already on :Fragment -> excluded
                {
                    "name": "fragment_embedding_other",
                    "type": "VECTOR",
                    "labelsOrTypes": ["Fragment"],
                    "properties": ["embedding_other"],
                    "options": {
                        "indexConfig": {
                            "vector.dimensions": 5,
                            "vector.similarity_function": "cosine",
                        }
                    },
                },
                # vector, :Sentence, but not a fragment_embedding_ index -> excluded
                {
                    "name": "utterance_embedding_testmodel",
                    "type": "VECTOR",
                    "labelsOrTypes": ["Sentence"],
                    "properties": ["embedding_testmodel"],
                    "options": {
                        "indexConfig": {
                            "vector.dimensions": 3,
                            "vector.similarity_function": "cosine",
                        }
                    },
                },
            ],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 0}],
        }
    )
    await migrate(session)
    show_indexes_calls = [
        q for q in session.queries if q.strip().startswith("SHOW INDEXES")
    ]
    assert len(show_indexes_calls) == 1
    # Regression guard: bare `SHOW INDEXES` doesn't return an `options`
    # column on Neo4j 5.26 (confirmed against the live test DB) -- it must
    # be explicitly YIELDed or `_recreate_vector_index_on_fragment` KeyErrors
    # reading row["options"].
    assert "YIELD" in show_indexes_calls[0]
    assert "options" in show_indexes_calls[0]
    drop_stale = [
        q for q in session.queries if "DROP INDEX fragment_embedding_testmodel" in q
    ]
    assert len(drop_stale) == 1
    assert not any("fragment_sentence_id" in q and "DROP" in q for q in session.queries)
    assert not any("fragment_embedding_other" in q for q in session.queries)
    assert not any("utterance_embedding_testmodel" in q for q in session.queries)


@pytest.mark.asyncio
async def test_recreates_discovered_vector_index_on_fragment_matching_ensure_ddl_shape():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [_index_row("fragment_embedding_testmodel", dim=768)],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 0}],
        }
    )
    result = await migrate(session)

    drop_idx = next(
        i
        for i, q in enumerate(session.queries)
        if "DROP INDEX fragment_embedding_testmodel" in q
    )
    create_idx = next(
        i
        for i, q in enumerate(session.queries)
        if "CREATE VECTOR INDEX fragment_embedding_testmodel" in q
    )
    assert drop_idx < create_idx
    assert "IF EXISTS" in session.queries[drop_idx]

    create_query = session.queries[create_idx]
    assert "IF NOT EXISTS" in create_query
    assert "FOR (f:Fragment) ON (f.embedding_testmodel)" in create_query or (
        "FOR (n:Fragment) ON n.embedding_testmodel" in create_query
    )
    assert "vector.dimensions" in create_query
    assert "vector.similarity_function" in create_query
    assert "cosine" in create_query

    assert result["vector_indexes"] == ["fragment_embedding_testmodel"]


@pytest.mark.asyncio
async def test_awaits_indexes_after_recreate_and_before_label_strip():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [_index_row("fragment_embedding_testmodel")],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 5}],
        }
    )
    await migrate(session)

    create_idx = next(
        i
        for i, q in enumerate(session.queries)
        if "CREATE VECTOR INDEX fragment_embedding_testmodel" in q
    )
    await_idx = next(
        i
        for i, q in enumerate(session.queries)
        if q.strip() == "CALL db.awaitIndexes()"
    )
    strip_idx = next(
        i for i, q in enumerate(session.queries) if "REMOVE s:Sentence" in q
    )

    assert create_idx < await_idx < strip_idx


@pytest.mark.asyncio
async def test_strips_sentence_label_via_new_subquery_syntax_and_reports_count():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 42}],
        }
    )
    result = await migrate(session)

    strip_query = next(q for q in session.queries if "REMOVE s:Sentence" in q)
    assert "MATCH (s:Sentence)" in strip_query
    assert "CALL (s) {" in strip_query
    assert "IN TRANSACTIONS OF 1000 ROWS" in strip_query
    # the deprecated form must NOT appear
    assert "CALL {\n" not in strip_query
    assert "WITH s" not in strip_query or "CALL (s)" in strip_query

    assert result["relabeled"] == 42


@pytest.mark.asyncio
async def test_drops_the_three_shim_btree_indexes():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 0}],
        }
    )
    result = await migrate(session)

    for name in BTREE_SHIM_INDEXES:
        drops = [
            q for q in session.queries if f"DROP INDEX {name}" in q and "IF EXISTS" in q
        ]
        assert len(drops) == 1, f"expected exactly one drop for {name}"
    assert result["btree_dropped"] == BTREE_SHIM_INDEXES


@pytest.mark.asyncio
async def test_idempotent_second_run_reports_zeros_and_skips_create_and_strip():
    session = FakeSession(
        query_rows={
            "SHOW INDEXES": [],
            "MATCH (s:Sentence) RETURN count(s)": [{"count(s)": 0}],
        }
    )
    result = await migrate(session)

    assert result == {
        "relabeled": 0,
        "vector_indexes": [],
        "btree_dropped": BTREE_SHIM_INDEXES,
    }
    assert not any("CREATE VECTOR INDEX" in q for q in session.queries)
    assert not any("REMOVE s:Sentence" in q for q in session.queries)
    # btree drops are IF EXISTS -- always issued, harmless
    assert any("DROP INDEX sentence_sentence_id" in q for q in session.queries)
