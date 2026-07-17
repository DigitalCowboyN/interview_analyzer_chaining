"""Ask reader: query-text pins for the three channels + context assembly rows."""

import pytest

from src.ask import reader


class FakeResult:
    """Async-iterable result rows, matching the export reader test idiom."""

    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for row in self._rows:
            yield row

    async def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """Adapted from tests/export/test_reader.py's make_session: records every
    query and its params in call order; rows list in, async iteration out."""

    def __init__(self, rows):
        self._rows = rows
        self.queries = []
        self.params = []

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        return FakeResult(self._rows)

    @property
    def last_query(self):
        return self.queries[-1]

    @property
    def last_params(self):
        return self.params[-1]


@pytest.mark.asyncio
async def test_project_exists_counts_project_node():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.project_exists(session, "proj-1") is True
    assert "MATCH (p:Project {project_id: $project_id})" in session.last_query


@pytest.mark.asyncio
async def test_name_rows_unions_live_canonicals_and_persons():
    session = FakeSession(rows=[])
    await reader.name_rows(session, "proj-1")
    q = session.last_query
    assert "c.merged_into IS NULL" in q
    assert "ALIAS_OF" in q and "surfaces" in q
    assert "Person {project_id: $project_id}" in q
    assert "UNION" in q


@pytest.mark.asyncio
async def test_vector_fragment_rows_scopes_to_project():
    session = FakeSession(rows=[{"fragment_id": "f1", "score": 0.9}])
    rows = await reader.vector_fragment_rows(session, "proj-1", "fragment_embedding_m", [0.1], 5)
    assert rows[0]["fragment_id"] == "f1"
    q = session.last_query
    assert "CALL db.index.vector.queryNodes($index_name, $fetch_k, $vector)" in q
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "LIMIT $k" in q
    assert session.last_params["index_name"] == "fragment_embedding_m"
    assert session.last_params["fetch_k"] == 5 * reader.OVERFETCH_FACTOR
    assert session.last_params["k"] == 5


@pytest.mark.asyncio
async def test_vector_utterance_rows_expands_to_member_fragments():
    session = FakeSession(rows=[])
    await reader.vector_utterance_rows(session, "proj-1", "utterance_embedding_m", [0.1], 5)
    q = session.last_query
    assert "PART_OF_UTTERANCE" in q
    assert "f.sentence_id AS fragment_id" in q
    assert "CALL db.index.vector.queryNodes($index_name, $fetch_k, $vector)" in q
    assert "LIMIT $k" in q
    assert session.last_params["fetch_k"] == 5 * reader.OVERFETCH_FACTOR
    assert session.last_params["k"] == 5


@pytest.mark.asyncio
async def test_fulltext_rows_uses_named_index_and_scopes_to_project():
    session = FakeSession(rows=[])
    await reader.fulltext_rows(session, "proj-1", "vendor timeline", 5)
    q = session.last_query
    assert "db.index.fulltext.queryNodes('fragment_text_ft'" in q
    assert "{limit: $fetch_k}" in q
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "LIMIT $k" in q
    assert session.last_params["fetch_k"] == 5 * reader.OVERFETCH_FACTOR
    assert session.last_params["k"] == 5


def test_sanitize_fulltext_query_strips_lucene_specials():
    assert reader.sanitize_fulltext_query('who said "X+Y" (really)?') == "who said X Y really"
    assert reader.sanitize_fulltext_query("///***") == ""


def test_sanitize_keeps_unicode_word_chars():
    assert reader.sanitize_fulltext_query("Où est Müller?") == "Où est Müller"
    assert reader.sanitize_fulltext_query("café + résumé") == "café résumé"


def test_sanitize_still_strips_lucene_specials():
    result = reader.sanitize_fulltext_query('a:b AND (c || d) "e"~2^')
    assert result == "a b AND c d e 2"
    lucene_specials = ['+', '-', '&&', '||', '!', '(', ')', '{', '}',
                       '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']
    for token in lucene_specials:
        assert token not in result


@pytest.mark.asyncio
async def test_graph_anchor_rows_anchors_entities_and_persons():
    session = FakeSession(rows=[])
    await reader.graph_anchor_rows(session, "proj-1", ["c1"], ["p1"])
    q = session.last_query
    assert "c.canonical_id IN $canonical_ids" in q
    assert "p.person_id IN $person_ids" in q
    assert "MENTIONS" in q and "IDENTIFIED_AS" in q and "SPOKEN_BY" in q
    assert "UNION ALL" in q


@pytest.mark.asyncio
async def test_context_rows_carries_speaker_person_segment_entities_siblings():
    session = FakeSession(rows=[])
    await reader.context_rows(session, ["f1", "f2"])
    q = session.last_query
    assert "UNWIND $fragment_ids AS fid" in q
    assert "IDENTIFIED_AS" in q and "Segment" in q and "MENTIONS" in q
    assert "PART_OF_UTTERANCE" in q
    assert "ORDER BY interview_id, sequence_order" in q
    assert "WHERE x.text IS NOT NULL" in q


@pytest.mark.asyncio
async def test_ensure_fulltext_index_is_idempotent_ddl():
    session = FakeSession(rows=[])
    await reader.ensure_fulltext_index(session)
    assert "CREATE FULLTEXT INDEX fragment_text_ft IF NOT EXISTS" in session.queries[0]
    assert "FOR (f:Fragment) ON EACH [f.text]" in session.queries[0]
