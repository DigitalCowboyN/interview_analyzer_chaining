"""tests/projections/test_schema.py"""
import pytest

from src.projections.schema import SCHEMA_DDL, ensure_schema
from tests.projections.conftest import FakeSession


EXPECTED_SUBSTRINGS = [
    # constraint
    "CREATE CONSTRAINT source_file_filename IF NOT EXISTS",
    # frozen-wire anchors (single-property MERGE keys)
    "FOR (s:Sentence) ON (s.sentence_id)",
    "FOR (f:Fragment) ON (f.sentence_id)",
    # documented existing indexes
    "FOR (t:Topic) ON (t.name)",
    "FOR (k:Keyword) ON (k.text)",
    "FOR (dk:DomainKeyword) ON (dk.text)",
    "FOR (ft:FunctionType) ON (ft.name)",
    "FOR (st:StructureType) ON (st.name)",
    "FOR (p:Purpose) ON (p.name)",
    # handler MERGE keys that never had indexes
    "FOR (i:Interview) ON (i.interview_id)",
    "FOR (sp:Speaker) ON (sp.speaker_id)",
    "FOR (u:Utterance) ON (u.utterance_id)",
    "FOR (c:Claim) ON (c.claim_id)",
    "FOR (e:Entity) ON (e.surface, e.entity_type)",
    "FOR (p:Project) ON (p.project_id)",
    # Layer 4 overlay keys (the backlog item)
    "FOR (c:CanonicalEntity) ON (c.canonical_id)",
    "FOR (p:Person) ON (p.person_id)",
    "FOR (s:Segment) ON (s.segment_id)",
    "FOR (n:LensItem) ON (n.item_id)",
    # fulltext + aggregate_id handler MATCH anchors (final review addition)
    "CREATE FULLTEXT INDEX fragment_text_ft IF NOT EXISTS FOR (f:Fragment) ON EACH [f.text]",
    "CREATE INDEX fragment_aggregate_id IF NOT EXISTS FOR (f:Fragment) ON (f.aggregate_id)",
    "CREATE INDEX interview_aggregate_id IF NOT EXISTS FOR (i:Interview) ON (i.aggregate_id)",
]


def test_ddl_covers_every_merge_key():
    joined = "\n".join(SCHEMA_DDL)
    for fragment in EXPECTED_SUBSTRINGS:
        assert fragment in joined, f"missing DDL: {fragment}"


def test_every_statement_is_idempotent():
    for stmt in SCHEMA_DDL:
        assert "IF NOT EXISTS" in stmt, stmt


@pytest.mark.asyncio
async def test_ensure_schema_runs_all_statements_then_awaits_indexes():
    session = FakeSession()
    result = await ensure_schema(session)
    assert result == {"statements": len(SCHEMA_DDL)}
    assert session.queries[:-1] == list(SCHEMA_DDL)
    assert session.queries[-1] == "CALL db.awaitIndexes()"
