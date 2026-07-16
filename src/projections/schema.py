"""Authoritative Neo4j schema DDL (M4.7).

One idempotent list covering every MERGE key the projection handlers use,
plus the pre-existing documented script entries. Applied at projection-service
startup and via `python -m src.projections.ensure_schema`.

Composite indexes do not serve single-property MERGE lookups, so every MERGE
anchor gets its own single-property index even where a documented composite
index exists. The `:Sentence` entries are the shim-window write anchors and
drop together with the shim label (see ROADMAP backlog).
"""

from typing import Dict, List

SCHEMA_DDL: List[str] = [
    "CREATE CONSTRAINT source_file_filename IF NOT EXISTS "
    "FOR (sf:SourceFile) REQUIRE sf.filename IS UNIQUE",
    # frozen-wire write anchors (single property — MERGE uses these alone)
    "CREATE INDEX sentence_sentence_id IF NOT EXISTS FOR (s:Sentence) ON (s.sentence_id)",
    "CREATE INDEX fragment_sentence_id IF NOT EXISTS FOR (f:Fragment) ON (f.sentence_id)",
    # documented composite read-path indexes (kept)
    "CREATE INDEX sentence_lookup IF NOT EXISTS FOR (s:Sentence) ON (s.sentence_id, s.filename)",
    "CREATE INDEX sentence_sequence IF NOT EXISTS FOR (s:Sentence) ON (s.filename, s.sequence_order)",
    "CREATE INDEX fragment_lookup IF NOT EXISTS FOR (f:Fragment) ON (f.sentence_id, f.filename)",
    "CREATE INDEX fragment_sequence IF NOT EXISTS FOR (f:Fragment) ON (f.filename, f.sequence_order)",
    # documented dimension-node indexes (kept)
    "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
    "CREATE INDEX keyword_text IF NOT EXISTS FOR (k:Keyword) ON (k.text)",
    "CREATE INDEX function_type_name IF NOT EXISTS FOR (ft:FunctionType) ON (ft.name)",
    "CREATE INDEX structure_type_name IF NOT EXISTS FOR (st:StructureType) ON (st.name)",
    "CREATE INDEX purpose_name IF NOT EXISTS FOR (p:Purpose) ON (p.name)",
    # handler MERGE keys without any index until now
    "CREATE INDEX interview_interview_id IF NOT EXISTS FOR (i:Interview) ON (i.interview_id)",
    "CREATE INDEX speaker_speaker_id IF NOT EXISTS FOR (sp:Speaker) ON (sp.speaker_id)",
    "CREATE INDEX utterance_utterance_id IF NOT EXISTS FOR (u:Utterance) ON (u.utterance_id)",
    "CREATE INDEX claim_claim_id IF NOT EXISTS FOR (c:Claim) ON (c.claim_id)",
    "CREATE INDEX entity_surface_type IF NOT EXISTS FOR (e:Entity) ON (e.surface, e.entity_type)",
    "CREATE INDEX project_project_id IF NOT EXISTS FOR (p:Project) ON (p.project_id)",
    # Layer 4 overlay keys
    "CREATE INDEX canonical_entity_canonical_id IF NOT EXISTS FOR (c:CanonicalEntity) ON (c.canonical_id)",
    "CREATE INDEX person_person_id IF NOT EXISTS FOR (p:Person) ON (p.person_id)",
    "CREATE INDEX segment_segment_id IF NOT EXISTS FOR (s:Segment) ON (s.segment_id)",
    "CREATE INDEX lens_item_item_id IF NOT EXISTS FOR (n:LensItem) ON (n.item_id)",
]


async def ensure_schema(session) -> Dict[str, int]:
    """Apply every DDL statement (all idempotent), then await population."""
    for statement in SCHEMA_DDL:
        await session.run(statement)
    await session.run("CALL db.awaitIndexes()")
    return {"statements": len(SCHEMA_DDL)}
