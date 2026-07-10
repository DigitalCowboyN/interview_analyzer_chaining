"""All Layer 5 Cypher: the exporter and the query endpoints read through here.

Every function takes an active async Neo4j session and returns plain dicts —
no rendering, no domain objects, no session management.
"""

from typing import Any, Dict, List, Optional


async def transcript_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
    OPTIONAL MATCH (s)-[:SPOKEN_BY]->(sp:Speaker)
    OPTIONAL MATCH (s)-[:PART_OF_UTTERANCE]->(u:Utterance)
    RETURN s.sentence_id AS sentence_id, s.sequence_order AS sequence_order,
           s.text AS text, sp.speaker_id AS speaker_id,
           sp.display_name AS speaker, u.utterance_id AS utterance_id
    ORDER BY s.sequence_order
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def speaker_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_PARTICIPANT]->(sp:Speaker)
    WHERE sp.merged_into IS NULL
    RETURN sp.speaker_id AS speaker_id, sp.handle AS handle,
           sp.display_name AS display_name, sp.provisional AS provisional
    ORDER BY sp.handle
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def lens_item_rows(
    session,
    interview_id: str,
    lens: str,
    node_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    query = """
    MATCH (n:LensItem {interview_id: $interview_id, lens: $lens})
    WHERE ($node_type IS NULL OR n.node_type = $node_type)
      AND ($min_confidence IS NULL OR n.confidence >= $min_confidence)
    OPTIONAL MATCH (n)-[r]->(sp:Speaker)
    OPTIONAL MATCH (n)-[:SUPPORTED_BY]->(s:Sentence)
    WITH n,
         [x IN collect(DISTINCT {relationship: type(r), speaker_id: sp.speaker_id,
                                 display_name: sp.display_name})
          WHERE x.speaker_id IS NOT NULL] AS speaker_links,
         [x IN collect(DISTINCT s.aggregate_id) WHERE x IS NOT NULL] AS supporting
    RETURN n.item_id AS item_id, n.node_type AS node_type,
           n.lens_version AS lens_version, n.confidence AS confidence,
           n.model AS model, n.provider AS provider,
           coalesce(n.locked, false) AS locked, properties(n) AS props,
           speaker_links, supporting AS supporting_fragment_ids
    ORDER BY n.node_type, n.item_id
    SKIP $offset LIMIT $limit
    """
    result = await session.run(
        query, interview_id=interview_id, lens=lens, node_type=node_type,
        min_confidence=min_confidence, offset=offset,
        limit=limit if limit is not None else 10_000,
    )
    return [dict(r) async for r in result]


async def claim_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (c:Claim {interview_id: $interview_id})
    OPTIONAL MATCH (c)-[:MADE_BY]->(sp:Speaker)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Sentence)
    WITH c, sp, [x IN collect(DISTINCT s.aggregate_id) WHERE x IS NOT NULL] AS supporting
    RETURN c.claim_id AS claim_id, c.text AS text, c.kind AS kind,
           c.confidence AS confidence, c.model AS model, c.provider AS provider,
           sp.speaker_id AS speaker_id, sp.display_name AS speaker,
           supporting AS supporting_fragment_ids
    ORDER BY c.claim_id
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def entity_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)-[m:MENTIONS]->(e:Entity)
    WITH e, collect({sentence_id: s.aggregate_id, start: m.start, end: m.end,
                      text: m.text, confidence: m.confidence}) AS mentions
    RETURN e.surface AS surface, e.entity_type AS entity_type, mentions
    ORDER BY e.surface
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def analysis_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
    OPTIONAL MATCH (s)-[:SPOKEN_BY]->(sp:Speaker)
    OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
    WITH s, sp, a ORDER BY a.created_at DESC
    WITH s, sp, collect(a)[0] AS a
    OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(ft:FunctionType)
    OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
    OPTIONAL MATCH (a)-[:HAS_PURPOSE]->(p:Purpose)
    OPTIONAL MATCH (a)-[:MENTIONS_TOPIC]->(t:Topic)
    OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
    RETURN s.sequence_order AS sequence_order, s.text AS text,
           sp.display_name AS speaker,
           ft.name AS function, st.name AS structure, p.name AS purpose,
           [x IN collect(DISTINCT t.name) WHERE x IS NOT NULL] AS topics,
           [x IN collect(DISTINCT k.text) WHERE x IS NOT NULL] AS keywords,
           a.confidence AS confidence, a.flags AS flags
    ORDER BY s.sequence_order
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]
