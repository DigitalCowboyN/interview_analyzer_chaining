"""UI reader (M5.0): every Cypher query the `/ui/*` read layer runs.

Read-side only, mirrors the ask/export reader idiom — plain async functions,
an active Neo4j session as the first arg, plain dicts out. Every
project-scoped query pins (:Project {project_id})-[:CONTAINS_INTERVIEW]->;
speaker-facing queries filter sp.merged_into IS NULL; persona reads filter
n.lens = 'persona'. Zero writes.
"""

from typing import Any, Dict, List, Optional

PERSONA_LENS = "persona"


async def project_exists(session, project_id: str) -> bool:
    query = "MATCH (p:Project {project_id: $project_id}) RETURN count(p) AS found"
    result = await session.run(query, project_id=project_id)
    record = await result.single()
    return bool(record and record["found"])


async def interview_exists(session, interview_id: str) -> bool:
    query = "MATCH (i:Interview {interview_id: $interview_id}) RETURN count(i) AS found"
    result = await session.run(query, interview_id=interview_id)
    record = await result.single()
    return bool(record and record["found"])


async def person_exists(session, project_id: str, person_id: str) -> bool:
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(:Fragment)-[:SPOKEN_BY]->
          (sp:Speaker)-[:IDENTIFIED_AS]->(p:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN count(p) AS found
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    record = await result.single()
    return bool(record and record["found"])


async def persona_exists(session, project_id: str, person_id: str) -> bool:
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(f:Fragment)
    MATCH (n:LensItem)-[:SUPPORTED_BY]->(f)
    WHERE n.lens = 'persona'
    MATCH (n)-[]->(sp:Speaker)-[:IDENTIFIED_AS]->(p:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN count(DISTINCT n) AS found
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    record = await result.single()
    return bool(record and record["found"])


async def project_rows(session) -> List[Dict[str, Any]]:
    """Every project with its interview count (nav landing)."""
    query = """
    MATCH (p:Project)
    OPTIONAL MATCH (p)-[:CONTAINS_INTERVIEW]->(i:Interview)
    RETURN p.project_id AS project_id, count(DISTINCT i) AS interview_count
    ORDER BY p.project_id
    """
    result = await session.run(query)
    return [dict(r) async for r in result]


async def interview_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Project's interviews with fragment counts."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
    OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
    RETURN i.interview_id AS interview_id, i.title AS title,
           toString(i.created_at) AS created_at, count(f) AS fragment_count
    ORDER BY created_at
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]


async def interview_header_row(session, interview_id: str) -> Optional[Dict[str, Any]]:
    """Interview title + graph-resident metadata (no ESDB aggregate reads here:
    the Interview node carries no front-matter/metadata property today, so
    this returns {} until a projection handler starts writing one)."""
    query = """
    MATCH (i:Interview {interview_id: $interview_id})
    RETURN i.interview_id AS interview_id, i.title AS title,
           coalesce(i.metadata, {}) AS metadata
    """
    result = await session.run(query, interview_id=interview_id)
    record = await result.single()
    return dict(record) if record else None


async def transcript_line_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    """One row per fragment, ordered, with every line-detail relation carried
    and null-stripped via the export-reader idiom."""
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(f:Fragment)
    OPTIONAL MATCH (f)-[:SPOKEN_BY]->(sp:Speaker)
    WHERE sp IS NULL OR sp.merged_into IS NULL
    OPTIONAL MATCH (sp)-[:IDENTIFIED_AS]->(person:Person)
    OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(u:Utterance)
    OPTIONAL MATCH (seg:Segment)-[:CONTAINS]->(f)
    OPTIONAL MATCH (f)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (n:LensItem)-[:SUPPORTED_BY]->(f)
    WITH f, sp, person, u, seg,
         [x IN collect(DISTINCT {surface: e.surface, entity_type: e.entity_type})
          WHERE x.surface IS NOT NULL] AS entities,
         [x IN collect(DISTINCT {item_id: n.item_id, lens: n.lens,
                                  node_type: n.node_type, text: n.text,
                                  confidence: n.confidence,
                                  human_locked: coalesce(n.locked, false)})
          WHERE x.item_id IS NOT NULL] AS lens_items
    RETURN f.sentence_id AS fragment_id, f.sequence_order AS sequence_order,
           f.text AS text, f.is_edited AS edited,
           sp.speaker_id AS speaker_id, sp.display_name AS speaker_display_name,
           person.person_id AS person_id, person.display_name AS person_display_name,
           u.utterance_id AS utterance_id,
           seg.segment_id AS segment_id, seg.topic AS segment_topic,
           entities, lens_items
    ORDER BY f.sequence_order
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def persona_card_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Persona-profile cards: one row per person with persona lens items,
    dimension counts + a representative quote + the interviews they appear in."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_SENTENCE]->(f:Fragment)
    MATCH (n:LensItem)-[:SUPPORTED_BY]->(f)
    WHERE n.lens = 'persona'
    MATCH (n)-[]->(sp:Speaker)-[:IDENTIFIED_AS]->(p:Person)
    WHERE sp.merged_into IS NULL
    WITH p, n, i
    ORDER BY n.confidence DESC
    WITH p,
         count(DISTINCT CASE WHEN n.node_type = 'Trait' THEN n END) AS trait_count,
         count(DISTINCT CASE WHEN n.node_type = 'Goal' THEN n END) AS goal_count,
         count(DISTINCT CASE WHEN n.node_type = 'PainPoint' THEN n END) AS pain_point_count,
         count(DISTINCT CASE WHEN n.node_type = 'NotableQuote' THEN n END) AS quote_count,
         collect(DISTINCT CASE WHEN n.node_type = 'NotableQuote' THEN n.text END)[0]
             AS representative_quote,
         [x IN collect(DISTINCT i.interview_id) WHERE x IS NOT NULL] AS interview_ids
    RETURN p.person_id AS person_id, p.display_name AS display_name,
           trait_count, goal_count, pain_point_count, quote_count,
           representative_quote, interview_ids
    ORDER BY p.display_name
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]


async def persona_detail_rows(session, project_id: str, person_id: str) -> List[Dict[str, Any]]:
    """Persona core view: per-interview provenance for every dimension item."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_SENTENCE]->(f:Fragment)
    MATCH (n:LensItem)-[:SUPPORTED_BY]->(f)
    WHERE n.lens = 'persona'
    MATCH (n)-[]->(sp:Speaker)-[:IDENTIFIED_AS]->(p:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN DISTINCT n.item_id AS item_id, n.node_type AS node_type, n.text AS text,
           n.confidence AS confidence, i.interview_id AS interview_id,
           i.title AS interview_title
    ORDER BY n.node_type, n.confidence DESC
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    return [dict(r) async for r in result]


async def person_display_name_row(session, project_id: str, person_id: str) -> Optional[Dict[str, Any]]:
    """Person's display_name, project-scoped — shared by the persona and
    person core views (both need this one fact beyond their detail rows)."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(:Fragment)-[:SPOKEN_BY]->
          (sp:Speaker)-[:IDENTIFIED_AS]->(p:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN p.person_id AS person_id, p.display_name AS display_name
    LIMIT 1
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    record = await result.single()
    return dict(record) if record else None


async def person_card_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Person cards: identity facts (speaker + interview counts)."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_SENTENCE]->(:Fragment)-[:SPOKEN_BY]->(sp:Speaker)
          -[:IDENTIFIED_AS]->(p:Person)
    WHERE sp.merged_into IS NULL
    RETURN p.person_id AS person_id, p.display_name AS display_name,
           count(DISTINCT sp.speaker_id) AS speaker_count,
           count(DISTINCT i.interview_id) AS interview_count
    ORDER BY p.display_name
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]


async def person_detail_rows(session, project_id: str, person_id: str) -> List[Dict[str, Any]]:
    """Person core view: linked speakers per interview."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_SENTENCE]->(:Fragment)-[:SPOKEN_BY]->(sp:Speaker)
          -[:IDENTIFIED_AS]->(p:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN DISTINCT i.interview_id AS interview_id, i.title AS interview_title,
           sp.speaker_id AS speaker_id, sp.display_name AS speaker_display_name
    ORDER BY i.title, sp.display_name
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    return [dict(r) async for r in result]


async def person_contributes_to_persona(session, project_id: str, person_id: str) -> bool:
    """Whether this person has any persona-lens items (loose link source
    for the person core view's `contributes_to_persona` flag)."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(f:Fragment)
    MATCH (n:LensItem)-[:SUPPORTED_BY]->(f)
    WHERE n.lens = 'persona'
    MATCH (n)-[]->(sp:Speaker)-[:IDENTIFIED_AS]->(:Person {person_id: $person_id})
    WHERE sp.merged_into IS NULL
    RETURN count(n) AS found
    """
    result = await session.run(query, project_id=project_id, person_id=person_id)
    record = await result.single()
    return bool(record and record["found"])
