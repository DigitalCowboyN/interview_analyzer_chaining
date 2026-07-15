"""All Layer 5 Cypher: the exporter and the query endpoints read through here.

Every function takes an active async Neo4j session and returns plain dicts —
no rendering, no domain objects, no session management.
"""

from typing import Any, Dict, List, Optional


async def transcript_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Fragment)
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
    OPTIONAL MATCH (n)-[:SUPPORTED_BY]->(s:Fragment)
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
        # Unbounded callers (the bundler) get a 10k safety cap, not "no limit":
        # an interview with >10k lens items for this lens would be silently
        # truncated here, which then surfaces upstream in the bundler's
        # OkfExporter._guard as a false "projection lag" (expected != projected)
        # rather than the real cause (truncated read).
        limit=limit if limit is not None else 10_000,
    )
    return [dict(r) async for r in result]


async def claim_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (c:Claim {interview_id: $interview_id})
    OPTIONAL MATCH (c)-[:MADE_BY]->(sp:Speaker)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Fragment)
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
    MATCH (proj:Project)-[:CONTAINS_INTERVIEW]->
          (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Fragment)-[m:MENTIONS]->(e:Entity)
    WITH proj, e, collect({sentence_id: s.aggregate_id, start: m.start, end: m.end,
                      text: m.text, confidence: m.confidence}) AS mentions
    OPTIONAL MATCH (e)-[a:ALIAS_OF]->(c:CanonicalEntity)
    WHERE a.project_id = proj.project_id AND c.merged_into IS NULL
    RETURN e.surface AS surface, e.entity_type AS entity_type, mentions,
           c.canonical_id AS canonical_id, c.name AS canonical_name
    ORDER BY e.surface
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def person_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    """Persons identified for this interview's speakers."""
    query = """
    MATCH (:Interview {interview_id: $interview_id})-[:HAS_PARTICIPANT]->
          (sp:Speaker)-[:IDENTIFIED_AS]->(p:Person)
    RETURN sp.speaker_id AS speaker_id, p.person_id AS person_id,
           p.display_name AS display_name
    ORDER BY speaker_id
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(record) async for record in result]


async def worklist_rows(
    session,
    project_id: Optional[str] = None,
    threshold: float = 0.7,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    lens_item_query = """
    MATCH (n:LensItem)
    WHERE ($project_id IS NULL OR EXISTS {
        MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
              (:Interview {interview_id: n.interview_id}) })
      AND (n.confidence < $threshold
           OR any(k IN keys(n) WHERE k ENDS WITH '_unresolved'))
    RETURN n.interview_id AS interview_id, n.item_id AS item_id,
           n.node_type AS node_type, n.lens AS lens, n.confidence AS confidence,
           CASE WHEN n.confidence < $threshold THEN 'low_confidence'
                ELSE 'unresolved_reference' END AS reason
    ORDER BY n.confidence ASC SKIP $offset LIMIT $limit
    """
    claim_query = """
    MATCH (c:Claim)
    WHERE ($project_id IS NULL OR EXISTS {
        MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
              (:Interview {interview_id: c.interview_id}) })
      AND c.confidence < $threshold
    RETURN c.interview_id AS interview_id, c.claim_id AS claim_id,
           c.text AS text, c.kind AS kind, c.confidence AS confidence,
           'low_confidence' AS reason
    ORDER BY c.confidence ASC SKIP $offset LIMIT $limit
    """
    lens_result = await session.run(
        lens_item_query, project_id=project_id, threshold=threshold,
        offset=offset, limit=limit,
    )
    lens_items = [dict(r) async for r in lens_result]
    claim_result = await session.run(
        claim_query, project_id=project_id, threshold=threshold,
        offset=offset, limit=limit,
    )
    claims = [dict(r) async for r in claim_result]
    return {"lens_items": lens_items, "claims": claims}


def _group_rollup_row(
    groups: Dict[str, Dict[str, Any]],
    display_name: str,
    person_id: Optional[str],
    person_name: Optional[str],
) -> Dict[str, Any]:
    key = person_id or f"name:{display_name.lower()}"
    return groups.setdefault(key, {
        "display_name": person_name or display_name,
        "linked": person_id is not None,
        "person_id": person_id,
        "items": [],
        "claims": [],
    })


async def speaker_rollup_rows(
    session,
    project_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    scan_cap: int = 5000,
) -> List[Dict[str, Any]]:
    # bounded scan: grouping/pagination happen in Python; raise scan_cap for very large projects
    items_query = """
    MATCH (n:LensItem)-[r]->(sp:Speaker)
    WHERE sp.merged_into IS NULL AND type(r) <> 'SUPPORTED_BY'
      AND ($project_id IS NULL OR EXISTS {
        MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
              (:Interview {interview_id: n.interview_id}) })
    OPTIONAL MATCH (sp)-[:IDENTIFIED_AS]->(person:Person)
    RETURN sp.display_name AS display_name, n.node_type AS node_type,
           type(r) AS relationship, n.text AS text,
           n.interview_id AS interview_id, n.item_id AS item_id,
           person.person_id AS person_id, person.display_name AS person_name
    LIMIT $scan_cap
    """
    claims_query = """
    MATCH (c:Claim)-[:MADE_BY]->(sp:Speaker)
    WHERE sp.merged_into IS NULL
      AND ($project_id IS NULL OR EXISTS {
        MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
              (:Interview {interview_id: c.interview_id}) })
    OPTIONAL MATCH (sp)-[:IDENTIFIED_AS]->(person:Person)
    RETURN sp.display_name AS display_name, c.text AS text, c.kind AS kind,
           c.interview_id AS interview_id, c.claim_id AS claim_id,
           person.person_id AS person_id, person.display_name AS person_name
    LIMIT $scan_cap
    """
    groups: Dict[str, Dict[str, Any]] = {}

    items_result = await session.run(items_query, project_id=project_id, scan_cap=scan_cap)
    async for r in items_result:
        row = dict(r)
        display_name = row.pop("display_name")
        person_id = row.pop("person_id")
        person_name = row.pop("person_name")
        _group_rollup_row(groups, display_name, person_id, person_name)["items"].append(row)

    claims_result = await session.run(claims_query, project_id=project_id, scan_cap=scan_cap)
    async for r in claims_result:
        row = dict(r)
        display_name = row.pop("display_name")
        person_id = row.pop("person_id")
        person_name = row.pop("person_name")
        _group_rollup_row(groups, display_name, person_id, person_name)["claims"].append(row)

    ordered = sorted(groups.values(), key=lambda g: g["display_name"])
    if name is not None:
        needle = name.lower()
        ordered = [g for g in ordered if needle in g["display_name"].lower()]
    return ordered[offset:offset + limit]


async def analysis_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Fragment)
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
