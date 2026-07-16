"""Ask retrieval reader (M4.6) — every Cypher query the ask surface runs.

Read-side only. All functions take an active Neo4j session; project scoping
runs through (:Project)-[:CONTAINS_INTERVIEW]-> in every channel.
"""

import re
from typing import Any, Dict, List

FULLTEXT_INDEX = "fragment_text_ft"


def sanitize_fulltext_query(text: str) -> str:
    """Strip Lucene special characters — the question is user input."""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return " ".join(cleaned.split())


async def ensure_fulltext_index(session) -> None:
    """Lazy, idempotent DDL (vector-index idiom); await population so a
    freshly created index is queryable in the same process."""
    await session.run(
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
        "FOR (f:Fragment) ON EACH [f.text]"
    )
    await session.run("CALL db.awaitIndexes()")


async def project_exists(session, project_id: str) -> bool:
    query = "MATCH (p:Project {project_id: $project_id}) RETURN count(p) AS found"
    result = await session.run(query, project_id=project_id)
    record = await result.single()
    return bool(record and record["found"])


async def name_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Live canonical-entity names/surfaces and person names (query analysis)."""
    query = """
    MATCH (c:CanonicalEntity {project_id: $project_id})
    WHERE c.merged_into IS NULL
    OPTIONAL MATCH (e:Entity)-[:ALIAS_OF {project_id: $project_id}]->(c)
    WITH c, collect(e.surface) AS surfaces
    RETURN 'entity' AS kind, c.canonical_id AS id, c.name AS name, surfaces
    UNION
    MATCH (p:Person {project_id: $project_id})
    RETURN 'person' AS kind, p.person_id AS id, p.display_name AS name,
           [] AS surfaces
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]


async def vector_fragment_rows(
    session, project_id: str, index_name: str, vector: List[float], k: int
) -> List[Dict[str, Any]]:
    query = """
    CALL db.index.vector.queryNodes($index_name, $k, $vector)
    YIELD node, score
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(node)
    RETURN node.sentence_id AS fragment_id, score
    ORDER BY score DESC
    """
    result = await session.run(
        query, index_name=index_name, k=k, vector=vector, project_id=project_id
    )
    return [dict(r) async for r in result]


async def vector_utterance_rows(
    session, project_id: str, index_name: str, vector: List[float], k: int
) -> List[Dict[str, Any]]:
    """Utterance hits expand to member fragments, inheriting the hit's score."""
    query = """
    CALL db.index.vector.queryNodes($index_name, $k, $vector)
    YIELD node, score
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(f:Fragment)-[:PART_OF_UTTERANCE]->(node)
    RETURN f.sentence_id AS fragment_id, score
    ORDER BY score DESC
    """
    result = await session.run(
        query, index_name=index_name, k=k, vector=vector, project_id=project_id
    )
    return [dict(r) async for r in result]


async def fulltext_rows(
    session, project_id: str, query_text: str, k: int
) -> List[Dict[str, Any]]:
    query = f"""
    CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $query_text, {{limit: $k}})
    YIELD node, score
    MATCH (:Project {{project_id: $project_id}})-[:CONTAINS_INTERVIEW]->
          (:Interview)-[:HAS_SENTENCE]->(node)
    RETURN node.sentence_id AS fragment_id, score
    ORDER BY score DESC
    """
    result = await session.run(
        query, query_text=query_text, k=k, project_id=project_id
    )
    return [dict(r) async for r in result]


async def graph_anchor_rows(
    session, project_id: str, canonical_ids: List[str], person_ids: List[str]
) -> List[Dict[str, Any]]:
    """One row per (anchor, fragment) hit — duplicates are the ranking signal."""
    query = """
    MATCH (c:CanonicalEntity)<-[:ALIAS_OF {project_id: $project_id}]-(:Entity)
          <-[:MENTIONS]-(f:Fragment)<-[:HAS_SENTENCE]-(:Interview)
          <-[:CONTAINS_INTERVIEW]-(:Project {project_id: $project_id})
    WHERE c.canonical_id IN $canonical_ids
    RETURN f.sentence_id AS fragment_id
    UNION ALL
    MATCH (p:Person)<-[:IDENTIFIED_AS]-(:Speaker)<-[:SPOKEN_BY]-(f:Fragment)
          <-[:HAS_SENTENCE]-(:Interview)
          <-[:CONTAINS_INTERVIEW]-(:Project {project_id: $project_id})
    WHERE p.person_id IN $person_ids
    RETURN f.sentence_id AS fragment_id
    """
    result = await session.run(
        query, project_id=project_id, canonical_ids=canonical_ids,
        person_ids=person_ids,
    )
    return [dict(r) async for r in result]


async def context_rows(session, fragment_ids: List[str]) -> List[Dict[str, Any]]:
    """Everything the context blocks need, one row per fragment."""
    query = """
    UNWIND $fragment_ids AS fid
    MATCH (f:Fragment {sentence_id: fid})<-[:HAS_SENTENCE]-(i:Interview)
    OPTIONAL MATCH (f)-[:SPOKEN_BY]->(sp:Speaker)
    OPTIONAL MATCH (sp)-[:IDENTIFIED_AS]->(person:Person)
    OPTIONAL MATCH (seg:Segment)-[:CONTAINS]->(f)
    OPTIONAL MATCH (f)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(:Utterance)
                   <-[:PART_OF_UTTERANCE]-(sib:Fragment)
    WHERE sib.sentence_id <> f.sentence_id
    RETURN f.sentence_id AS fragment_id, f.text AS text,
           f.sequence_order AS sequence_order,
           i.interview_id AS interview_id, i.title AS title,
           sp.display_name AS speaker, person.display_name AS person,
           [t IN collect(DISTINCT seg.topic) WHERE t IS NOT NULL] AS segment_topics,
           [s IN collect(DISTINCT e.surface) WHERE s IS NOT NULL] AS entities,
           [x IN collect(DISTINCT {text: sib.text, order: sib.sequence_order})
            WHERE x.text IS NOT NULL] AS siblings
    ORDER BY interview_id, sequence_order
    """
    result = await session.run(query, fragment_ids=fragment_ids)
    return [dict(r) async for r in result]
