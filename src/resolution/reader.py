"""Layer 4 input reads: the resolution engine's own project-scoped queries.

Layer 5's consumer Cypher stays in src/export/reader.py; these queries are
the engine's INPUT surface (spec M4.5b step 1).
"""

from typing import Any, Dict, List


async def entity_surface_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Distinct entity surfaces mentioned anywhere in the project, with counts."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(:Interview)
          -[:HAS_SENTENCE]->(:Fragment)-[m:MENTIONS]->(e:Entity)
    RETURN e.surface AS surface, e.entity_type AS entity_type, count(m) AS mentions
    ORDER BY entity_type, surface
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]


async def speaker_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Live (unmerged) speakers across the project's interviews."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_PARTICIPANT]->(sp:Speaker)
    WHERE sp.merged_into IS NULL
    RETURN i.interview_id AS interview_id, sp.speaker_id AS speaker_id,
           sp.display_name AS display_name, sp.handle AS handle,
           coalesce(sp.provisional, false) AS provisional
    ORDER BY interview_id, speaker_id
    """
    result = await session.run(query, project_id=project_id)
    return [dict(r) async for r in result]
