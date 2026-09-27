"""Dev-only read-model cleanup of test-run projects (ADR-0030).

Deletes a project's Neo4j subgraph: the Project, its Interviews, their
Fragments (and their Analyses)/Speakers/Utterances, and every node
carrying one of those interview_ids (LensItem, Segment, Claim, ...),
and every node carrying the project_id (CanonicalEntity, Person). Like
the integration tests' own teardowns this is test/dev tooling, not
application code, so ADR-0003's "projection service is the sole writer"
still holds for the running system.

Caveat: ESDB events are untouched — a projection replay would restore
purged projects. Acceptable for dev.

    python -m tools.dev.purge --test-projects [--dry-run]
    python -m tools.dev.purge --project ledgerline-demo
"""

import argparse
import asyncio
from typing import Iterable, List

from src.ui.project_kind import classify
from src.utils.neo4j_driver import Neo4jConnectionManager

_PURGE_QUERY = """
MATCH (p:Project {project_id: $project_id})
OPTIONAL MATCH (p)-[:CONTAINS_INTERVIEW]->(i:Interview)
WITH p, collect(DISTINCT i) AS interviews,
     [x IN collect(DISTINCT i.interview_id) WHERE x IS NOT NULL] AS iids
OPTIONAL MATCH (x) WHERE x.interview_id IN iids OR x.project_id = $project_id
WITH p, interviews, iids, collect(DISTINCT x) AS scoped
UNWIND (interviews + [null]) AS i
OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
OPTIONAL MATCH (i)-[:HAS_PARTICIPANT]->(sp:Speaker)
OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(u:Utterance)
OPTIONAL MATCH (f)-[:HAS_ANALYSIS]->(a:Analysis)
WITH p, interviews, scoped,
     collect(DISTINCT f) + collect(DISTINCT sp) + collect(DISTINCT u)
     + collect(DISTINCT a) AS owned
FOREACH (n IN scoped + owned + interviews | DETACH DELETE n)
DETACH DELETE p
"""


async def purge_project(session, project_id: str) -> None:
    """Delete one project's read-model subgraph."""
    await session.run(_PURGE_QUERY, project_id=project_id)


async def purge_projects(session, project_ids: Iterable[str]) -> int:
    count = 0
    for project_id in project_ids:
        await purge_project(session, project_id)
        count += 1
    return count


async def test_project_ids(session) -> List[str]:
    result = await session.run("MATCH (p:Project) RETURN p.project_id AS project_id")
    ids = [r["project_id"] async for r in result]
    return sorted(pid for pid in ids if classify(pid).kind == "test")


async def _main(args) -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        ids = await test_project_ids(session) if args.test_projects else [args.project]
        if args.dry_run:
            for project_id in ids:
                print(project_id)
            print(f"would purge {len(ids)} project(s)")
            return
        print(f"purged {await purge_projects(session, ids)} project(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test-projects", action="store_true")
    group.add_argument("--project")
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
