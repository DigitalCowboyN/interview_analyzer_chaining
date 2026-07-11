"""One-shot idempotent migration: add :Fragment to every :Sentence node.

Usage: python -m src.projections.migrate_fragment_label
Safe to re-run (relabels 0 the second time). The :Sentence label is retained
through M4.5 as a deprecation shim; wire format (event/stream names) is frozen.
"""

import asyncio
import json

from src.utils.neo4j_driver import Neo4jConnectionManager

QUERY = """
MATCH (s:Sentence)
WHERE NOT s:Fragment
CALL {
    WITH s
    SET s:Fragment
} IN TRANSACTIONS OF 1000 ROWS
RETURN count(s) AS relabeled
"""


async def migrate(session) -> int:
    result = await session.run(QUERY)
    record = await result.single()
    return record["relabeled"] if record else 0


async def main() -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        relabeled = await migrate(session)
    print(json.dumps({"relabeled": relabeled}))


if __name__ == "__main__":
    asyncio.run(main())
