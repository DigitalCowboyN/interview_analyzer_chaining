"""Apply the authoritative Neo4j schema DDL (M4.7).

Usage: python -m src.projections.ensure_schema
Safe to re-run (every statement is `IF NOT EXISTS`). Also applied
automatically at projection-service startup; see src/run_projection_service.py.
"""

import asyncio
import json

from src.projections.schema import ensure_schema
from src.utils.neo4j_driver import Neo4jConnectionManager


async def main() -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        result = await ensure_schema(session)
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
