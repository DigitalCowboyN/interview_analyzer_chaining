from unittest.mock import AsyncMock, MagicMock

import pytest

from src.projections.migrate_fragment_label import migrate


@pytest.mark.asyncio
async def test_migrate_runs_batched_relabel_and_reports_count():
    session = MagicMock()
    record = {"relabeled": 42}
    result = MagicMock()
    result.single = AsyncMock(return_value=record)
    session.run = AsyncMock(return_value=result)
    count = await migrate(session)
    query = session.run.call_args[0][0]
    assert "MATCH (s:Sentence)" in query and "WHERE NOT s:Fragment" in query
    assert "SET s:Fragment" in query and "IN TRANSACTIONS" in query
    assert count == 42
