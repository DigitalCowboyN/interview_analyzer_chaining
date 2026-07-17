"""Shared test fakes for tests/projections."""


class FakeResult:
    """Minimal async-iterable result, matching the ask/export reader test idiom."""

    def __init__(self, rows=None):
        self._rows = rows or []

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for row in self._rows:
            yield row

    async def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """Records every query string passed to run(), in call order.

    `rows` is the default result set for any query. `query_rows` optionally
    maps a query-string prefix to its own canned rows (checked first, longest
    prefix wins) — used to serve distinct results to distinct statements
    (e.g. `SHOW INDEXES` vs. a count query) within the same fake session.
    """

    def __init__(self, rows=None, query_rows=None):
        self._rows = rows or []
        self._query_rows = query_rows or {}
        self.queries = []

    async def run(self, query, **params):
        self.queries.append(query)
        prefix = max(
            (p for p in self._query_rows if query.strip().startswith(p)),
            key=len,
            default=None,
        )
        rows = self._query_rows[prefix] if prefix is not None else self._rows
        return FakeResult(rows)
