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
    """Records every query string passed to run(), in call order."""

    def __init__(self, rows=None):
        self._rows = rows or []
        self.queries = []

    async def run(self, query, **params):
        self.queries.append(query)
        return FakeResult(self._rows)
