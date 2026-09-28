"""Read-time project classification (ADR-0034).

Projects carry no stored name or kind. Integration tests mint one project per
run under a known id prefix, so kind is derived here from the id alone — one
table, no event, no migration. A new test suite with a new prefix must be
added to TEST_PREFIXES or its runs will show up as real projects.
"""

from dataclasses import dataclass
from typing import Literal, Optional

# governed-by: ADR-0034

TEST_RUNS_ID = "test-runs"  # reserved: the UI's single bucket for all test runs

# Matched longest-first, so order here is documentation only.
TEST_PREFIXES: tuple = (
    "deployed-smoke-",
    "projection-smoke-",
    "live-gallery-smoke-",
    "live-feed-smoke-",
    "ui-smoke-",
    "smoke-persona-",
    "smoke-",
)
TEST_EXACT_IDS = frozenset(
    {"test-project", "replay-test-project", "partial-replay-test", TEST_RUNS_ID}
)
_PREFIXES_LONGEST_FIRST = tuple(sorted(TEST_PREFIXES, key=len, reverse=True))


@dataclass(frozen=True)
class ProjectKind:
    kind: Literal["real", "test"]
    suite: Optional[str]


def classify(project_id: str) -> ProjectKind:
    """Classify a project id as a real project or a test run (with its suite)."""
    if project_id in TEST_EXACT_IDS:
        return ProjectKind(kind="test", suite=project_id)
    for prefix in _PREFIXES_LONGEST_FIRST:
        if project_id.startswith(prefix):
            return ProjectKind(kind="test", suite=prefix[:-1])
    return ProjectKind(kind="real", suite=None)
