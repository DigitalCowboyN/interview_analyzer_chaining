"""project_kind (ADR-0034): read-time real/test classification of project ids."""

import pytest

from src.ui.project_kind import TEST_RUNS_ID, ProjectKind, classify


@pytest.mark.parametrize(
    "project_id, suite",
    [
        ("deployed-smoke-f4ee7682", "deployed-smoke"),
        ("projection-smoke-c779", "projection-smoke"),
        ("live-gallery-smoke-8f42", "live-gallery-smoke"),
        ("live-feed-smoke-54eb", "live-feed-smoke"),
        ("ui-smoke-1505", "ui-smoke"),
        ("smoke-persona-3885", "smoke-persona"),
        ("smoke-27bd6007", "smoke"),
        ("test-project", "test-project"),
        ("replay-test-project", "replay-test-project"),
        ("partial-replay-test", "partial-replay-test"),
    ],
)
def test_test_projects_classify_with_suite(project_id, suite):
    assert classify(project_id) == ProjectKind(kind="test", suite=suite)


def test_longest_prefix_wins_over_bare_smoke():
    # "smoke-persona-" must not be swallowed by "smoke-".
    assert classify("smoke-persona-x").suite == "smoke-persona"


@pytest.mark.parametrize(
    "project_id",
    [
        "samples",
        "real-interviews",
        "ledgerline-demo",
        "smokehouse-research",
        "my-test-project",
    ],
)
def test_real_projects(project_id):
    assert classify(project_id) == ProjectKind(kind="real", suite=None)


def test_reserved_bucket_id_is_test_kind():
    assert classify(TEST_RUNS_ID).kind == "test"
