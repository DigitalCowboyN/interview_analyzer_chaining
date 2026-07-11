"""ProjectRepository stream naming and factory wiring (M4.5b)."""

from unittest.mock import MagicMock

from src.events.aggregates import Project
from src.events.repository import (
    ProjectRepository,
    RepositoryFactory,
    get_project_repository,
)


class TestProjectRepository:
    def test_stream_name_is_wire_frozen(self):
        repo = ProjectRepository(MagicMock())
        assert repo._get_stream_name("abc-123") == "Project-abc-123"

    def test_creates_project_aggregate(self):
        repo = ProjectRepository(MagicMock())
        aggregate = repo._create_aggregate("abc-123")
        assert isinstance(aggregate, Project)
        assert aggregate.aggregate_id == "abc-123"

    def test_factory_creates_project_repository(self):
        factory = RepositoryFactory(MagicMock())
        assert isinstance(factory.create_project_repository(), ProjectRepository)

    def test_module_getter_returns_repository(self, monkeypatch):
        import src.events.repository as repo_module

        monkeypatch.setattr(repo_module, "_global_factory", RepositoryFactory(MagicMock()))
        assert isinstance(get_project_repository(), ProjectRepository)
