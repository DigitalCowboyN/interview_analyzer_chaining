from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.main import app


def test_trigger_analysis_runs_ingest_then_enrich(tmp_path: Path):
    input_file = tmp_path / "meeting.txt"
    input_file.write_text("Alice: Hi.\nBob: Hello.\n")

    ingest_result = MagicMock(interview_id="abc")
    ingest_instance = MagicMock()
    ingest_instance.ingest_file = AsyncMock(return_value=ingest_result)
    enrich_instance = MagicMock()
    enrich_instance.enrich_interview = AsyncMock(return_value=MagicMock())

    with patch("src.api.routers.analysis.IngestionOrchestrator", return_value=ingest_instance) as ing_cls, \
         patch("src.api.routers.analysis.EnrichmentOrchestrator", return_value=enrich_instance), \
         patch("src.api.routers.analysis.config", {"paths": {"input_dir": str(tmp_path), "map_dir": str(tmp_path)}}):
        client = TestClient(app)
        resp = client.post("/analysis/", json={"input_filename": "meeting.txt"})

    assert resp.status_code == 202
    ingest_instance.ingest_file.assert_awaited_once()
    enrich_instance.enrich_interview.assert_awaited_once_with("abc")
    ing_cls.assert_called_once()


def test_trigger_analysis_rejects_path_traversal():
    with patch("src.api.routers.analysis.config", {"paths": {"input_dir": "/tmp", "map_dir": "/tmp"}}):
        client = TestClient(app)
        resp = client.post("/analysis/", json={"input_filename": "../etc/passwd"})
    assert resp.status_code == 400


def test_trigger_analysis_404_when_file_missing(tmp_path: Path):
    with patch("src.api.routers.analysis.config", {"paths": {"input_dir": str(tmp_path), "map_dir": str(tmp_path)}}):
        client = TestClient(app)
        resp = client.post("/analysis/", json={"input_filename": "nope.txt"})
    assert resp.status_code == 404
