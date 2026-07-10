from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def client():
    return TestClient(app)


def fake_export(tmp_path):
    async def _export(self, interview_id, lens_name, out_dir="exports", zip_bundle=False):
        from src.export.bundler import ExportResult
        zip_path = tmp_path / f"{interview_id}-{lens_name}.zip"
        zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
        return ExportResult(interview_id=interview_id, lens=lens_name, lens_version=1,
                            bundle_path=str(zip_path), files_written=1,
                            items=0, claims=0, entities=0)
    return _export


def test_export_returns_zip(client, tmp_path):
    with patch("src.api.routers.exports.OkfExporter.export", new=fake_export(tmp_path)):
        resp = client.get(f"/exports/{IID}/meeting_minutes")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]


@pytest.mark.parametrize("exc,status", [
    (ValueError(f"Interview {IID} not found"), 404),
    (ValueError("Unknown lens: nope"), 422),
    (RuntimeError("projection lag: retry shortly"), 409),
])
def test_export_error_mapping(client, exc, status):
    with patch("src.api.routers.exports.OkfExporter.export", new=AsyncMock(side_effect=exc)):
        resp = client.get(f"/exports/{IID}/meeting_minutes")
    assert resp.status_code == status
