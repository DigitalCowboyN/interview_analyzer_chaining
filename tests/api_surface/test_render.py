from tools.api.reader import Endpoint
from tools.api.render import render_catalog

EPS = [
    Endpoint("GET", "/exports/{interview_id}/{lens_name}", "src.api.routers.exports", "Export bundle"),
    Endpoint("GET", "/files/", "src.api.routers.files", "List files"),
    Endpoint("POST", "/files/", "src.api.routers.files", "Ingest a transcript"),
]


def test_render_catalog_groups_by_router():
    out = render_catalog(EPS)
    assert "## src.api.routers.exports" in out
    assert "`GET /exports/{interview_id}/{lens_name}` — Export bundle" in out
    assert "## src.api.routers.files" in out
    assert "`POST /files/` — Ingest a transcript" in out
