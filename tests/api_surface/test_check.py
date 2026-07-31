from tools.api.reader import Endpoint
from tools.api.check import (
    check_openapi_fresh, check_docs_reference_real, check_catalog_in_sync, Finding,
)

EPS = [Endpoint("GET", "/exports/{interview_id}/{lens_name}", "src.api.routers.exports", "Export")]


def test_openapi_fresh_flags_added_and_missing():
    live = {("GET", "/a"), ("GET", "/b")}
    committed = {("GET", "/a")}
    msgs = " ".join(f.message for f in check_openapi_fresh(committed, live))
    assert "GET /b exists in the app but not in" in msgs   # stale: app has it, contract doesn't
    assert check_openapi_fresh(live, live) == []           # in sync -> no findings


def test_openapi_fresh_missing_file():
    msgs = " ".join(f.message for f in check_openapi_fresh(None, {("GET", "/a")}))
    assert "missing" in msgs


def test_docs_reference_real_normalizes_path_params(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("Use `GET /exports/{id}/{lens}` and `POST /gone`. Prose /exports/x ignored.\n", encoding="utf-8")
    msgs = " ".join(f.message for f in check_docs_reference_real(EPS, [str(doc)]))
    assert "POST /gone" in msgs                       # not a real endpoint
    assert "/exports/{id}/{lens}" not in msgs          # param-name difference tolerated -> real


def test_catalog_in_sync(tmp_path):
    from tools.api.render import render_catalog
    cat = tmp_path / "index.md"
    cat.write_text("stale\n", encoding="utf-8")
    assert check_catalog_in_sync(str(cat), EPS)
    cat.write_text(render_catalog(EPS), encoding="utf-8")
    assert check_catalog_in_sync(str(cat), EPS) == []
