import json
import pytest
from tools.api.reader import Endpoint, committed_pairs, load_app, live_endpoints


def test_committed_pairs_parses_openapi(tmp_path):
    f = tmp_path / "openapi.json"
    f.write_text(json.dumps({"paths": {"/x": {"get": {}, "post": {}}, "/y": {"get": {}}}}), encoding="utf-8")
    assert committed_pairs(str(f)) == {("GET", "/x"), ("POST", "/x"), ("GET", "/y")}


def test_committed_pairs_missing_is_none(tmp_path):
    assert committed_pairs(str(tmp_path / "nope.json")) is None


@pytest.mark.integration
def test_live_endpoints_against_real_app():
    try:
        app = load_app()
    except Exception as e:
        pytest.skip(f"app import unavailable: {e}")
    eps = live_endpoints(app)
    assert len(eps) > 20
    assert all(isinstance(e, Endpoint) for e in eps)
    assert any(e.path == "/exports/{interview_id}/{lens_name}" and e.method == "GET" for e in eps)
