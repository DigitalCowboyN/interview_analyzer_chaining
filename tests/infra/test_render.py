from tools.infra.render import render_index
from tools.infra.reader import load_services, load_env_vars


def test_index_lists_services_by_kind():
    out = render_index(load_services("."), load_env_vars("."),
                       runs=[], talks_to=[], requires=[], configured_by=[])
    assert "app" in out and "neo4j" in out
    assert "code" in out and "backing" in out
