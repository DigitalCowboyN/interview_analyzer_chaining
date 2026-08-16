import os

from tools.code.reader import CodeUnit, contains_edges, discover_units


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    # a package with a sub-package and modules; a tools package; a top-level src module
    _w(str(tmp / "src/api/__init__.py"), '"""The API surface."""\n')
    _w(str(tmp / "src/api/main.py"), "from src.api.routers import segments\n")
    _w(str(tmp / "src/api/routers/__init__.py"))
    _w(str(tmp / "src/api/routers/segments.py"),
       '"""Segment routes."""\nfrom src.events.store import thing\n')
    _w(str(tmp / "src/events/__init__.py"))
    _w(str(tmp / "src/events/store.py"), "x = 1\n")
    _w(str(tmp / "src/config.py"), '"""Settings."""\n')
    _w(str(tmp / "tools/graph/__init__.py"))
    _w(str(tmp / "tools/graph/traverse.py"), "from src.config import settings\n")


def test_discovers_packages_and_modules_with_level(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    assert by_id["api"].level == "package"
    assert by_id["api.routers"].level == "package"
    assert by_id["api.routers.segments"].level == "module"
    assert by_id["config"].level == "module"          # top-level src/*.py is a module
    assert by_id["tools.graph"].level == "package"
    assert by_id["tools.graph.traverse"].level == "module"
    assert "src" not in by_id and "tools" not in by_id  # roots are not nodes


def test_context_comes_from_docstring(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    assert by_id["api"].description == "The API surface."           # package __init__ docstring
    assert by_id["api.routers.segments"].description == "Segment routes."
    assert by_id["events.store"].description == ""                  # no docstring


def test_module_depends_on_is_dotted_and_longest_prefix(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    # full dotted resolution to the module, not just the top package:
    assert by_id["api.routers.segments"].depends_on == ["events.store"]
    # 'from src.api.routers import segments' resolves to the sub-package (segments is a name here):
    assert by_id["api.main"].depends_on == ["api.routers"]
    # tools module importing a src top-level module resolves to that module:
    assert by_id["tools.graph.traverse"].depends_on == ["config"]
    # packages carry no depends_on
    assert by_id["api"].depends_on == []


def test_contains_edges_form_the_hierarchy(tmp_path):
    _fixture(tmp_path)
    edges = set(contains_edges(str(tmp_path)))
    assert ("api", "api.routers") in edges
    assert ("api.routers", "api.routers.segments") in edges
    assert ("api", "api.main") in edges
    assert ("tools.graph", "tools.graph.traverse") in edges
    # roots (no parent node) appear as no child: 'config' and 'api' are never a child
    assert not any(child == "config" for _, child in edges)
    assert not any(child == "api" for _, child in edges)
