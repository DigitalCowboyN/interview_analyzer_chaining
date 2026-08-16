import os

from tools.code.reader import CodeUnit, dep_edges, load_units


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_load_units_is_the_derived_registry(tmp_path):
    _w(str(tmp_path / "src/api/__init__.py"))
    _w(str(tmp_path / "src/api/main.py"), "from src.events import store\n")
    _w(str(tmp_path / "src/events/__init__.py"))
    _w(str(tmp_path / "src/events/store.py"), "x = 1\n")
    by_id = {u.unit: u for u in load_units(str(tmp_path))}
    assert by_id["api"].level == "package"
    assert by_id["api.main"].level == "module"
    assert by_id["api.main"].depends_on == ["events"]   # 'events' is the longest node prefix


def test_dep_edges_are_module_granular(tmp_path):
    _w(str(tmp_path / "src/a/__init__.py"))
    _w(str(tmp_path / "src/a/m.py"), "from src.b import x\n")
    _w(str(tmp_path / "src/b/__init__.py"))
    edges = dep_edges(str(tmp_path))
    assert edges["a.m"] == ["b"]        # keyed by the importing MODULE, not the package
    assert "a" not in edges            # packages carry no depends_on
