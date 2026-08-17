import os

from tools.code.reader import symbols_of


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_symbols_functions_classes_methods(tmp_path):
    _w(str(tmp_path / "src/api/__init__.py"), "")
    _w(str(tmp_path / "src/api/main.py"),
       'def make(x, y=1) -> int:\n    """Build."""\n    return x + y\n\n'
       'class Router:\n    """Routes."""\n    def add(self, path):\n        return path\n')
    by_id = {s.id: s for s in symbols_of("api.main", str(tmp_path))}
    assert by_id["api.main.make"].kind == "function"
    assert by_id["api.main.make"].signature == "make(x, y=1) -> int"
    assert by_id["api.main.make"].docstring == "Build."
    assert by_id["api.main.Router"].kind == "class"
    assert by_id["api.main.Router.add"].kind == "method"
    assert by_id["api.main.Router.add"].parent == "api.main.Router"
    assert by_id["api.main.make"].parent == "api.main"


def test_symbols_without_docstring_are_thin_not_absent(tmp_path):
    _w(str(tmp_path / "src/x/__init__.py"), "")
    _w(str(tmp_path / "src/x/m.py"), "def f(a):\n    return a\n")
    by_id = {s.id: s for s in symbols_of("x.m", str(tmp_path))}
    assert "x.m.f" in by_id                       # exists from the AST
    assert by_id["x.m.f"].docstring == ""         # thin, not absent
    assert by_id["x.m.f"].signature == "f(a)"
