import os

from tools.code.reader import symbols_of


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    _w(str(tmp / "src/svc/__init__.py"), "")
    _w(str(tmp / "src/svc/render.py"), "def draw(x):\n    return x\n")
    _w(str(tmp / "src/svc/main.py"),
       "from src.svc.render import draw\n\n"
       "def helper():\n    return 1\n\n"
       "def run(obj):\n    helper()\n    draw(3)\n    obj.method()\n")


def test_calls_resolve_local_and_imported(tmp_path):
    _fixture(tmp_path)
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    calls = set(by_id["svc.main.run"].calls)
    assert "svc.main.helper" in calls          # local def
    assert "svc.render.draw" in calls          # imported symbol
    assert not any(c.endswith(".method") for c in calls)   # obj.method() unresolved -> skipped


def test_calls_marker_escape_hatch(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "src/svc/main.py").write_text(
        "def run(obj):\n    # calls: code:svc.render.draw\n    obj.method()\n", encoding="utf-8")
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    assert "svc.render.draw" in by_id["svc.main.run"].calls   # asserted by marker


def test_calls_resolve_relative_import(tmp_path):
    # `from .render import draw` (relative) must resolve against the module's package
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/render.py"), "def draw(x):\n    return x\n")
    _w(str(tmp_path / "src/svc/main.py"),
       "from .render import draw\n\ndef run():\n    draw(3)\n")
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    assert "svc.render.draw" in by_id["svc.main.run"].calls


def test_calls_marker_is_scoped_to_its_own_function(tmp_path):
    # a marker in one function must NOT attach to a sibling function in the same module
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/render.py"), "def draw(x):\n    return x\n")
    _w(str(tmp_path / "src/svc/main.py"),
       "def a():\n    # calls: code:svc.render.draw\n    pass\n\ndef b():\n    return 1\n")
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    assert "svc.render.draw" in by_id["svc.main.a"].calls        # marker's own function
    assert "svc.render.draw" not in by_id["svc.main.b"].calls    # sibling: unaffected
