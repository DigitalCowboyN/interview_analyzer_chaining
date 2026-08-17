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
