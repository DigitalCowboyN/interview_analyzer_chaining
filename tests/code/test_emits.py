import os
from tools.code.reader import symbols_of


def _w(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(t)


def test_emits_from_event_constructor_call(tmp_path):
    _w(str(tmp_path / "src/events/__init__.py"), "")
    _w(str(tmp_path / "src/events/foo_events.py"), "class FooHappenedData:\n    pass\n")
    _w(str(tmp_path / "src/events/aggregates.py"),
       "from src.events.foo_events import FooHappenedData\n\n"
       "def do_it():\n    return FooHappenedData()\n")
    by = {s.id: s for s in symbols_of("events.aggregates", str(tmp_path))}
    assert "events.foo_events.FooHappenedData" in by["events.aggregates.do_it"].emits


def test_emits_marker_for_dynamic_emission(tmp_path):
    _w(str(tmp_path / "src/events/__init__.py"), "")
    _w(str(tmp_path / "src/events/foo_events.py"), "class FooHappenedData:\n    pass\n")
    _w(str(tmp_path / "src/events/aggregates.py"),
       "def do_it(cls):\n    # emits: events.foo_events.FooHappenedData\n    return cls()\n")
    by = {s.id: s for s in symbols_of("events.aggregates", str(tmp_path))}
    assert "events.foo_events.FooHappenedData" in by["events.aggregates.do_it"].emits


def test_non_event_call_is_not_emit(tmp_path):
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/m.py"), "class Thing:\n    pass\n\ndef f():\n    return Thing()\n")
    by = {s.id: s for s in symbols_of("svc.m", str(tmp_path))}
    assert by["svc.m.f"].emits == []
