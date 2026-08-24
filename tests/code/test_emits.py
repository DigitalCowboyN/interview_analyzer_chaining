import os
from tools.code.reader import _is_event_class, symbols_of


def test_is_event_class_requires_events_module():
    # real payload classes live in an `*_events` module
    assert _is_event_class("events.sentence_events.SentenceEditedData")
    assert _is_event_class("events.interview_events.InterviewCreatedData")
    # a type under events/ whose name ends in "Data" but is NOT in an `*_events` module
    # must NOT count as an emitted event (the `_events` guard, not just endswith("Data"))
    assert not _is_event_class("events.envelope.EnvelopeData")
    assert not _is_event_class("events.store.CheckpointData")
    # ends in "Data" and under a `*_events` module but not the `events` package
    assert not _is_event_class("commands.sentence_events.CommandData")


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
