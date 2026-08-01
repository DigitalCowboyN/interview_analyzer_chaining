# tests/graphq/test_check.py
from tools.graphq.reader import QueryEntry
from tools.graphq.check import check_schema_drift, check_missing_marker, Finding

def test_schema_drift_flags_unknown_label():
    vocab = {"Interview": None, "HAS_SENTENCE": None}   # keys = known names
    e = QueryEntry("b.py", "q", "export", "task", ["export"], ["Interview", "Ghost"], ["HAS_SENTENCE"], [], [], [])
    msgs = " ".join(f.message for f in check_schema_drift([e], vocab))
    assert "Ghost" in msgs and "Interview" not in msgs.replace("Ghost", "")

def test_missing_marker_flags_unannotated():
    e = QueryEntry("b.py", "q", "", "", [], ["Interview"], [], [], [], [])
    msgs = " ".join(f.message for f in check_missing_marker([e]))
    assert "q" in msgs
