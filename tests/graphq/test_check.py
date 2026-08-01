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

def test_output_contract_is_scoped_to_bound_var(tmp_path):
    from tools.graphq.reader import QueryEntry
    from tools.graphq.check import check_output_contract
    d = tmp_path / "src" / "api"; d.mkdir(parents=True)
    (d / "c.py").write_text(
        "def handler(cfg):\n"
        "    x = cfg['unrelated']\n"                      # not from a query -> never flagged
        "    rows = worklist_rows(session)\n"
        "    for r in rows:\n"
        "        a = r['interview_id']\n"                 # returned -> ok
        "        b = r['ghost_field']\n",                 # NOT returned -> flag
        encoding="utf-8")
    entries = [QueryEntry("src/export/reader.py", "worklist_rows", "export", "task",
                          ["export"], ["Interview"], [], [], ["interview_id"], ["api"])]
    msgs = " ".join(f.message for f in check_output_contract(entries, root=str(tmp_path)))
    assert "ghost_field" in msgs
    assert "unrelated" not in msgs and "interview_id" not in msgs
