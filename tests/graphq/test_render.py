from tools.graphq.reader import QueryEntry
from tools.graphq.render import render_catalog


def test_render_catalog():
    e = QueryEntry("src/export/reader.py", "worklist_rows", "export", "domain-broad",
                   ["export"], ["Interview"], [], [], ["interview_id"], ["api"])
    out = render_catalog([e])
    assert "## src/export/reader.py" in out
    assert "worklist_rows" in out and "domain-broad" in out and "Interview" in out and "interview_id" in out
