from tools.adr.index import load_bundle, render_index, render_log, write_generated

def _write(dir_, name, body):
    (dir_ / name).write_text(body, encoding="utf-8")

ADR_TMPL = """---
type: ADR
id: {id}
title: {title}
status: {status}
date: {date}
supersedes: {supersedes}
superseded_by: []
tags: []
source: docs/x.md
---
body
"""

def test_load_bundle_skips_reserved_and_sorts(tmp_path):
    _write(tmp_path, "index.md", "# generated\n")
    _write(tmp_path, "0002-b.md", ADR_TMPL.format(id=2, title="B", status="accepted", date="2026-07-05", supersedes="[]"))
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    adrs = load_bundle(str(tmp_path))
    assert [a.id for a in adrs] == [1, 2]        # sorted, index.md skipped

def test_render_index_and_log(tmp_path):
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    adrs = load_bundle(str(tmp_path))
    idx = render_index(adrs)
    assert "| 0001 | A | accepted |" in idx
    log = render_log(adrs)
    assert "0001" in log and "2026-07-04" in log

def test_write_generated_is_idempotent(tmp_path):
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    write_generated(str(tmp_path))
    first = (tmp_path / "index.md").read_text()
    write_generated(str(tmp_path))
    assert (tmp_path / "index.md").read_text() == first
