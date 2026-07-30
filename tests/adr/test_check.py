from tools.adr.model import Adr
from tools.adr.check import (
    check_structural, check_specs_reference_adr, check_staleness, Finding,
)

def _adr(**kw):
    base = dict(id=1, title="A", status="accepted", date="2026-07-04",
                supersedes=[], superseded_by=[], tags=[], source="docs/x.md",
                path=f"docs/adr/{kw.get('id',1):04d}.md", body="")
    base.update(kw); return Adr(**base)

def test_structural_flags_duplicate_id_and_one_directional_supersede():
    a = _adr(id=1, supersedes=[2])
    b = _adr(id=2, superseded_by=[])          # missing back-edge
    c = _adr(id=2)                             # duplicate id 2
    msgs = " ".join(f.message for f in check_structural([a, b, c]))
    assert "duplicate id 0002" in msgs
    assert "0001 supersedes 0002" in msgs      # one-directional edge flagged

def test_structural_flags_bad_status():
    msgs = " ".join(f.message for f in check_structural([_adr(id=1, status="bogus")]))
    assert "invalid status" in msgs

def test_specs_reference_adr_warns_when_decisions_locked_no_adr(tmp_path):
    (tmp_path / "s1.md").write_text("## Decisions locked\nwe chose X\n", encoding="utf-8")
    (tmp_path / "s2.md").write_text("## Decisions locked\nsee ADR-0003\n", encoding="utf-8")
    msgs = [f.message for f in check_specs_reference_adr(str(tmp_path))]
    assert any("s1.md" in m for m in msgs)
    assert not any("s2.md" in m for m in msgs)   # references an ADR → no warning

def test_staleness_warns_when_source_newer_than_adr():
    a = _adr(id=1, source="docs/x.md", path="docs/adr/0001.md")
    def fake_ts(path):
        return 200 if path == "docs/x.md" else 100   # source newer than adr
    msgs = [f.message for f in check_staleness([a], ts_fn=fake_ts)]
    assert any("0001" in m and "docs/x.md" in m for m in msgs)

def test_generated_in_sync_flags_out_of_sync_then_clean(tmp_path):
    from tools.adr.index import load_bundle, write_generated
    from tools.adr.check import check_generated_in_sync
    (tmp_path / "0001-a.md").write_text(
        "---\ntype: ADR\nid: 1\ntitle: A\nstatus: accepted\ndate: 2026-07-04\n"
        "supersedes: []\nsuperseded_by: []\ntags: []\nsource: docs/x.md\n---\nbody\n",
        encoding="utf-8")
    adrs = load_bundle(str(tmp_path))
    assert check_generated_in_sync(str(tmp_path), adrs)          # no index/log yet -> flagged
    write_generated(str(tmp_path))
    adrs = load_bundle(str(tmp_path))
    assert check_generated_in_sync(str(tmp_path), adrs) == []    # now in sync

def test_run_all_aggregates_multiple_check_families(tmp_path):
    from tools.adr.check import run_all
    (tmp_path / "0001-a.md").write_text(
        "---\ntype: ADR\nid: 1\ntitle: A\nstatus: accepted\ndate: 2026-07-04\n"
        "supersedes: [2]\nsuperseded_by: []\ntags: []\nsource: docs/x.md\n---\nbody\n",
        encoding="utf-8")
    specs = tmp_path / "specs"; specs.mkdir()
    (specs / "s.md").write_text("## Decisions locked\nchose X\n", encoding="utf-8")
    msgs = " ".join(f.message for f in run_all(str(tmp_path), str(specs)))
    assert "unknown 0002" in msgs        # structural family
    assert "out of sync" in msgs          # generated-in-sync family
    assert "s.md" in msgs                 # spec-references-adr family

def test_structural_flags_dangling_superseded_by():
    a = _adr(id=1, superseded_by=[9])     # ADR 9 does not exist
    msgs = " ".join(f.message for f in check_structural([a]))
    assert "superseded_by unknown 0009" in msgs
