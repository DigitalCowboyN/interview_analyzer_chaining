from tools.corpus.check import check_misfiled
from tools.corpus.model import Record


def _rec(type_, path):
    return Record(type=type_, id="x", path=path, frontmatter={"type": type_}, body="")


def test_clean_when_in_home():
    recs = [_rec("Capability", "docs/capabilities/x.md"), _rec("ADR", "docs/adr/x.md")]
    assert check_misfiled(recs) == []


def test_flags_record_outside_its_home():
    findings = check_misfiled([_rec("Capability", "docs/adr/x.md")])
    assert len(findings) == 1
    assert "misfiled" in findings[0].message and "docs/capabilities" in findings[0].message
