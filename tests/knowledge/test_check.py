import os
from tools.knowledge.check import (
    Finding, DOMAINS, ADDENDUM_HEADING,
    check_addendum_present, check_cascade_covers_domains, run_all,
)


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def test_addendum_missing_on_new_spec_is_flagged(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-08-05-new-thing-design.md"), "# New thing\nno addendum here\n")
    msgs = " ".join(f.message for f in check_addendum_present(str(specs), str(tmp_path / "plans")))
    assert "new-thing" in msgs


def test_addendum_present_on_new_spec_is_clean(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-08-05-new-thing-design.md"), f"# New thing\n{ADDENDUM_HEADING}\nreviewed\n")
    assert check_addendum_present(str(specs), str(tmp_path / "plans")) == []


def test_pre_adoption_spec_is_grandfathered(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-07-04-old-design.md"), "# Old\nno addendum, but predates the process\n")
    assert check_addendum_present(str(specs), str(tmp_path / "plans")) == []


def test_cascade_root_missing_domain_is_flagged(tmp_path):
    docs = tmp_path / "docs"
    _write(str(docs / "index.md"), "# Knowledge map\n[adr/](adr/index.md)\n")  # only adr
    msgs = " ".join(f.message for f in check_cascade_covers_domains(str(tmp_path)))
    assert "glossary" in msgs and "code" in msgs


def test_cascade_root_absent_is_one_finding(tmp_path):
    findings = check_cascade_covers_domains(str(tmp_path))  # no docs/index.md
    assert len(findings) == 1 and "cascade root" in findings[0].message


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)
