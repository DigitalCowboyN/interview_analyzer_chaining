from tools.code.reader import CodeUnit
from tools.code.check import check_coverage, check_classification, check_stale, Finding


def test_coverage_flags_undocumented_package():
    msgs = " ".join(f.message for f in check_coverage(["resolution", "events"], [CodeUnit("events", "infrastructure")]))
    assert "resolution" in msgs


def test_classification_flags_missing_role():
    msgs = " ".join(f.message for f in check_classification([CodeUnit("x", "")]))
    assert "x" in msgs


def test_stale_flags_unit_not_in_code():
    msgs = " ".join(f.message for f in check_stale([CodeUnit("gone", "surface")], ["events", "api"]))
    assert "gone" in msgs
