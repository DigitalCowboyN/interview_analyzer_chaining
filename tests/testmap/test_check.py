from tools.testmap.reader import Test
from tools.testmap.verification import UNVERIFIED, VERIFIED
from tools.testmap.check import (
    check_test_type,
    check_unmapped,
    check_unverified,
    run_all,
)


def _t(slug, tt="unit", target="", verifies=None):
    return Test(
        slug=slug, path="p", test_type=tt, target=target, verifies=verifies or []
    )


def test_check_test_type_flags_unknown():
    assert check_test_type([_t("a", tt="fuzz")])
    assert check_test_type([_t("a", tt="unit")]) == []


def test_check_unmapped_flags_targetless_markerless():
    assert check_unmapped([_t("orphan")])  # no target, no marker
    assert check_unmapped([_t("ok", target="api")]) == []
    assert check_unmapped([_t("ok2", verifies=["use-cases:x"])]) == []


def test_check_unverified_flags_unverified_use_cases():
    flagged = check_unverified({"a": UNVERIFIED, "b": VERIFIED})
    assert len(flagged) == 1 and "a" in flagged[0].message


def test_run_all_never_raises_on_real_repo():
    assert isinstance(run_all("."), list)
