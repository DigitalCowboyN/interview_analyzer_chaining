from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.testmap.reader import Test
from tools.testmap.verification import (
    UNVERIFIED,
    PARTIALLY_VERIFIED,
    VERIFIED,
    verified_units,
    verify_capabilities,
    verify_use_cases,
)


def _test(slug, target="", verifies=None, tt="unit"):
    return Test(
        slug=slug, path="p", test_type=tt, target=target, verifies=verifies or []
    )


def _cap(slug, impl):
    return Capability(
        slug=slug,
        kind="primary",
        tier="core",
        parent="",
        implemented_by=impl,
        statement="s",
        path="p",
        category="product",
    )


def _uc(slug, fulfilled_by):
    return UseCase(
        slug=slug,
        form="use-case",
        category="product",
        actor="a",
        statement="s",
        path="p",
        fulfilled_by=fulfilled_by,
    )


def test_verified_units_and_capability():
    tests = [_test("t1", target="api"), _test("t2", target="lens")]
    assert verified_units(tests) == {"api", "lens"}
    caps = [
        _cap("built", ["api"]),
        _cap("half", ["api", "unt"]),
        _cap("none", ["x"]),
        _cap("bare", []),
    ]
    cv = verify_capabilities(caps, tests)
    assert cv["built"] == VERIFIED
    assert cv["half"] == PARTIALLY_VERIFIED
    assert cv["none"] == UNVERIFIED
    assert cv["bare"] == UNVERIFIED


def test_direct_marker_on_capability_verifies_it():
    tests = [_test("t", verifies=["capabilities:c"])]
    cv = verify_capabilities([_cap("c", [])], tests)
    assert cv["c"] == VERIFIED  # direct marker beats empty implemented_by


def test_use_case_rollup_and_direct():
    tests = [
        _test("u", target="api"),
        _test("acc", verifies=["use-cases:direct"], tt="e2e"),
    ]
    caps = [_cap("built", ["api"]), _cap("unbuilt", ["x"])]
    ucs = [
        _uc("proven", ["built"]),
        _uc("partial", ["built", "unbuilt"]),
        _uc("none", ["unbuilt"]),
        _uc("bare", []),
        _uc("direct", []),
    ]
    uv = verify_use_cases(ucs, caps, tests)
    assert uv["proven"] == VERIFIED
    assert uv["partial"] == PARTIALLY_VERIFIED
    assert uv["none"] == UNVERIFIED
    assert uv["bare"] == UNVERIFIED
    assert (
        uv["direct"] == VERIFIED
    )  # direct acceptance marker, no fulfilling cap needed
