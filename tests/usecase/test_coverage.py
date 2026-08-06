from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.usecase.coverage import (
    NOT_COVERED,
    PARTIALLY_COVERED,
    FULLY_COVERED,
    coverage,
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


def _cap(slug, implemented_by):
    return Capability(
        slug=slug,
        kind="primary",
        tier="core",
        parent="",
        implemented_by=implemented_by,
        statement="s",
        path="p",
        category="product",
    )


def test_three_states():
    caps = [_cap("built", ["api"]), _cap("unbuilt", [])]
    valid = {"api"}
    ucs = [
        _uc("bare", []),  # nothing fulfills -> NOT
        _uc("aspirational", ["unbuilt"]),  # fulfilled by unimplemented -> PARTIAL
        _uc("done", ["built"]),  # fulfilled by implemented -> FULL
        _uc("mixed", ["built", "unbuilt"]),  # one gap -> PARTIAL
    ]
    cov = coverage(ucs, caps, valid)
    assert cov["bare"] == NOT_COVERED
    assert cov["aspirational"] == PARTIALLY_COVERED
    assert cov["done"] == FULLY_COVERED
    assert cov["mixed"] == PARTIALLY_COVERED


def test_unresolvable_capability_slug_ignored():
    # a fulfilled_by pointing at a nonexistent capability contributes nothing
    cov = coverage([_uc("ghost", ["no-such-cap"])], [], set())
    assert cov["ghost"] == NOT_COVERED
