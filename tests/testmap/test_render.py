from tools.testmap.reader import Test
from tools.testmap.render import render_index
from tools.testmap.verification import VERIFIED, UNVERIFIED


def _t(slug, tt, target="", verifies=None, n=1):
    return Test(
        slug=slug,
        path="tests/" + slug.replace(".", "/") + ".py",
        test_type=tt,
        target=target,
        verifies=verifies or [],
        n_tests=n,
    )


def test_groups_by_type_and_shows_rollup():
    tests = [
        _t("cap.test_x", "unit", target="tools.capability"),
        _t("integration.test_e2e", "e2e", verifies=["use-cases:uc1"]),
    ]
    out = render_index(tests, {"capA": VERIFIED}, {"uc1": VERIFIED, "uc2": UNVERIFIED})
    assert out.startswith("# Tests")
    assert "## unit" in out and "## e2e" in out
    assert "cap.test_x" in out and "tools.capability" in out
    assert "use-cases:uc1" in out  # authored marker shown
    assert "## Verification rollup" in out
    assert "uc1" in out and "VERIFIED" in out and "uc2" in out and "UNVERIFIED" in out
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_empty_type_omitted():
    out = render_index([_t("a.test_a", "unit")], {}, {})
    assert "## integration" not in out and "## e2e" not in out
