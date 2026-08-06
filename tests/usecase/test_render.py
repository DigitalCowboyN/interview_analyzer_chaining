from tools.usecase.reader import UseCase
from tools.usecase.render import render_index
from tools.usecase.coverage import FULLY_COVERED, NOT_COVERED


def _uc(
    slug,
    form="use-case",
    category="product",
    actor="analyst",
    fulfilled_by=None,
    ac=None,
):
    return UseCase(
        slug=slug,
        form=form,
        category=category,
        actor=actor,
        statement=f"statement for {slug}",
        path="p",
        acceptance_criteria=ac or [],
        fulfilled_by=fulfilled_by or [],
    )


def test_groups_by_category_then_form_with_coverage():
    ucs = [
        _uc("z-signal", fulfilled_by=["c"], ac=["x"]),
        _uc("a-import", form="requirement", fulfilled_by=[]),
    ]
    cov = {"z-signal": FULLY_COVERED, "a-import": NOT_COVERED}
    out = render_index(ucs, cov)
    assert out.startswith("# Use-Cases")
    assert "## product" in out
    assert "### use-case" in out and "### requirement" in out
    assert "#### z-signal — FULLY_COVERED" in out
    assert "#### a-import — NOT_COVERED" in out
    assert "- **fulfilled_by:** c" in out
    assert "- **acceptance_criteria:** 1" in out  # z-signal has 1
    assert "- **acceptance_criteria:** — none yet" in out  # a-import has none
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_empty_category_and_form_omitted():
    out = render_index([_uc("only", category="product")], {"only": NOT_COVERED})
    assert "## operations" not in out and "## support" not in out
