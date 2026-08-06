from tools.usecase.reader import UseCase
from tools.usecase.coverage import NOT_COVERED, PARTIALLY_COVERED
from tools.usecase.check import (
    check_forms,
    check_categories,
    check_acceptance_criteria,
    check_uncovered,
    run_all,
)


def _uc(slug, form="use-case", category="product", ac=None, fulfilled_by=None):
    return UseCase(
        slug=slug,
        form=form,
        category=category,
        actor="a",
        statement="s",
        path="p",
        acceptance_criteria=ac or [],
        fulfilled_by=fulfilled_by or [],
    )


def test_check_forms_flags_unknown():
    assert check_forms([_uc("x", form="job-story")])  # not in FORMS -> flagged
    assert check_forms([_uc("x", form="use-case")]) == []


def test_check_categories_flags_unknown():
    assert check_categories([_uc("x", category="marketing")])
    assert check_categories([_uc("x", category="operations")]) == []


def test_empty_acceptance_criteria_is_advisory():
    assert check_acceptance_criteria([_uc("x", ac=[])])
    assert check_acceptance_criteria([_uc("x", ac=["c"])]) == []


def test_uncovered_flagged():
    cov = {"bare": NOT_COVERED, "part": PARTIALLY_COVERED}
    flagged = check_uncovered([_uc("bare"), _uc("part")], cov)
    assert len(flagged) == 1 and "bare" in flagged[0].message


def test_run_all_never_raises_on_real_repo():
    findings = run_all(".")  # advisory findings allowed; must not raise
    assert isinstance(findings, list)
