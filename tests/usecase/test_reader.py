from tools.usecase.reader import FORMS, UseCase, load_use_cases

CORE = """---
type: UseCase
form: use-case
category: product
actor: analyst
acceptance_criteria:
  - "Given a transcript, when analyzed, then insights are surfaced"
fulfilled_by: [extract-insights-via-lenses]
level: user-goal
---
As an analyst drowning in transcripts, I want the signal surfaced so I stop missing what matters.
"""


def _write(tmp_path, name, text):
    d = tmp_path / "docs" / "use-cases"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")


def test_loads_core_and_optional_fields(tmp_path):
    _write(tmp_path, "surface-the-signal.md", CORE)
    ucs = load_use_cases(str(tmp_path))
    assert len(ucs) == 1
    u = ucs[0]
    assert u.slug == "surface-the-signal"
    assert u.form == "use-case" and u.category == "product" and u.actor == "analyst"
    assert u.fulfilled_by == ["extract-insights-via-lenses"]
    assert u.acceptance_criteria == [
        "Given a transcript, when analyzed, then insights are surfaced"
    ]
    assert u.level == "user-goal"
    assert u.statement.startswith("As an analyst")


def test_skips_index_readme_and_non_usecase(tmp_path):
    _write(tmp_path, "index.md", "# Use-Cases\n")
    _write(tmp_path, "README.md", "# concept\n")
    _write(tmp_path, "other.md", "---\ntype: Capability\n---\nnope\n")
    assert load_use_cases(str(tmp_path)) == []


def test_missing_optional_fields_default_empty(tmp_path):
    _write(
        tmp_path,
        "bare.md",
        "---\ntype: UseCase\nform: user-story\ncategory: operations\n---\nA bare intent.\n",
    )
    u = load_use_cases(str(tmp_path))[0]
    assert u.acceptance_criteria == [] and u.fulfilled_by == [] and u.level == ""
    assert "user-story" in FORMS
