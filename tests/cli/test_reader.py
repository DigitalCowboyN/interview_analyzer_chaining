from tools.cli.reader import parse_makefile, module_entrypoints, Command

MAKEFILE = """\
PYTHON := python
.PHONY: test lint

test: ## Run the tests
\t$(PYTHON) -m pytest

lint: ## Lint the code
\tflake8 src

wait-db: ##@ Wait for the test DB
\tsleep 1

mystery:
\techo hi
"""


def test_parse_makefile_classifies(tmp_path):
    mk = tmp_path / "Makefile"; mk.write_text(MAKEFILE, encoding="utf-8")
    cmds = {c.name: c for c in parse_makefile(str(mk))}
    assert cmds["test"].visibility == "everyday" and cmds["test"].description == "Run the tests"
    assert cmds["wait-db"].visibility == "internal" and cmds["wait-db"].description == "Wait for the test DB"
    assert cmds["mystery"].visibility == "undocumented" and cmds["mystery"].description == ""
    assert "PYTHON" not in cmds        # := assignment is not a target
    assert all(c.kind == "make" for c in cmds.values())


def test_module_entrypoints(tmp_path):
    pkg = tmp_path / "src" / "thing"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Thing tool.\n\nmore\n"""\n', encoding="utf-8")
    (pkg / "__main__.py").write_text("print('hi')\n", encoding="utf-8")
    cmds = module_entrypoints(str(tmp_path))
    assert any(c.name == "python -m src.thing" and c.description == "Thing tool." for c in cmds)
    assert all(c.kind == "module" and c.visibility == "everyday" for c in cmds)
