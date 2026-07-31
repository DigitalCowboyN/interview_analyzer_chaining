from tools.cli.reader import Command
from tools.cli.check import (
    check_docs_reference_real, check_catalog_in_sync, check_undocumented, Finding,
)

CMDS = [
    Command("test", "make", "Run tests", "everyday"),
    Command("mystery", "make", "", "undocumented"),
    Command("python -m src.lens", "module", "Lens engine.", "everyday"),
]


def test_docs_reference_real_flags_only_backticked_missing(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("Run `make test` then `make gone`. Also make sure to `python -m src.gone`.\n", encoding="utf-8")
    # root defaults to the repo root during pytest, so on-disk module resolution applies
    msgs = " ".join(f.message for f in check_docs_reference_real(CMDS, [str(doc)]))
    assert "make gone" in msgs                      # backticked, not real -> flagged
    assert "src.gone" in msgs                        # does not resolve on disk -> flagged
    assert "make sure" not in msgs and "make test" not in msgs   # prose + real command not flagged

def test_docs_reference_real_accepts_plain_module(tmp_path):
    # a `python -m X` that resolves to a real module file (not a catalogued package) is fine
    doc = tmp_path / "README.md"
    doc.write_text("Run `python -m src.projections.ensure_schema` to set up the schema.\n", encoding="utf-8")
    msgs = " ".join(f.message for f in check_docs_reference_real(CMDS, [str(doc)]))
    assert "ensure_schema" not in msgs


def test_catalog_in_sync(tmp_path):
    from tools.cli.render import render_catalog
    cat = tmp_path / "index.md"
    cat.write_text("stale\n", encoding="utf-8")
    assert check_catalog_in_sync(str(cat), CMDS)     # out of sync
    cat.write_text(render_catalog(CMDS), encoding="utf-8")
    assert check_catalog_in_sync(str(cat), CMDS) == []


def test_undocumented_informational():
    msgs = " ".join(f.message for f in check_undocumented(CMDS))
    assert "mystery" in msgs and "test" not in msgs
