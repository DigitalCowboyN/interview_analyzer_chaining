from tools.cli.reader import Command
from tools.cli.render import render_help, render_catalog

CMDS = [
    Command("test", "make", "Run the tests", "everyday"),
    Command("wait-db", "make", "Wait for the test DB", "internal"),
    Command("mystery", "make", "", "undocumented"),
    Command("python -m src.lens", "module", "Lens engine.", "everyday"),
]


def test_render_help_shows_only_everyday_make():
    out = render_help(CMDS)
    assert "test" in out and "Run the tests" in out
    assert "wait-db" not in out            # internal hidden
    assert "python -m src.lens" not in out  # modules not in make help


def test_render_catalog_shows_all_labeled():
    out = render_catalog(CMDS)
    assert "test" in out and "wait-db" in out and "internal" in out
    assert "python -m src.lens" in out and "Lens engine." in out
