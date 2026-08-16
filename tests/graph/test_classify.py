import os

from tools.graph.classify import derive_axes


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _seed(tmp):
    # capability (category=product) implements the 'ask' package
    _w(str(tmp / "docs/capabilities/answer.md"),
       "---\ntype: Capability\nkind: primary\ntier: core\ncategory: product\n"
       "implemented_by: [ask]\n---\nAnswer questions.\n")
    _w(str(tmp / "src/ask/__init__.py"))
    _w(str(tmp / "src/ask/engine.py"), "from src.agents import factory\n")  # depends_on agents
    _w(str(tmp / "src/agents/__init__.py"))
    _w(str(tmp / "src/agents/factory.py"), "x = 1\n")
    _w(str(tmp / "src/events/__init__.py"))
    _w(str(tmp / "src/events/store.py"), "x = 1\n")


def test_category_from_implementing_capability(tmp_path):
    _seed(tmp_path)
    axes = derive_axes(str(tmp_path))
    assert axes["ask"][0] == "product"          # implemented by a product capability
    assert axes["events"][0] == ""              # implemented by nobody -> no category (the signal)


def test_determinism_from_agents_dependency(tmp_path):
    _seed(tmp_path)
    axes = derive_axes(str(tmp_path))
    assert axes["ask.engine"][1] == "probabilistic"   # depends_on agents
    assert axes["events.store"][1] == "deterministic"
