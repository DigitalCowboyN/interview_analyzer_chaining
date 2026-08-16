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


def test_category_inherited_from_parent_primary(tmp_path):
    # parent primary carries category=operations with implemented_by=[]; the CHILD does the
    # implementing and leaves category unset — the unit must inherit the parent's category.
    _w(str(tmp_path / "docs/capabilities/maintain-graph.md"),
       "---\ntype: Capability\nkind: primary\ntier: core\ncategory: operations\n"
       "implemented_by: []\n---\nMaintain the graph.\n")
    _w(str(tmp_path / "docs/capabilities/catalog-api.md"),
       "---\ntype: Capability\nkind: child\nparent: maintain-graph\n"
       "implemented_by: [tools.api]\n---\nCatalog the API.\n")
    _w(str(tmp_path / "tools/api/__init__.py"))
    axes = derive_axes(str(tmp_path))
    assert axes["tools.api"][0] == "operations"   # inherited from the parent primary
