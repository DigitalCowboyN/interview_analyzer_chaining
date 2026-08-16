import os
from tools.capability.reader import (
    Capability,
    load_capabilities,
    real_code_units,
    CATEGORIES,
    category_defined,
)


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def test_load_parses_node_and_links(tmp_path):
    cap = tmp_path / "docs/capabilities/enrich-fragments.md"
    _write(str(cap), "---\ntype: Capability\nkind: primary\ntier: core\n"
                     "implemented_by: [enrichment, agents]\n---\nEnrich each fragment.\n")
    _write(str(tmp_path / "docs/capabilities/index.md"), "# Capabilities\n")  # skipped
    caps = load_capabilities(str(tmp_path))
    assert len(caps) == 1
    c = caps[0]
    assert c.slug == "enrich-fragments" and c.kind == "primary" and c.tier == "core"
    assert c.implemented_by == ["enrichment", "agents"]
    assert c.statement == "Enrich each fragment."


def test_load_skips_non_capability_files(tmp_path):
    _write(str(tmp_path / "docs/capabilities/notes.md"), "# just notes, no frontmatter\n")
    assert load_capabilities(str(tmp_path)) == []


def test_real_code_units_covers_packages_and_modules():
    units = real_code_units(".")
    assert "enrichment" in units          # a package
    assert "lens.engine" in units         # a module (src/lens/engine.py)
    assert "ask.reader" in units          # a module


def test_load_parses_category(tmp_path):
    cap = tmp_path / "docs/capabilities/x.md"
    os.makedirs(os.path.dirname(cap), exist_ok=True)
    open(cap, "w").write("---\ntype: Capability\nkind: primary\ntier: core\n"
                         "category: operations\nimplemented_by: [tools.code]\n---\nDoes a thing.\n")
    c = load_capabilities(str(tmp_path))[0]
    assert c.category == "operations"
    assert list(CATEGORIES)[:2] == ["product", "operations"]  # product/operations populated; then reserved


def test_categories_is_defined_axis():
    # membership + iteration still behave like the old list
    assert "product" in CATEGORIES and "nonsense" not in CATEGORIES
    assert list(CATEGORIES)[:2] == ["product", "operations"]  # order preserved for render
    # product/operations/supporting are defined; strategic is reserved ("")
    assert category_defined("product") and category_defined("operations")
    assert category_defined("supporting")
    assert not category_defined("strategic")   # reserved
    assert not category_defined("unknown")     # not in the axis
