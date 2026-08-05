import os
from tools.capability.reader import Capability, load_capabilities, real_code_units, code_nodes


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


def test_real_code_units_includes_packages_and_key_modules():
    units = real_code_units(".")
    assert "enrichment" in units and "lens.engine" in units and "ask.reader" in units


def test_code_nodes_carry_roles():
    nodes = code_nodes(".")
    roles = {n.unit: n.role for n in nodes}
    assert roles.get("api") == "surface" and roles.get("lens") == "pipeline-layer"
