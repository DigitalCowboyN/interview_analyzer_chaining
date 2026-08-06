# tests/graph/test_verifies_edge.py
from tools.graph.reader import harvest, nodes
from tools.graph.check import check_endpoints


def _seed(tmp_path):
    caps = tmp_path / "docs" / "capabilities"
    caps.mkdir(parents=True)
    (caps / "map-the-tests.md").write_text(
        "---\ntype: Capability\nkind: primary\ntier: core\ncategory: operations\n"
        "implemented_by: [tools.capability]\n---\nMap tests.\n",
        encoding="utf-8",
    )
    code = tmp_path / "docs" / "code"
    code.mkdir(parents=True)
    (code / "tools.capability.md").write_text(
        "---\ntype: CodeUnit\nunit: tools.capability\nrole: tooling\n---\nx\n",
        encoding="utf-8",
    )
    # real tools dir so target resolution (packages()) sees tools.capability, in addition
    # to the docs/code node above (which drives the CodeUnit node inventory):
    (tmp_path / "tools" / "capability").mkdir(parents=True)
    t = tmp_path / "tests" / "capability"
    t.mkdir(parents=True)
    (t / "test_check.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    ti = tmp_path / "tests" / "integration"
    ti.mkdir(parents=True)
    (ti / "test_e2e_x.py").write_text(
        "# verifies: capabilities:map-the-tests\ndef test_flow():\n    pass\n",
        encoding="utf-8",
    )


def test_verifies_harvested_with_test_type(tmp_path):
    _seed(tmp_path)
    edges = harvest(str(tmp_path))
    ve = [e for e in edges if e.type == "verifies"]
    assert any(
        e.src == "tests:capability.test_check"
        and e.dst == "code:tools.capability"
        and e.props.get("test_type") == "unit"
        for e in ve
    )
    assert any(
        e.src == "tests:integration.test_e2e_x"
        and e.dst == "capabilities:map-the-tests"
        and e.props.get("test_type") == "e2e"
        for e in ve
    )
    assert "capability.test_check" in nodes(str(tmp_path))["Test"]


def test_dangling_marker_flagged(tmp_path):
    _seed(tmp_path)
    ti = tmp_path / "tests" / "integration"
    (ti / "test_ghost.py").write_text(
        "# verifies: use-cases:no-such-uc\ndef test_g():\n    pass\n", encoding="utf-8"
    )
    edges = harvest(str(tmp_path))
    findings = check_endpoints(edges, nodes(str(tmp_path)))
    assert any("no-such-uc" in f.message for f in findings)
