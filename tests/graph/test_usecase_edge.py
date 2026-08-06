from tools.graph.reader import harvest, nodes
from tools.graph.check import check_endpoints


def _seed(tmp_path):
    caps = tmp_path / "docs" / "capabilities"
    caps.mkdir(parents=True)
    (caps / "surface.md").write_text(
        "---\ntype: Capability\nkind: primary\ntier: core\ncategory: product\n"
        "implemented_by: []\n---\nSurface the signal.\n",
        encoding="utf-8",
    )
    ucs = tmp_path / "docs" / "use-cases"
    ucs.mkdir(parents=True)
    (ucs / "see-the-signal.md").write_text(
        "---\ntype: UseCase\nform: use-case\ncategory: product\nactor: analyst\n"
        "fulfilled_by: [surface]\n---\nAs an analyst, I want the signal.\n",
        encoding="utf-8",
    )


def test_fulfilled_by_harvested(tmp_path):
    _seed(tmp_path)
    edges = harvest(str(tmp_path))
    fb = [e for e in edges if e.type == "fulfilled_by"]
    assert any(
        e.src == "use-cases:see-the-signal" and e.dst == "capabilities:surface"
        for e in fb
    )
    assert "see-the-signal" in nodes(str(tmp_path))["UseCase"]


def test_dangling_fulfilled_by_flagged(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs" / "use-cases" / "ghost.md").write_text(
        "---\ntype: UseCase\nform: user-story\ncategory: product\nactor: a\n"
        "fulfilled_by: [no-such-cap]\n---\nGhost intent.\n",
        encoding="utf-8",
    )
    edges = harvest(str(tmp_path))
    findings = check_endpoints(edges, nodes(str(tmp_path)))
    assert any("no-such-cap" in f.message for f in findings)
