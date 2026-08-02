# tests/code/test_reader.py
from tools.code.reader import packages, dep_edges, io_of, load_units, CodeUnit

def test_packages_and_dep_edges(tmp_path):
    (tmp_path / "src" / "a").mkdir(parents=True)
    (tmp_path / "src" / "b").mkdir(parents=True)
    (tmp_path / "src" / "a" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "a" / "m.py").write_text("from src.b import x\n", encoding="utf-8")
    (tmp_path / "src" / "b" / "__init__.py").write_text("", encoding="utf-8")
    assert set(packages(str(tmp_path))) == {"a", "b"}
    edges = dep_edges(str(tmp_path))
    assert edges["a"] == ["b"] and edges.get("b", []) == []

def test_io_of_detects_signals(tmp_path):
    p = tmp_path / "src" / "x"; p.mkdir(parents=True)
    (p / "h.py").write_text("from src.events import E\nimport neo4j\n", encoding="utf-8")
    io = io_of("x", str(tmp_path))
    assert "ESDB" in io and "Neo4j" in io

def test_load_units_attaches_derived(tmp_path):
    (tmp_path / "src" / "ingestion").mkdir(parents=True)
    (tmp_path / "src" / "ingestion" / "m.py").write_text("from src.events import E\n", encoding="utf-8")
    (tmp_path / "src" / "events").mkdir(parents=True)
    (tmp_path / "src" / "events" / "__init__.py").write_text("", encoding="utf-8")
    cd = tmp_path / "docs" / "code"; cd.mkdir(parents=True)
    (cd / "ingestion.md").write_text(
        "---\ntype: CodeUnit\nunit: ingestion\nrole: pipeline-layer\nkey_modules: [m]\n---\nIngests.\n",
        encoding="utf-8")
    units = load_units(str(tmp_path))
    u = next(x for x in units if x.unit == "ingestion")
    assert u.role == "pipeline-layer" and "events" in u.depends_on and "ESDB" in u.io
    assert "Ingests" in u.description
