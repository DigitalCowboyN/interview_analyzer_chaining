# tests/code/test_reader_tools.py
import os
from tools.code.reader import packages, dep_edges, _files_of


def _w(p, text=""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    _w(str(tmp / "src/a/__init__.py"))
    _w(str(tmp / "tools/y/__init__.py"))
    _w(str(tmp / "tools/x/reader.py"), "from tools.y import z\nfrom src.a import q\n")
    _w(str(tmp / "tools/x/__init__.py"))


def test_packages_includes_tools_prefixed(tmp_path):
    _fixture(tmp_path)
    pkgs = set(packages(str(tmp_path)))
    assert "a" in pkgs and "tools.x" in pkgs and "tools.y" in pkgs


def test_files_of_resolves_tools_package(tmp_path):
    _fixture(tmp_path)
    files = _files_of("tools.x", str(tmp_path))
    assert any(f.endswith(os.path.join("tools", "x", "reader.py")) for f in files)


def test_dep_edges_tool_to_tool_and_tool_to_src(tmp_path):
    _fixture(tmp_path)
    edges = dep_edges(str(tmp_path))
    assert edges["tools.x"] == ["a", "tools.y"]  # tool->src (a) + tool->tool (tools.y)


def test_bare_src_unit_edges_unchanged(tmp_path):
    # a src package importing another src package still yields a bare edge
    _w(str(tmp_path / "src/a/__init__.py"), "from src.b import q\n")
    _w(str(tmp_path / "src/b/__init__.py"))
    assert dep_edges(str(tmp_path))["a"] == ["b"]
