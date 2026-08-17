import tools.code.reader as code_reader
from tools.graph.traverse import walk


def test_module_level_surfaces_no_symbols():
    sg = walk("code:tools.graph.traverse", direction="out", depth=1, level="module")
    assert not any(a.startswith("code:tools.graph.traverse.") for a in sg.nodes)


def test_symbol_level_discloses_symbols_and_contains():
    sg = walk("code:tools.graph.traverse", direction="out", depth=1, level="symbol")
    assert "code:tools.graph.traverse.walk" in sg.nodes              # symbol node present
    assert sg.nodes["code:tools.graph.traverse.walk"].context        # signature/docstring context
    assert any(e.type == "contains" and e.dst == "code:tools.graph.traverse.walk"
               for e in sg.edges)                                    # module contains symbol


def test_symbol_calls_edge_present():
    sg = walk("code:tools.code.__main__.cmd_index", direction="out", depth=1, level="symbol")
    assert any(e.type == "calls" and e.src == "code:tools.code.__main__.cmd_index"
               and e.dst == "code:tools.code.render.render_index" for e in sg.edges)


def test_symbol_walk_up_reaches_module_then_intent():
    # from a symbol, walk 'in' reaches its module (contained_by), then its capability
    sg = walk("code:tools.graph.classify.derive_axes", direction="in", depth=3, level="symbol")
    assert "code:tools.graph.classify" in sg.nodes                   # its module (walk-up)
    assert any(a.startswith("capabilities:") for a in sg.nodes)      # inherited intent via walk-up


def test_frontier_laziness_parses_only_visited_modules(monkeypatch):
    calls = []
    real = code_reader.symbols_of
    monkeypatch.setattr(code_reader, "symbols_of",
                        lambda m, root=".": calls.append(m) or real(m, root))
    # a depth-1 out walk from ONE module at symbol level must not parse the whole repo
    walk("code:tools.graph.registry", direction="out", depth=1, level="symbol")
    assert "tools.graph.registry" in calls
    assert "tools.code.reader" not in calls          # an unvisited module is never symbol-parsed
