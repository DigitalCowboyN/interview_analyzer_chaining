from tools.graph.traverse import walk


def test_aggregate_emits_event_at_symbol_level():
    # a real aggregate METHOD constructs an event payload class; depth 3 = module->class->method->event
    # (the emitting method must be EXPANDED, so it sits one hop inside the depth boundary).
    sg = walk("code:events.aggregates", direction="out", depth=3, level="symbol")
    assert any(e.type == "emits" and e.dst.endswith("Data") for e in sg.edges)


def test_module_grain_has_no_flow_edges():
    sg = walk("code:events.aggregates", direction="out", depth=1, level="module")
    assert not any(e.type in ("emits", "handled_by", "writes") for e in sg.edges)
