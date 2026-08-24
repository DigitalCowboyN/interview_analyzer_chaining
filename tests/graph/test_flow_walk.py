from tools.graph.traverse import walk


def test_aggregate_emits_event_at_symbol_level():
    # a real aggregate METHOD constructs an event payload class; depth 3 = module->class->method->event
    # (the emitting method must be EXPANDED, so it sits one hop inside the depth boundary).
    sg = walk("code:events.aggregates", direction="out", depth=3, level="symbol")
    assert any(e.type == "emits" and e.dst.endswith("Data") for e in sg.edges)


def test_module_grain_has_no_symbol_flow_edges():
    # emits/handled_by are symbol-lazy — absent at module grain. (writes/reads are harvest-grain by
    # design, so they can appear; they just don't originate from the aggregates module here.)
    sg = walk("code:events.aggregates", direction="out", depth=1, level="module")
    assert not any(e.type in ("emits", "handled_by") for e in sg.edges)


def test_event_handled_by_handler():
    sg = walk("code:events.interview_events.InterviewCreatedData", direction="out",
              depth=1, level="symbol")
    assert any(e.type == "handled_by" and e.dst.endswith("InterviewCreatedHandler")
               for e in sg.edges)


def test_handler_module_writes_label():
    # writes is harvest-grain: the handler MODULE -> glossary:Fragment (Cypher MERGE (s:Fragment ...))
    sg = walk("code:projections.handlers.sentence_handlers", direction="out", depth=1, level="module")
    assert any(e.type == "writes" and e.dst == "glossary:Fragment" for e in sg.edges)


def test_schema_blast_radius_from_a_label():
    # SCHEMA-GAP #2 closed: from a label, reach BOTH who WRITES it (handler modules) and who READS it
    # (graph queries) — both harvest-grain, so discoverable inbound at module grain.
    sg = walk("glossary:Fragment", direction="in", depth=1, level="module")
    writers = [e.src for e in sg.edges if e.type == "writes"]
    readers = [e.src for e in sg.edges if e.type == "reads"]
    assert any(w.startswith("code:projections.handlers.") for w in writers)   # who populates it
    assert any(r.startswith("graph-queries:") for r in readers)               # who consumes it
