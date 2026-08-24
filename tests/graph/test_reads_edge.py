from tools.graph.reader import harvest


def test_reads_edges_link_queries_to_label_terms():
    edges = harvest(".")
    reads = {(e.src, e.dst) for e in edges if e.type == "reads"}
    # a real query declares labels=['Project'] -> reads glossary:Project (Project IS a glossary term)
    assert ("graph-queries:reader.project_exists", "glossary:Project") in reads
    # every reads edge targets a real glossary term (no dangling)
    from tools.graph.reader import nodes
    terms = nodes(".").get("GlossaryTerm", set())
    assert all(dst.split(":", 1)[1] in terms for _, dst in reads)
