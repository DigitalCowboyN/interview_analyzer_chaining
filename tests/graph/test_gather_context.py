from tools.graph.traverse import gather_context, walk


def test_gather_context_is_small_and_reaches_intent():
    # entry: the traversal module. Full closure is ~644 nodes; gather_context must be far smaller
    # and must include the governing ADR (post-govern-edges) plus the entry's own out-neighbors.
    sg = gather_context("code:tools.graph.traverse", level="module")
    full = walk("code:tools.graph.traverse", direction="both", depth=None, level="module")
    assert len(sg.nodes) < len(full.nodes)                     # minimal, not the whole closure
    assert "code:tools.graph.traverse" in sg.nodes             # the entry
    # reached the nearest governing intent by walking up (adr:25/27 govern it directly)
    assert any(a.partition(":")[0] in ("adr", "capabilities", "use-cases") for a in sg.nodes)


def test_gather_context_stops_at_first_intent_layer(monkeypatch):
    import tools.graph.traverse as tr
    calls = []
    real = tr.walk

    def spy(entry, direction="both", depth=None, root=".", level="module"):
        calls.append((direction, depth))
        return real(entry, direction=direction, depth=depth, root=root, level=level)

    monkeypatch.setattr(tr, "walk", spy)
    gather_context("code:tools.graph.traverse", level="module")
    in_depths = [d for (dr, d) in calls if dr == "in"]
    # climbed progressively from depth 1 and stopped once intent was found (did not run to max_up=6)
    assert in_depths == sorted(in_depths) and max(in_depths) < 6
