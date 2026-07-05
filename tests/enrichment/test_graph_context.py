from src.enrichment.graph_context import FragmentView, GraphContextBuilder

WINDOWS = {
    "immediate_context": 1,
    "observer_context": 2,
    "broader_context": 3,
    "overall_context": 5,
}


def make_fragments():
    rows = [
        ("S1", "Hi, thanks for joining."),
        ("S2", "Happy to be here."),
        ("S1", "First question:"),
        ("S2", "Sure."),
        ("S1", "how do you build flashable files?"),
    ]
    return [
        FragmentView(
            index=i, text=t, speaker_handle=h, utterance_id=("u-1" if i in (2, 4) else None)
        )
        for i, (h, t) in enumerate(rows)
    ]


def test_contexts_render_speaker_labels_and_mark_target():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(
        make_fragments(), {"u-1": "First question: how do you build flashable files?"}
    )
    immediate = contexts[2]["immediate_context"]
    assert "[S2]: Happy to be here." in immediate
    assert ">>> [S1]: First question: <<<" in immediate
    assert "[S2]: Sure." in immediate


def test_window_sizes_respected():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {})
    immediate = contexts[0]["immediate_context"]
    assert "First question" not in immediate  # window 1: only fragments 0-1


def test_utterance_context_supplies_stitched_thought():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(
        make_fragments(), {"u-1": "First question: how do you build flashable files?"}
    )
    assert contexts[2]["utterance_context"] == "First question: how do you build flashable files?"
    assert contexts[1]["utterance_context"] == "Happy to be here."  # no utterance -> own text


def test_all_configured_windows_present():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {})
    assert set(contexts[0].keys()) == set(WINDOWS.keys()) | {"utterance_context"}


def test_empty_fragments_returns_empty():
    builder = GraphContextBuilder(WINDOWS)
    assert builder.build_all([], {}) == []
