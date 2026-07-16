"""RRF fusion: 1-based ranks, deterministic ties, count ranking."""

from src.ask.fusion import rank_by_count, rrf_merge, top_k


def test_rank_by_count_orders_by_hits_then_id():
    rows = [{"fragment_id": "b"}, {"fragment_id": "a"},
            {"fragment_id": "b"}, {"fragment_id": "c"}]
    assert rank_by_count(rows) == ["b", "a", "c"]


def test_rrf_merge_scores_are_reciprocal_rank_sums():
    rankings = {"vector": ["f1", "f2"], "fulltext": ["f2", "f1"], "graph": ["f2"]}
    scored = dict(rrf_merge(rankings))
    assert scored["f2"] == 1 / 62 + 1 / 61 + 1 / 61
    assert scored["f1"] == 1 / 61 + 1 / 62


def test_rrf_merge_orders_ties_by_fragment_id():
    scored = rrf_merge({"a": ["f2"], "b": ["f1"]})
    assert [f for f, _ in scored] == ["f1", "f2"]  # equal scores → id ascending


def test_rrf_empty_channels_are_skipped():
    assert rrf_merge({"vector": [], "graph": ["f1"]})[0][0] == "f1"


def test_top_k_truncates():
    assert top_k([("a", 0.9), ("b", 0.5)], 1) == ["a"]
