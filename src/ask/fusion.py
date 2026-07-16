"""Reciprocal-rank fusion across retrieval channels (pure functions)."""

from collections import Counter
from typing import Dict, List, Tuple

RRF_K = 60


def rank_by_count(rows: List[Dict]) -> List[str]:
    """Graph-channel ranking: more anchor hits rank higher; ties by id."""
    counts = Counter(r["fragment_id"] for r in rows)
    return [fid for fid, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]


def rrf_merge(rankings: Dict[str, List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """score(f) = sum over channels of 1/(k + rank), rank 1-based."""
    scores: Dict[str, float] = {}
    for ranked in rankings.values():
        for rank, fid in enumerate(ranked, start=1):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def top_k(scored: List[Tuple[str, float]], k: int) -> List[str]:
    return [fid for fid, _ in scored[:k]]
