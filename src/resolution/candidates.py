"""Pure candidate logic for entity and person resolution (M4.5b).

Deterministic + review (spec): normalization and exact grouping auto-merge;
embedding cosine promotes near-duplicates at auto_merge_threshold and
surfaces a review band above suggest_threshold. No IO here — the engine and
the on-demand worklist suggestions both call these functions.
"""

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

Key = Tuple[str, str]  # (normalized_surface, entity_type)

_ARTICLES = ("the ", "a ", "an ")


def normalize_surface(surface: str) -> str:
    """casefold, collapse whitespace, strip leading article/possessive, naive plural-fold."""
    s = " ".join(surface.casefold().split())
    for article in _ARTICLES:
        if s.startswith(article):
            s = s[len(article):]
            break
    if s.endswith("'s"):
        s = s[:-2]
    elif s.endswith("s'"):
        s = s[:-1]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s.strip()


def normalize_name(name: str) -> str:
    """casefold + whitespace collapse (person display names)."""
    return " ".join(name.casefold().split())


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def exact_groups(rows: Iterable[Dict[str, Any]]) -> Dict[Key, List[Dict[str, Any]]]:
    """Group surface rows by (normalized surface, entity_type)."""
    groups: Dict[Key, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (normalize_surface(row["surface"]), row["entity_type"])
        groups.setdefault(key, []).append(row)
    return groups


def representative(rows: List[Dict[str, Any]]) -> str:
    """The surface with the most mentions; ties break lexicographically."""
    return sorted(rows, key=lambda r: (-r.get("mentions", 0), r["surface"]))[0]["surface"]


class UnionFind:
    """Minimal union-find over hashable keys."""

    def __init__(self, keys: Iterable[Key]):
        self._parent = {k: k for k in keys}

    def find(self, key: Key) -> Key:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: Key, b: Key) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def clusters(self) -> List[List[Key]]:
        by_root: Dict[Key, List[Key]] = {}
        for key in self._parent:
            by_root.setdefault(self.find(key), []).append(key)
        return [sorted(members) for _, members in sorted(by_root.items())]


def embedding_pairs(
    keys: List[Key],
    vectors: Dict[Key, List[float]],
    auto_thr: float,
    suggest_thr: float,
) -> Tuple[List[Tuple[Key, Key]], List[Dict[str, Any]]]:
    """Pairwise cosine within an entity_type: >= auto_thr unions, [suggest, auto) suggests."""
    unions: List[Tuple[Key, Key]] = []
    suggestions: List[Dict[str, Any]] = []
    ordered = sorted(keys)
    for i, key_a in enumerate(ordered):
        for key_b in ordered[i + 1:]:
            if key_a[1] != key_b[1]:
                continue
            score = cosine(vectors[key_a], vectors[key_b])
            if score >= auto_thr:
                unions.append((key_a, key_b))
            elif score >= suggest_thr:
                suggestions.append({"key_a": key_a, "key_b": key_b, "score": score})
    return unions, suggestions


def _most_common(names: List[str]) -> str:
    counts = Counter(names)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def person_groups(
    speaker_rows: List[Dict[str, Any]],
    participants_by_interview: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Auto-link person groups + first-name-only suggestions.

    Only non-provisional speakers whose display_name differs from their handle
    participate — otherwise every unnamed "S1" across the project would
    collapse into one person. Auto when a normalized-name group spans >= 2
    speaker links OR matches a front-matter participant; the participant
    spelling wins the display name, else the most common raw display_name.
    """
    participant_pool: Dict[str, str] = {}
    for interview_id in sorted(participants_by_interview):
        for name in participants_by_interview[interview_id]:
            participant_pool.setdefault(normalize_name(name), name)

    groups: Dict[str, Dict[str, Any]] = {}
    for row in speaker_rows:
        display_name = (row.get("display_name") or "").strip()
        if not display_name or row.get("provisional") or display_name == row.get("handle"):
            continue
        key = normalize_name(display_name)
        group = groups.setdefault(key, {"links": [], "names": []})
        group["links"].append((row["interview_id"], row["speaker_id"]))
        group["names"].append(display_name)

    auto: List[Dict[str, Any]] = []
    auto_by_first_token: Dict[str, List[Dict[str, Any]]] = {}
    auto_keys = set()
    for key, group in sorted(groups.items()):
        in_pool = key in participant_pool
        if len(group["links"]) >= 2 or in_pool:
            entry = {
                "person_key": key,
                "display_name": participant_pool[key] if in_pool else _most_common(group["names"]),
                "method": "front_matter" if in_pool else "exact_name",
                "links": sorted(group["links"]),
            }
            auto.append(entry)
            auto_keys.add(key)
            auto_by_first_token.setdefault(key.split()[0], []).append(entry)

    suggestions: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if key in auto_keys:
            continue
        for candidate in auto_by_first_token.get(key.split()[0], []):
            if candidate["person_key"] == key:
                continue
            for interview_id, speaker_id in sorted(group["links"]):
                suggestions.append({
                    "person_key": candidate["person_key"],
                    "display_name": candidate["display_name"],
                    "interview_id": interview_id,
                    "speaker_id": speaker_id,
                    "speaker_display_name": _most_common(group["names"]),
                    "reason": "first_name_match",
                })
    return auto, suggestions
