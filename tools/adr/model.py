"""The `Adr` record and its front-matter parsing/validation: `parse_adr` turns an ADR
file's text into an `Adr`, and `validate_frontmatter` reports missing required keys or an
invalid `status` — the shared model every other `tools.adr` module reads or checks against."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.front_matter import parse_front_matter

VALID_STATUS = {"proposed", "accepted", "superseded", "deprecated"}
REQUIRED_KEYS = ("type", "id", "title", "status", "date")


@dataclass
class Adr:
    id: int
    title: str
    status: str
    date: str
    supersedes: List[int] = field(default_factory=list)
    superseded_by: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    governs: List[str] = field(default_factory=list)
    source: Optional[str] = None
    path: Optional[str] = None
    body: str = ""


def validate_frontmatter(fm: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in fm:
            problems.append(f"missing required key: {key}")
    status = fm.get("status")
    if status is not None and status not in VALID_STATUS:
        problems.append(f"invalid status: {status!r} (want one of {sorted(VALID_STATUS)})")
    return problems


def _int_list(value: Any) -> List[int]:
    return [int(x) for x in (value or [])]


def parse_adr(text: str, path: Optional[str] = None) -> Adr:
    fm, offset = parse_front_matter(text)
    if fm is None:
        raise ValueError(f"{path or '<text>'}: missing front matter")
    return Adr(
        id=int(fm["id"]),
        title=str(fm["title"]),
        status=str(fm["status"]),
        date=str(fm["date"]),
        supersedes=_int_list(fm.get("supersedes")),
        superseded_by=_int_list(fm.get("superseded_by")),
        tags=list(fm.get("tags") or []),
        governs=[str(p) for p in (fm.get("governs") or [])],
        source=fm.get("source"),
        path=path,
        body=text[offset:],
    )
