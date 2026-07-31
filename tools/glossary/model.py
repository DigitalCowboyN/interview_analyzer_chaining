from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

from src.ingestion.front_matter import parse_front_matter

RESERVED = {"index.md"}


@dataclass
class Term:
    term: str
    kind: str
    source: Optional[str]
    values: List[str] = field(default_factory=list)
    definition: str = ""
    path: Optional[str] = None
    code_symbol: Optional[str] = None


def parse_term(text: str, path: Optional[str] = None) -> Term:
    fm, offset = parse_front_matter(text)
    if fm is None:
        raise ValueError(f"{path or '<text>'}: missing front matter")
    return Term(
        term=str(fm["term"]),
        kind=str(fm["kind"]),
        source=fm.get("source"),
        values=[str(v) for v in (fm.get("values") or [])],
        code_symbol=fm.get("code_symbol"),
        definition=text[offset:],
        path=path,
    )


def load_glossary(glossary_dir: str) -> List[Term]:
    terms: List[Term] = []
    for p in sorted(glob.glob(os.path.join(glossary_dir, "*.md"))):
        if os.path.basename(p) in RESERVED:
            continue
        try:
            terms.append(parse_term(open(p, encoding="utf-8").read(), path=p))
        except Exception:
            continue  # malformed tolerated (best-effort)
    return terms
