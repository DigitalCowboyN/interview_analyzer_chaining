"""Use-case domain reader: parse docs/use-cases/*.md frontmatter into UseCase
records (slug, form, fulfilled_by, statement) for the graph and coverage checks.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List

from src.ingestion.front_matter import parse_front_matter

# `form` is an open, ordered set (like the capability `category` axis). Add a value here.
FORMS = ["user-story", "feature", "requirement", "use-case"]


@dataclass
class UseCase:
    slug: str
    form: str
    category: str
    actor: str
    statement: str
    path: str
    acceptance_criteria: List[str] = field(default_factory=list)
    fulfilled_by: List[str] = field(default_factory=list)
    level: str = ""              # Cockburn: user-goal | summary | subfunction
    preconditions: str = ""
    main_scenario: str = ""
    extensions: str = ""
    end_conditions: str = ""


def load_use_cases(root: str = ".", uc_dir: str = "docs/use-cases") -> List[UseCase]:
    ucs: List[UseCase] = []
    for path in sorted(glob.glob(os.path.join(root, uc_dir, "*.md"))):
        base = os.path.basename(path)
        if base in ("index.md", "README.md"):
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") != "UseCase":
            continue
        ucs.append(UseCase(
            slug=os.path.splitext(base)[0],
            form=str(fm.get("form", "")),
            category=str(fm.get("category", "")),
            actor=str(fm.get("actor", "")),
            statement=text[offset:].strip(),
            path=path,
            acceptance_criteria=list(fm.get("acceptance_criteria") or []),
            fulfilled_by=list(fm.get("fulfilled_by") or []),
            level=str(fm.get("level", "")),
            preconditions=str(fm.get("preconditions", "")),
            main_scenario=str(fm.get("main_scenario", "")),
            extensions=str(fm.get("extensions", "")),
            end_conditions=str(fm.get("end_conditions", "")),
        ))
    return ucs
