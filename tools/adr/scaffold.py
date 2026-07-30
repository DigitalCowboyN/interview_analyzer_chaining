from __future__ import annotations

import os
import re

from tools.adr.index import load_bundle

_TEMPLATE = """---
type: ADR
id: {id}
title: {title}
status: proposed
date: {date}
supersedes: []
superseded_by: []
tags: []
source:
---
## Context

## Decision

## Consequences

## Alternatives considered
"""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def next_id(adr_dir: str) -> int:
    adrs = load_bundle(adr_dir) if os.path.isdir(adr_dir) else []
    return (max((a.id for a in adrs), default=0)) + 1


def new_adr(adr_dir: str, title: str, date: str = "TODO-SET-DATE") -> str:
    os.makedirs(adr_dir, exist_ok=True)
    nid = next_id(adr_dir)
    path = os.path.join(adr_dir, f"{nid:04d}-{_slug(title)}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_TEMPLATE.format(id=nid, title=title, date=date))
    return path
