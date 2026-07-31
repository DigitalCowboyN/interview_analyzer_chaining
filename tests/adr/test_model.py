import pytest
from tools.adr.model import parse_adr, validate_frontmatter, Adr

GOOD = """---
type: ADR
id: 1
title: EventStoreDB is the single source of truth
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing]
source: docs/architecture/README.md
---
## Context
Body text here.
"""

def test_parse_adr_reads_all_fields():
    adr = parse_adr(GOOD, path="docs/adr/0001-esdb.md")
    assert adr.id == 1
    assert adr.status == "accepted"
    assert adr.tags == ["event-sourcing"]
    assert adr.source == "docs/architecture/README.md"
    assert adr.path == "docs/adr/0001-esdb.md"
    assert "## Context" in adr.body

def test_validate_frontmatter_flags_missing_keys_and_bad_status():
    problems = validate_frontmatter({"type": "ADR", "id": 1, "title": "x", "status": "bogus"})
    joined = " ".join(problems)
    assert "date" in joined            # missing required key
    assert "bogus" in joined           # invalid status

def test_validate_frontmatter_accepts_good():
    fm = {"type": "ADR", "id": 1, "title": "x", "status": "accepted", "date": "2026-07-04"}
    assert validate_frontmatter(fm) == []
