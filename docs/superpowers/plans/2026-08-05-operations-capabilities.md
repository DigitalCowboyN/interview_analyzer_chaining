# Operations Capabilities + `category` Axis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `category: product | operations (…strategic, supporting reserved)` axis to the capability domain and author the operations capabilities (the knowledge-graph program itself) + a file-management product capability.

**Architecture:** Small extensions to `tools/capability/` (reader gains `category` + the shared `CATEGORIES` constant; render groups by category → tier; check validates category + makes `tooling` coverage-mandatory), then a backfill that stamps the 11 product primaries and authors 11 new nodes linked to the `tools.*` code units (added in Round A).

**Tech Stack:** Python 3 stdlib, pytest, Make.

## Global Constraints

- **Non-blocking, always:** every check returns `list[Finding]`; no check raises; `make capability-check` stays exit 0.
- **Interpreter:** `~/.pyenv/shims/python`. **Run tests:** `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.
- **`CATEGORIES` lives in `reader.py`** (the shared, no-cycle home — `render` and `check` both import `reader`; `check`→`render`→`reader` means the constant cannot live in `check`). Value: `["product", "operations", "strategic", "supporting"]`. `strategic`/`supporting` are **reserved** — recognized by the guard, skipped by the renderer until populated; do NOT hardcode a two-value assumption.
- **`category` is a NEW dataclass field appended at the END of `Capability`** (after `path`, default `""`) so existing positional constructions in tests keep working.
- Coverage mandatory roles become `pipeline-layer, surface, tooling`; infra/model advisory.
- DRY, YAGNI, TDD, frequent commits.

---

### Task 1: `reader.py` — `category` field + `CATEGORIES`

**Files:** Modify `tools/capability/reader.py`; Test `tests/capability/test_reader.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `tests/capability/test_reader.py`:

```python
def test_load_parses_category(tmp_path):
    import os
    from tools.capability.reader import load_capabilities, CATEGORIES
    cap = tmp_path / "docs/capabilities/x.md"
    os.makedirs(os.path.dirname(cap), exist_ok=True)
    open(cap, "w").write("---\ntype: Capability\nkind: primary\ntier: core\n"
                         "category: operations\nimplemented_by: [tools.code]\n---\nDoes a thing.\n")
    c = load_capabilities(str(tmp_path))[0]
    assert c.category == "operations"
    assert CATEGORIES[:2] == ["product", "operations"]  # product/operations populated; then reserved
```

- [ ] **Step 2: Run to verify fail** — `AttributeError: 'Capability' object has no attribute 'category'` / `ImportError: CATEGORIES`.

- [ ] **Step 3: Implement** — in `tools/capability/reader.py`:

Add the constant (below the imports):

```python
# The capability category axis — an open, ordered set. product/operations are
# populated; strategic/supporting are reserved (recognized by the guard, skipped by
# the renderer until a node uses them). Adding a value = one edit here.
CATEGORIES = ["product", "operations", "strategic", "supporting"]
```

Append `category` to the dataclass (LAST field, with a default):

```python
@dataclass
class Capability:
    slug: str
    kind: str            # primary | child | variant
    tier: str            # core | enabling  ("" on children/variants — inherited)
    parent: str          # "" on primaries
    implemented_by: List[str]
    statement: str
    path: str
    category: str = ""   # product | operations | … (primaries; children inherit)
```

In `load_capabilities`, add the `category` read to the `Capability(...)` call:

```python
            statement=text[offset:].strip(),
            path=path,
            category=str(fm.get("category", "")),
        ))
```

- [ ] **Step 4: Run tests** — `~/.pyenv/shims/python -m pytest tests/capability/ -p no:cacheprovider -q -o addopts=""` — all green (existing tests unaffected: positional constructions still valid, render/check don't use `category` yet).

- [ ] **Step 5: Commit**

```bash
git add tools/capability/reader.py tests/capability/test_reader.py
git commit -m "feat(capability): add category axis field + CATEGORIES (product/operations, strategic/supporting reserved)"
```

---

### Task 2: `render.py` — group by category → tier

**Files:** Modify `tools/capability/render.py`; Modify `tests/capability/test_render.py`

- [ ] **Step 1: Update the test to the new grouping** — replace `tests/capability/test_render.py` body:

```python
from tools.capability.reader import Capability
from tools.capability.render import render_index

CAPS = [
    Capability("enrich-fragments", "primary", "core", "", ["enrichment"], "Enrich fragments.", "p", "product"),
    Capability("extract-claims", "child", "", "enrich-fragments", ["enrichment.executor"], "Pull claims.", "p", ""),
    Capability("map-the-code", "child", "", "maintain-a-guarded-knowledge-graph", ["tools.code"], "Map the code.", "p", ""),
    Capability("maintain-a-guarded-knowledge-graph", "primary", "core", "", [], "Keep the repo honest.", "p", "operations"),
]


def test_index_groups_by_category_then_tier():
    out = render_index(CAPS)
    assert "## product" in out and "## operations" in out
    assert "### core" in out
    assert "#### enrich-fragments" in out and "enrichment" in out
    # product section precedes operations (CATEGORIES order)
    assert out.index("## product") < out.index("## operations")
    # child nested under its operations primary
    assert out.index("## operations") < out.index("map-the-code")


def test_empty_categories_are_omitted():
    out = render_index(CAPS)
    assert "## strategic" not in out and "## supporting" not in out  # reserved, unpopulated


def test_index_is_deterministic():
    assert render_index(CAPS) == render_index(list(reversed(CAPS)))
```

- [ ] **Step 2: Run to verify fail** — current render groups by tier only (no `## product`).

- [ ] **Step 3: Implement** — replace `render_index` in `tools/capability/render.py`:

```python
from __future__ import annotations

from typing import Dict, List

from tools.capability.reader import CATEGORIES, Capability

_TIERS = ["core", "enabling"]


def render_index(caps: List[Capability]) -> str:
    primaries = [c for c in caps if c.kind == "primary"]
    children_of: Dict[str, List[Capability]] = {}
    for c in caps:
        if c.parent:
            children_of.setdefault(c.parent, []).append(c)
    lines = ["# Capabilities", "",
             "What the system can do, linked to the code map (`../code/`).", ""]
    for category in CATEGORIES:
        cat_primaries = [p for p in primaries if p.category == category]
        if not cat_primaries:
            continue  # reserved/empty category — omit
        lines.append(f"## {category}")
        lines.append("")
        for tier in _TIERS:
            tier_primaries = sorted((p for p in cat_primaries if p.tier == tier), key=lambda c: c.slug)
            if not tier_primaries:
                continue
            lines.append(f"### {tier}")
            lines.append("")
            for p in tier_primaries:
                lines.append(f"#### {p.slug}")
                lines.append(p.statement)
                lines.append("")
                lines.append(f"- **implemented_by:** {', '.join(p.implemented_by) or '—'}")
                for k in sorted(children_of.get(p.slug, []), key=lambda c: c.slug):
                    tag = " _(variant)_" if k.kind == "variant" else ""
                    impl = ', '.join(k.implemented_by) or '—'
                    lines.append(f"- {k.slug}{tag} — {k.statement} ({impl})")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run tests** — `~/.pyenv/shims/python -m pytest tests/capability/test_render.py -p no:cacheprovider -q -o addopts=""` → PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/capability/render.py tests/capability/test_render.py
git commit -m "feat(capability): render groups by category then tier (empty categories omitted)"
```

---

### Task 3: `check.py` — category classification + `tooling` coverage

**Files:** Modify `tools/capability/check.py`; Modify `tests/capability/test_check.py`

- [ ] **Step 1: Update/extend the test** — in `tests/capability/test_check.py`, update the `_cap` helper to take a category and add cases. Append:

```python
from tools.capability.check import CATEGORIES  # re-exported from reader for convenience


def test_classification_flags_primary_missing_category():
    from tools.capability.check import check_classification
    from tools.capability.reader import Capability
    caps = [Capability("p", "primary", "core", "", [], "x", "path", "")]  # category ""
    msgs = " ".join(f.message for f in check_classification(caps))
    assert "category" in msgs


def test_coverage_now_flags_unclaimed_tooling():
    from types import SimpleNamespace as NS
    from tools.capability.check import check_coverage
    from tools.capability.reader import Capability
    nodes = [NS(unit="tools.adr", role="tooling"), NS(unit="utils", role="infrastructure")]
    caps = [Capability("x", "primary", "core", "", ["tools.code"], "x", "p", "operations")]
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "tools.adr" in msgs and "utils" not in msgs  # tooling mandatory; infra still advisory
```

(Also update any existing `check_classification` test whose primary lacks a `category` so it still asserts what it means to — a primary with a valid `category` should not be flagged for category.)

- [ ] **Step 2: Run to verify fail** — `CATEGORIES` not importable from `check`; `tools.adr` not flagged (tooling not yet mandatory).

- [ ] **Step 3: Implement** — in `tools/capability/check.py`:

Change the import + mandatory roles:

```python
from tools.capability.reader import CATEGORIES, code_nodes, load_capabilities, real_code_units
from tools.capability.render import render_index

_MANDATORY_ROLES = ("pipeline-layer", "surface", "tooling")
_VALID_KINDS = ("primary", "child", "variant")
```

In `check_classification`, add the category rule to the primary branch:

```python
        if c.kind == "primary":
            if c.tier not in ("core", "enabling"):
                findings.append(Finding(f"capability: primary {c.slug} has no tier"))
            if c.category not in CATEGORIES:
                findings.append(Finding(f"capability: primary {c.slug} has no/invalid category"))
```

(Replace the existing single `if c.kind == "primary" and c.tier not in …` line with the block above.)

- [ ] **Step 4: Run tests** — `~/.pyenv/shims/python -m pytest tests/capability/test_check.py -p no:cacheprovider -q -o addopts=""` → PASS. (Real-repo `capability-check` will now report the 11 primaries missing `category` + 9 unclaimed `tools.*` — expected until Task 4.)

- [ ] **Step 5: Commit**

```bash
git add tools/capability/check.py tests/capability/test_check.py
git commit -m "feat(capability): validate category on primaries + make tooling coverage-mandatory"
```

---

### Task 4: Backfill — stamp product, author operations + file-management, reconcile

**Files:** Modify the 11 existing `docs/capabilities/<primary>.md`; Create 11 new nodes; regenerate `docs/capabilities/index.md`

- [ ] **Step 1: Stamp `category: product` on the 11 existing primaries.** The primaries (each gets a `category: product` line in its frontmatter, alongside `tier`): `ingest-transcripts, enrich-fragments, extract-insights-via-lenses, resolve-entities-and-people, correct-the-analysis, ask-the-corpus, export-a-portable-bundle, serve-workbench-and-gallery, maintain-event-source-of-truth, project-events-to-graph, provider-strategy-and-focused-calls`. (Children/variants inherit — do NOT add `category` to them.)

Identify primaries programmatically:

```bash
~/.pyenv/shims/python -c "from tools.capability.reader import load_capabilities as L; \
[print(c.path) for c in L('.') if c.kind=='primary']"
```

- [ ] **Step 2: Author the operations tree** — 1 primary + 9 children (per the spec inventory):

```markdown
<!-- docs/capabilities/maintain-a-guarded-knowledge-graph.md -->
---
type: Capability
kind: primary
tier: core
category: operations
implemented_by: []
---
Keep the codebase's own knowledge correct and discoverable: catalog each facet (decisions, code, capabilities, surfaces, vocabulary), guard it against drift, and disclose it just-in-time — so the work compounds.
```

```markdown
<!-- docs/capabilities/map-the-code.md -->
---
type: Capability
kind: child
parent: maintain-a-guarded-knowledge-graph
implemented_by: [tools.code]
---
Classify every package/module by role and derive its dependencies + I/O into a pipeline map.
```

The nine children (`kind: child`, `parent: maintain-a-guarded-knowledge-graph`, no `tier`/`category`) and their `implemented_by`: govern-architectural-decisions→`[tools.adr]`, map-the-code→`[tools.code]`, map-capabilities→`[tools.capability]`, catalog-the-cli-surface→`[tools.cli]`, catalog-the-api-surface→`[tools.api]`, maintain-the-glossary→`[tools.glossary]`, catalog-the-graph-queries→`[tools.graphq]`, catalog-the-prompt-registry→`[tools.prompts]`, disclose-knowledge-and-check-specs→`[tools.knowledge]`. Terse value statement each (mine each `tools/<pkg>/` + `docs/<domain>/` for accuracy).

- [ ] **Step 3: Author the file-management primary**

```markdown
<!-- docs/capabilities/manage-transcript-files.md -->
---
type: Capability
kind: primary
tier: core
category: product
implemented_by: [api]
---
Get source transcripts into the system: upload and list the transcript files an analyst works from (the pre-ingest step).
```

- [ ] **Step 4: Generate + reconcile to clean**

```bash
make capability-index                              # or: ~/.pyenv/shims/python -m tools.capability index
~/.pyenv/shims/python -m tools.capability check     # iterate until: capability-check: clean
```

`clean` = every `implemented_by` resolves (incl. the `tools.*` slugs), every pipeline/surface/**tooling** unit claimed (all 9 `tools.*` now claimed by the operations children), every primary has `tier` + `category`, index in sync.

- [ ] **Step 5: Reconcile neighbours + commit**

```bash
~/.pyenv/shims/python -m tools.knowledge check   # knowledge-check: clean (this spec+plan carry addenda)
~/.pyenv/shims/python -m tools.cli check          # cli-check: clean
git add docs/capabilities/
git commit -m "docs(capability): stamp product + author operations tree (9 tools.*) + file-management; regenerate"
```

---

### Task 5: Capture ADR-0018

**Files:** Create `docs/adr/0018-*.md` (via scaffold); Modify (regenerate) `docs/adr/index.md`, `docs/adr/log.md`

- [ ] **Step 1: Scaffold** — `~/.pyenv/shims/python -m tools.adr new "Adopt the capability category axis and operations capabilities"`.
- [ ] **Step 2: Fill** — `status: accepted`; `date: 2026-08-05`; `source:` = `docs/superpowers/specs/2026-08-05-operations-capabilities-design.md`; `supersedes: []`. Body (durable what/why): the `category` axis is an open, ordered set (`product, operations, strategic, supporting`; product/operations populated, **strategic + supporting reserved** so support tools/systems can be classified later with no code change); orthogonal to `tier`; operations capabilities capture the guarded-knowledge-graph program itself, linked to `tools.*` code units; `tooling` coverage is now mandatory. Refines ADR-0017; supersedes nothing.
- [ ] **Step 3: Regenerate + verify** — `make adr-index`; `~/.pyenv/shims/python -m tools.adr check` → clean apart from the 3 known pre-existing staleness warnings.
- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): ADR-0018 — category axis (strategic/supporting reserved) + operations capabilities"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/capability/ -p no:cacheprovider -q -o addopts=""` — all green.
- [ ] `make capability-check` — clean (all `tools.*` claimed; every primary categorized; links resolve; index in sync).
- [ ] `make capability-index` then `git status` — `docs/capabilities/index.md` regenerates identically.
- [ ] `make knowledge-check` + `make cli-check` + `make adr-check` — clean (adr apart from the 3 known warnings).
- [ ] Open `docs/capabilities/index.md` — a `## product` section (incl. `manage-transcript-files`) and a `## operations` section (the knowledge-graph program); NO `## strategic` / `## supporting` (reserved, unpopulated).

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | the domain being extended — `category` axis, 11 new nodes, tooling coverage | — |
| code | yes (read-only) | operations `implemented_by` → `tools.*` (Round A nodes); `load_units` registry | no code-map change |
| adr | yes | ADR-0018 (Task 5); refines ADR-0017 | — |
| cli / glossary / api / prompts / graph-queries | no | — | no new targets/vocabulary/surface/prompt/query |

**Verdict:** reconciled — capabilities (subject) + code (read-only via Round A) consulted; adr reconciled in Task 5.
