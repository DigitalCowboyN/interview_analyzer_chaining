# Capabilities Domain (#8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catalogue what the system can do as value-framed `Capability` nodes linked to the code map, guarded by a non-blocking `make capability-check`.

**Architecture:** A new `tools/capability/` domain in the established `reader → render → check → CLI` shape. Flat `type: Capability` markdown files under `docs/capabilities/` (kind/tier/parent/implemented_by), a generated `index.md`, and a guard that reconciles `implemented_by` against the code map's unit registry + checks coverage of pipeline/surface units. Adds the domain to the knowledge cascade + registry.

**Tech Stack:** Python 3 stdlib + `src.ingestion.front_matter.parse_front_matter` + `tools.code.reader` (the code-unit registry), pytest, Make.

## Global Constraints

- **Non-blocking, always:** every check returns `list[Finding]`; no check raises; every CLI command `return 0`. `make capability-check` must never exit non-zero.
- **Interpreter:** `~/.pyenv/shims/python`. **Run tests:** `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.
- **Reuse the code registry:** valid `implemented_by` targets = `tools.code.reader.packages(root)` ∪ `tools.code.reader.KEY_MODULES` (30 units). Coverage roles come from `tools.code.reader.load_units(root)` (each CodeUnit has `.unit` + `.role`). Never hardcode the code-unit list in `tools/capability`.
- **Coverage scope:** a CodeUnit with `role in {pipeline-layer, surface}` must be claimed by some capability (its own slug OR its parent package slug in some `implemented_by`); all other roles (infrastructure, model, agent, tooling) are **advisory — never flagged**.
- **Slug = filename stem** (`ingest-transcripts.md` → slug `ingest-transcripts`); no `slug:` frontmatter key.
- **Idiom:** `Finding` is a message-only dataclass; CLI prints `capability-check: N warning(s):` / `capability-check: clean`. Mirror `tools/code/`.
- The authoritative capability inventory is the table in `docs/superpowers/specs/2026-08-05-capabilities-domain-design.md` — Task 5 authors exactly those nodes.
- DRY, YAGNI, TDD, frequent commits.

---

### Task 1: `tools/capability/reader.py`

**Files:** Create `tools/capability/__init__.py` (empty), `tools/capability/reader.py`; Test `tests/capability/test_reader.py`

**Interfaces:**
- Produces: `@dataclass Capability(slug, kind, tier, parent, implemented_by, statement, path)`; `load_capabilities(root=".", cap_dir="docs/capabilities") -> list[Capability]`; `real_code_units(root=".") -> set[str]`; `code_nodes(root=".") -> list` (CodeUnit passthrough for coverage roles).

- [ ] **Step 1: Write the failing test**

```python
# tests/capability/test_reader.py
import os
from tools.capability.reader import Capability, load_capabilities, real_code_units, code_nodes


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def test_load_parses_node_and_links(tmp_path):
    cap = tmp_path / "docs/capabilities/enrich-fragments.md"
    _write(str(cap), "---\ntype: Capability\nkind: primary\ntier: core\n"
                     "implemented_by: [enrichment, agents]\n---\nEnrich each fragment.\n")
    _write(str(tmp_path / "docs/capabilities/index.md"), "# Capabilities\n")  # skipped
    caps = load_capabilities(str(tmp_path))
    assert len(caps) == 1
    c = caps[0]
    assert c.slug == "enrich-fragments" and c.kind == "primary" and c.tier == "core"
    assert c.implemented_by == ["enrichment", "agents"]
    assert c.statement == "Enrich each fragment."


def test_load_skips_non_capability_files(tmp_path):
    _write(str(tmp_path / "docs/capabilities/notes.md"), "# just notes, no frontmatter\n")
    assert load_capabilities(str(tmp_path)) == []


def test_real_code_units_includes_packages_and_key_modules():
    units = real_code_units(".")
    assert "enrichment" in units and "lens.engine" in units and "ask.reader" in units


def test_code_nodes_carry_roles():
    nodes = code_nodes(".")
    roles = {n.unit: n.role for n in nodes}
    assert roles.get("api") == "surface" and roles.get("lens") == "pipeline-layer"
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError: tools.capability`.

- [ ] **Step 3: Implement**

```python
# tools/capability/reader.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

from src.ingestion.front_matter import parse_front_matter
from tools.code.reader import KEY_MODULES, load_units, packages


@dataclass
class Capability:
    slug: str
    kind: str            # primary | child | variant
    tier: str            # core | enabling  ("" on children/variants — inherited)
    parent: str          # "" on primaries
    implemented_by: List[str]
    statement: str
    path: str


def load_capabilities(root: str = ".", cap_dir: str = "docs/capabilities") -> List[Capability]:
    caps: List[Capability] = []
    for path in sorted(glob.glob(os.path.join(root, cap_dir, "*.md"))):
        if os.path.basename(path) == "index.md":
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") != "Capability":
            continue
        caps.append(Capability(
            slug=os.path.splitext(os.path.basename(path))[0],
            kind=str(fm.get("kind", "")),
            tier=str(fm.get("tier", "")),
            parent=str(fm.get("parent", "")),
            implemented_by=list(fm.get("implemented_by") or []),
            statement=text[offset:].strip(),
            path=path,
        ))
    return caps


def real_code_units(root: str = ".") -> set:
    """Valid implemented_by targets — the code map's unit registry (single source)."""
    return set(packages(root)) | set(KEY_MODULES)


def code_nodes(root: str = "."):
    """CodeUnit nodes (with .unit + .role) for the coverage check."""
    return load_units(root)
```

- [ ] **Step 4: Run tests** → PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/capability/__init__.py tools/capability/reader.py tests/capability/test_reader.py
git commit -m "feat(capability): reader — Capability nodes + code-unit registry reuse"
```

---

### Task 2: `tools/capability/render.py`

**Files:** Create `tools/capability/render.py`; Test `tests/capability/test_render.py`

**Interfaces:**
- Consumes: `Capability`.
- Produces: `render_index(caps: list[Capability]) -> str` — grouped `tier` → primary (statement + implemented_by) → children/variants. Deterministic (sorted).

- [ ] **Step 1: Write the failing test**

```python
# tests/capability/test_render.py
from tools.capability.reader import Capability
from tools.capability.render import render_index

CAPS = [
    Capability("enrich-fragments", "primary", "core", "", ["enrichment", "agents"], "Enrich fragments.", "p"),
    Capability("extract-claims", "child", "", "enrich-fragments", ["enrichment.executor"], "Pull claims.", "p"),
    Capability("project-events-to-graph", "primary", "enabling", "", ["projections"], "Build the read model.", "p"),
]


def test_index_groups_by_tier_and_nests_children():
    out = render_index(CAPS)
    assert "## core" in out and "## enabling" in out
    assert "### enrich-fragments" in out and "enrichment, agents" in out
    # child nested under its primary, with its own implemented_by
    assert "extract-claims" in out and "enrichment.executor" in out
    assert out.index("### enrich-fragments") < out.index("## enabling")


def test_index_is_deterministic():
    assert render_index(CAPS) == render_index(list(reversed(CAPS)))
```

- [ ] **Step 2: Run to verify fail** — no `render_index`.

- [ ] **Step 3: Implement**

```python
# tools/capability/render.py
from __future__ import annotations

from typing import Dict, List

from tools.capability.reader import Capability

_TIERS = ["core", "enabling"]


def render_index(caps: List[Capability]) -> str:
    primaries = [c for c in caps if c.kind == "primary"]
    children_of: Dict[str, List[Capability]] = {}
    for c in caps:
        if c.parent:
            children_of.setdefault(c.parent, []).append(c)
    lines = ["# Capabilities", "",
             "What the system can do, linked to the code map (`../code/`).", ""]
    for tier in _TIERS:
        tier_primaries = sorted((p for p in primaries if p.tier == tier), key=lambda c: c.slug)
        if not tier_primaries:
            continue
        lines.append(f"## {tier}")
        lines.append("")
        for p in tier_primaries:
            lines.append(f"### {p.slug}")
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

- [ ] **Step 4: Run tests** → PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/capability/render.py tests/capability/test_render.py
git commit -m "feat(capability): catalogue renderer (grouped by tier, children nested)"
```

---

### Task 3: `tools/capability/check.py`

**Files:** Create `tools/capability/check.py`; Test `tests/capability/test_check.py`

**Interfaces:**
- Consumes: `load_capabilities`, `real_code_units`, `code_nodes`, `render_index`.
- Produces: `Finding`; `check_links(caps, valid_units)`; `check_coverage(caps, nodes)`; `check_classification(caps)`; `check_index_sync(index_path, caps)`; `run_all(root=".")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/capability/test_check.py
from types import SimpleNamespace as NS
from tools.capability.reader import Capability
from tools.capability.check import (
    Finding, check_links, check_coverage, check_classification, check_index_sync, run_all,
)
from tools.capability.render import render_index


def _cap(slug, kind="primary", tier="core", parent="", impl=None):
    return Capability(slug, kind, tier, parent, impl or [], f"{slug} does a thing.", "p")


def test_links_flag_unknown_unit():
    caps = [_cap("x", impl=["enrichment", "not_a_unit"])]
    msgs = " ".join(f.message for f in check_links(caps, {"enrichment"}))
    assert "not_a_unit" in msgs


def test_coverage_flags_unclaimed_pipeline_unit_but_not_infra():
    nodes = [NS(unit="lens", role="pipeline-layer"), NS(unit="utils", role="infrastructure")]
    caps = [_cap("x", impl=["ingestion"])]  # claims neither
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "lens" in msgs and "utils" not in msgs  # infra advisory, not flagged


def test_coverage_parent_package_covers_key_module():
    nodes = [NS(unit="lens.engine", role="pipeline-layer")]
    caps = [_cap("x", impl=["lens"])]  # claims the package → covers the module
    assert check_coverage(caps, nodes) == []


def test_classification_flags_missing_kind_tier_parent():
    caps = [
        _cap("noprimarytier", tier=""),                     # primary w/o tier
        _cap("orphan", kind="child", tier="", parent=""),   # child w/o parent
    ]
    msgs = " ".join(f.message for f in check_classification(caps))
    assert "noprimarytier" in msgs and "orphan" in msgs


def test_index_sync_flags_stale(tmp_path):
    caps = [_cap("x", impl=["enrichment"])]
    idx = tmp_path / "index.md"
    idx.write_text("stale", encoding="utf-8")
    assert check_index_sync(str(idx), caps)  # non-empty
    idx.write_text(render_index(caps), encoding="utf-8")
    assert check_index_sync(str(idx), caps) == []


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)
```

- [ ] **Step 2: Run to verify fail** — no `tools.capability.check`.

- [ ] **Step 3: Implement**

```python
# tools/capability/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.capability.reader import code_nodes, load_capabilities, real_code_units
from tools.capability.render import render_index

_MANDATORY_ROLES = ("pipeline-layer", "surface")
_VALID_KINDS = ("primary", "child", "variant")


@dataclass
class Finding:
    message: str


def check_links(caps, valid_units) -> List[Finding]:
    findings: List[Finding] = []
    for c in caps:
        for u in c.implemented_by:
            if u not in valid_units:
                findings.append(Finding(
                    f"capability: {c.slug} implemented_by unknown code unit '{u}'"))
    return findings


def check_coverage(caps, nodes) -> List[Finding]:
    claimed = set()
    for c in caps:
        claimed.update(c.implemented_by)
    findings: List[Finding] = []
    for n in nodes:
        if n.role not in _MANDATORY_ROLES:
            continue  # infrastructure/model/agent/tooling — advisory, never flagged
        parent_pkg = n.unit.split(".")[0]
        if n.unit not in claimed and parent_pkg not in claimed:
            findings.append(Finding(
                f"capability: code unit {n.unit} ({n.role}) is claimed by no capability"))
    return findings


def check_classification(caps) -> List[Finding]:
    slugs = {c.slug for c in caps}
    findings: List[Finding] = []
    for c in caps:
        if c.kind not in _VALID_KINDS:
            findings.append(Finding(f"capability: {c.slug} has no/invalid kind"))
        if c.kind == "primary" and c.tier not in ("core", "enabling"):
            findings.append(Finding(f"capability: primary {c.slug} has no tier"))
        if c.kind in ("child", "variant"):
            if not c.parent:
                findings.append(Finding(f"capability: {c.kind} {c.slug} has no parent"))
            elif c.parent not in slugs:
                findings.append(Finding(
                    f"capability: {c.slug} parent '{c.parent}' does not resolve"))
    return findings


def check_index_sync(index_path: str, caps) -> List[Finding]:
    want = render_index(caps)
    have = open(index_path, encoding="utf-8", errors="ignore").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("capability: docs/capabilities/index.md out of sync — run make capability-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    caps = load_capabilities(root)
    findings: List[Finding] = []
    findings += check_links(caps, real_code_units(root))
    findings += check_coverage(caps, code_nodes(root))
    findings += check_classification(caps)
    findings += check_index_sync(os.path.join(root, "docs/capabilities/index.md"), caps)
    return findings
```

- [ ] **Step 4: Run tests** → PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/capability/check.py tests/capability/test_check.py
git commit -m "feat(capability): guard — links, coverage (pipeline/surface), classification, index-sync"
```

---

### Task 4: CLI + Makefile + cascade/registry wiring

**Files:** Create `tools/capability/__main__.py`; Modify `Makefile`, `tools/knowledge/check.py`, `docs/index.md`; Test `tests/capability/test_cli.py`

**Interfaces:**
- Consumes: `run_all`, `load_capabilities`, `render_index`.
- Produces: `python -m tools.capability {index|check}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/capability/test_cli.py
import subprocess
import sys


def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.capability", "check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "capability-check" in proc.stdout


def test_capabilities_in_knowledge_registry():
    from tools.knowledge.check import DOMAINS
    slugs = {slug for slug, _ in DOMAINS}
    assert "capabilities" in slugs
```

- [ ] **Step 2: Run to verify fail** — no `tools.capability.__main__`; `capabilities` not in `DOMAINS`.

- [ ] **Step 3: Implement**

```python
# tools/capability/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.capability.check import run_all
from tools.capability.reader import load_capabilities
from tools.capability.render import render_index

INDEX = "docs/capabilities/index.md"


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_capabilities()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"capability-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("capability-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.capability")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` immediately after the `knowledge-check` target:

```makefile
.PHONY: capability-index
capability-index: ## Regenerate docs/capabilities/index.md (the capability catalogue)
	@$(PYTHON) -m tools.capability index

.PHONY: capability-check
capability-check: ## Reconcile capabilities vs the code map + coverage (non-blocking)
	@$(PYTHON) -m tools.capability check
```

In `tools/knowledge/check.py`, add the domain to `DOMAINS` (keep alphabetical-ish, after `("cli", "cli")`... place logically):

```python
    ("cli", "cli"),
    ("code", "code"),
    ("capabilities", "capability"),
```

In `docs/index.md`, add a row to the domain table (under `code/`):

```markdown
| [capabilities/](capabilities/index.md) | what the system can do (value-framed), linked to the code map | `make capability-check` |
```

- [ ] **Step 4: Run test + smoke** — `~/.pyenv/shims/python -m pytest tests/capability/test_cli.py -p no:cacheprovider -q -o addopts=""` → PASS. Smoke: `~/.pyenv/shims/python -m tools.capability check` → exit 0 (findings expected until Task 5: coverage warnings for all pipeline/surface packages + missing index). `~/.pyenv/shims/python -m tools.knowledge check` → `knowledge-check: clean` (the `capabilities/` cascade row was just added, and this spec + plan already carry `## Knowledge-graph check` addenda). Confirm NO exception.

- [ ] **Step 5: Commit**

```bash
git add tools/capability/__main__.py Makefile tools/knowledge/check.py docs/index.md tests/capability/test_cli.py
git commit -m "feat(capability): CLI + make targets + cascade row + knowledge registry entry"
```

---

### Task 5: Backfill the capability map

**Files:** Create `docs/capabilities/<slug>.md` × ~36 (11 primaries + ~25 children/variants) + generated `docs/capabilities/index.md`; Modify (regenerate) `docs/cli/index.md`

**The node set is the inventory table in the spec** (`docs/superpowers/specs/2026-08-05-capabilities-domain-design.md`). Author exactly those primaries, children, and variants.

- [ ] **Step 1: Author the primaries** — one file per primary, e.g.:

```markdown
<!-- docs/capabilities/ingest-transcripts.md -->
---
type: Capability
kind: primary
tier: core
implemented_by: [ingestion, ingestion.orchestrator, agents]
---
Turn raw transcript files into structured, speaker-attributed, stitched utterances the rest of the system analyses.
```

```markdown
<!-- docs/capabilities/project-events-to-graph.md -->
---
type: Capability
kind: primary
tier: enabling
implemented_by: [projections]
---
Replay the event log in causal order into Neo4j as the sole writer, maintaining the queryable read model.
```

Primary → tier → implemented_by (from the spec table): ingest-transcripts (core), enrich-fragments (core), extract-insights-via-lenses (core), resolve-entities-and-people (core), correct-the-analysis (core), ask-the-corpus (core), export-a-portable-bundle (core), serve-workbench-and-gallery (core); maintain-event-source-of-truth (enabling), project-events-to-graph (enabling), provider-strategy-and-focused-calls (enabling).

- [ ] **Step 2: Author the children/variants** — `kind: child` (or `variant`), `parent: <primary-slug>`, its own `implemented_by`, no `tier`. e.g.:

```markdown
<!-- docs/capabilities/infer-speakers.md -->
---
type: Capability
kind: child
parent: ingest-transcripts
implemented_by: [ingestion.speaker_inference, agents]
---
Infer which speaker each utterance belongs to when the transcript doesn't label them.
```

Children per primary (from the spec inventory): ingest → parse-fragments, infer-speakers, stitch-utterances, segment-conversation · enrich → extract-claims, classify-dimensions, tag-topics-keywords · lenses → run-lens-engine (+ variant `per-lens-extractors`) · resolve → canonicalize-entities, resolve-persons, merge-split-link-alias · correct → edit-text, rename-reattribute-speakers, remove-segments, override-lens-items, correct-resolution · ask → hybrid-retrieval, cited-synthesis · export → assemble-bundle, render-bundle · serve → run-read-queries, workbench-write, gallery-read, live-notifications · provider-strategy → variants `chat-failover`, `pinned-embeddings`. Give each a terse value statement and an `implemented_by` drawn from the parent's code units (mine `docs/code/`, `system-overview.md`, `data-flow.md` for accuracy — a child may narrow to a key module, e.g. `run-lens-engine` → `[lens.engine]`).

- [ ] **Step 3: Generate + reconcile to clean**

```bash
make capability-index                       # writes docs/capabilities/index.md
~/.pyenv/shims/python -m tools.capability check   # iterate until: capability-check: clean
```

`clean` = every `implemented_by` resolves, every pipeline/surface package is claimed, every node classified, index in sync. If coverage flags a unit, either a capability should claim it or (rare) it is genuinely orphaned — report rather than inventing a bogus link.

- [ ] **Step 4: Reconcile the neighbouring domains**

```bash
make cli-index                               # capability-* targets enter the CLI catalogue
~/.pyenv/shims/python -m tools.cli check      # cli-check: clean
~/.pyenv/shims/python -m tools.knowledge check  # knowledge-check: clean (cascade row present; this plan+spec carry addenda)
```

- [ ] **Step 5: Commit**

```bash
git add docs/capabilities/ docs/cli/
git commit -m "docs(capability): backfill the capability map (11 primaries + children/variants) + generated index"
```

---

### Task 6: Capture ADR-0017

**Files:** Create `docs/adr/0017-*.md` (via scaffold); Modify (regenerate) `docs/adr/index.md`, `docs/adr/log.md`

- [ ] **Step 1: Scaffold** — `~/.pyenv/shims/python -m tools.adr new "Adopt a capabilities domain linked to the code map"`.
- [ ] **Step 2: Fill** — `status: accepted`; `date: 2026-08-05`; `source:` = `docs/superpowers/specs/2026-08-05-capabilities-domain-design.md`; `supersedes: []`. Body (durable what/why): capabilities are the value-framed "what" layer (round 2 of the vertical stack), stable and implementation-independent, linked to CodeUnit nodes via `implemented_by`; core/enabling tiers; guard reconciles links + pipeline/surface coverage, non-blocking; use-case links deferred. Does not supersede any ADR.
- [ ] **Step 3: Regenerate + verify** — `make adr-index`; `~/.pyenv/shims/python -m tools.adr check` → clean apart from the 3 known pre-existing staleness warnings.
- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): ADR-0017 — adopt a capabilities domain linked to the code map"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/capability/ tests/knowledge/ -p no:cacheprovider -q -o addopts=""` — all green.
- [ ] `make capability-check` — clean (all links resolve; all 9 pipeline/surface packages claimed; every node classified; index in sync).
- [ ] `make capability-index` then `git status` — `docs/capabilities/index.md` regenerates identically.
- [ ] `make knowledge-check` — clean (cascade row + registry entry present; this spec + plan carry `## Knowledge-graph check` addenda).
- [ ] `make cli-check` — clean (`capability-*` targets catalogued).
- [ ] `make adr-check` — clean apart from the 3 known pre-existing staleness warnings.
- [ ] Open `docs/capabilities/index.md` — core tier then enabling; each primary shows its `implemented_by` and nested children.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | the new domain itself | — |
| code | yes (read-only) | `implemented_by` slugs + coverage roles read from `tools.code.reader`; no `src/` change | the link/coverage registry |
| cli | yes | `capability-*` targets → `cli-index`; `cli-check` clean (Task 5) | new make targets |
| adr | yes | ADR-0017 captured (Task 6) | — |
| glossary / api / prompts / graph-queries | no | — | no code-vocabulary/surface/prompt/query change |

**Verdict:** reconciled — code (read-only) consulted; cli + adr reconciled in-plan (Tasks 5–6).
