# Capabilities as Intent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Capture the "capabilities are durable intent; implementation is a derived, replaceable link" model — as an ADR + a capability-domain concept doc — and add one concrete aspirational capability (`import-transcripts`) demonstrating that intent can outrun implementation.

**Architecture:** Schema-free / docs-only. No `tools/` code change — the guard already permits an `implemented_by: []` intent capability (the operations primary already ships that way). Three authored artifacts + a regenerated `docs/capabilities/index.md`.

**Tech Stack:** Markdown, `tools.capability` / `tools.adr` CLIs, Make.

## Global Constraints

- **No code/schema change.** If any task seems to need one, stop — it's out of scope.
- **Interpreter:** `~/.pyenv/shims/python`.
- Authored value statements state **intent** (what/why), never "not implemented" (that's derived from the empty links).
- DRY, YAGNI.

---

### Task 1: The `import-transcripts` intent capability

**Files:** Create `docs/capabilities/import-transcripts.md`; regenerate `docs/capabilities/index.md`

- [ ] **Step 1: Author the node**

```markdown
<!-- docs/capabilities/import-transcripts.md -->
---
type: Capability
kind: primary
tier: core
category: product
implemented_by: []
---
Let an analyst bring source transcripts into the system to be analysed.
```

(Empty `implemented_by` is deliberate and legitimate — the product intent exists; no import/upload feature implements it today, ingestion only reads a pre-populated directory. The statement describes the intent; the un-reached-ness is derived from the empty links, not stated.)

- [ ] **Step 2: Regenerate + reconcile**

```bash
~/.pyenv/shims/python -m tools.capability index    # regenerate docs/capabilities/index.md
~/.pyenv/shims/python -m tools.capability check     # expect: capability-check: clean
```

`clean` because: `check_links` skips empty `implemented_by`; `check_coverage` is code→capability (an intent-only node adds no obligation); `check_classification` passes (kind/tier/category all set); index in sync. The node renders under `## product` → `### core`.

- [ ] **Step 3: Commit**

```bash
git add docs/capabilities/import-transcripts.md docs/capabilities/index.md
git commit -m "docs(capability): add import-transcripts intent capability (no implementation yet — intent outruns code)"
```

---

### Task 2: The capability-domain concept doc

**Files:** Create `docs/capabilities/README.md`; Modify `docs/index.md` (point the capabilities row at the concept doc)

- [ ] **Step 1: Author `docs/capabilities/README.md`** — the human/agent-facing "how to think about capabilities here." Cover, concisely (mine the spec `2026-08-06-capabilities-as-intent-design.md`):
  - A capability is **durable intent** — never "built," only *currently implemented* by a replaceable iteration.
  - `primary` / `child` / `variant` are **all intent** (broad / narrower / alternative *what*); `parent` is decomposition, not a how-chain.
  - **Code (`implemented_by`) is the only HOW**; the degree of implementation is **derived** from the links — **empty/partial is legitimate** (an intent current code only partly reaches).
  - The three artifacts table: **capability** = what/why · **ADR/spec** = how we decided (the how-definition) · **code** = current implementation. No middle capability layer.
  - `implements` inverse is **derived** (not authored in code); surfacing it is the graph-links topic.
  - Categories are an open set (product/operations populated; strategic/supporting reserved); capability↔use-case is indirect (round 3).
  - Point to `index.md` (the generated catalog) as the live map.

  (This is inert to the tooling — `load_capabilities` skips any file without `type: Capability` frontmatter, and the guard only compares `index.md`.)

- [ ] **Step 2: Point the cascade at it** — in `docs/index.md`, update the capabilities row so the concept doc is the landing page, e.g.:

```markdown
| [capabilities/](capabilities/README.md) | what the system can do (value-framed intent), linked to the code map | `make capability-check` |
```

- [ ] **Step 3: Verify + commit**

```bash
~/.pyenv/shims/python -m tools.knowledge check   # knowledge-check: clean (cascade still references capabilities/)
git add docs/capabilities/README.md docs/index.md
git commit -m "docs(capability): concept doc — capabilities are intent, implementation is a derived link"
```

---

### Task 3: Capture ADR-0019

**Files:** Create `docs/adr/0019-*.md` (via scaffold); regenerate `docs/adr/index.md`, `docs/adr/log.md`

- [ ] **Step 1: Scaffold** — `~/.pyenv/shims/python -m tools.adr new "Capabilities are durable intent; implementation is a derived, replaceable link"`.
- [ ] **Step 2: Fill** — `status: accepted`; `date: 2026-08-06`; `source:` = `docs/superpowers/specs/2026-08-06-capabilities-as-intent-design.md`; `supersedes: []`. Body (durable what/why): a capability is durable intent, never "built" — only currently implemented by a replaceable iteration; implementation degree lives entirely in the `implemented_by` links and is derived, never an authored status (empty/partial is legitimate); primary/child/variant are all intent (narrower/alternative what), code is the sole how, and how-decisions live in ADRs/specs (no middle capability layer); the `implements` inverse is derived; capability↔use-case is indirect (round 3). Refines ADR-0017/0018; supersedes nothing.
- [ ] **Step 3: Regenerate + verify** — `~/.pyenv/shims/python -m tools.adr index`; `~/.pyenv/shims/python -m tools.adr check` → clean apart from the 3 known pre-existing staleness warnings.
- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): ADR-0019 — capabilities are durable intent; implementation is a derived link"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m tools.capability check` — clean (incl. the empty-link `import-transcripts` node).
- [ ] `~/.pyenv/shims/python -m tools.capability index` then `git status` — `docs/capabilities/index.md` regenerates identically.
- [ ] `make knowledge-check` — clean; `make adr-check` — clean apart from the 3 known warnings.
- [ ] `docs/capabilities/index.md` shows `import-transcripts` under `## product` → `### core`; `docs/capabilities/README.md` renders on GitHub folder view; the cascade row points at it.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | concept README + `import-transcripts` intent node; no schema change | the subject |
| adr | yes | ADR-0019; refines 0017/0018 | — |
| code | no | — | `implements` inverse / code-map surfacing deferred to graph-links |
| cli / glossary / api / prompts / graph-queries | no | — | no target/vocabulary/surface/prompt/query change |

**Verdict:** reconciled — capabilities (subject) + adr consulted; derived-inverse surfacing deferred to the graph-links topic.
