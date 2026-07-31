# ADR↔Code Linking + Drift Guard (design)

**Status:** approved by owner 2026-07-31 (brainstorm dialogue).
**Program:** first sub-project ("B") of the *guarded knowledge graph over the
codebase* — knowledge nodes linked to the code they govern, via OKF links-as-edges,
with a non-blocking drift guard. Builds directly on the ADR + OKF knowledge system
(`docs/adr/`, `tools/adr/`, merged in PR #19).

## Goal

Make architectural decisions and code **mutually navigable and drift-checked**:

1. Stand at a decision → see the code it governs.
2. Stand at a piece of code → see the decisions that govern it.
3. Have the guard flag when the two sides disagree, or when governed code changes
   underneath a decision.

This is the first concrete instance of the graph layer. The link mechanism is
built **generic on purpose** so later sub-projects (API surface, CLI surface, code
documentation) reuse it unchanged — only the node *type* differs.

## Design decisions (locked in brainstorm)

- **Both sides, cross-checked** — the ADR declares what it governs *and* the code
  carries a back-marker; the guard verifies the two agree. (Rejected: frontmatter-only
  with a generated reverse index — non-invasive but the reverse direction is only as
  honest as the ADR's own claims; rejected: markers-only — loses the ADR-side
  declaration.)
- **Path granularity (directory + file)** — governance attaches to paths, not
  symbols. An architectural decision governs a subsystem or a file, rarely one
  function; symbol-level is rename-fragile and needs AST resolution. (Symbol-level
  is a documented v-next, not v1.)
- **Docstring markers** — the code-side back-link lives in the module docstring
  (file target) or the package `__init__.py` docstring / directory `README.md`
  (directory target). Greppable, in-place, survives edits; no sidecar files.
- **Staleness stays, informational** — "governed code changed since the ADR" is
  surfaced as a non-blocking finding, not dropped. Noisy in active areas by design;
  it is a prompt to revisit, never a gate.

## The two sides of each edge

### ADR side — `governs` frontmatter

A new **optional** frontmatter field on any ADR: a list of repo-relative paths.
Directories end with `/`; files do not.

```yaml
---
type: ADR
id: 3
title: The projection service is the sole writer to Neo4j
status: accepted
date: 2026-07-04
governs:
  - src/projections/
  - src/services/projection_service.py
# ... existing fields (supersedes, source, tags, ...) unchanged
---
```

`governs` is optional — an ADR that governs no code (e.g. a process decision) simply
omits it and draws no findings.

### Code side — `governed-by` marker

A greppable line naming ADR ids, placed by the granularity rule:

- **File target** → the module docstring of that file.
- **Directory target** → the package `__init__.py` docstring, or a `README.md` in
  the directory.

```python
"""Projection service — sole writer to Neo4j.

governed-by: ADR-0003, ADR-0001
"""
```

Parsed by a simple line-scan for `governed-by:` followed by `ADR-NNNN` tokens (the
same `ADR[-\s]?\d+` shape the existing `check_specs_reference_adr` already uses). No
AST, no import.

## Generated reverse index — `docs/adr/by-code.md`

`adr-index` gains a third generated artifact alongside `index.md` / `log.md`:
`by-code.md`, a table of `code path → governing ADR(s)`, built from every ADR's
`governs` field. Generated, never hand-edited (treated like the other reserved
generated files — excluded from the bundle's own ADR scans).

```markdown
# Code → ADR map

| code path | governed by |
| --- | --- |
| src/projections/ | 0001, 0003 |
| src/export/ | 0013 |
```

## The guard — four new non-blocking checks

Added to `check.py`, wired into `run_all` / `make adr-check`. Every check **returns
findings; none raises** (the corpus-wide invariant). All are informational.

1. **`check_governs_resolve`** — every path in any ADR's `governs` exists on disk.
   Missing → `0003 governs src/gone/ which does not exist`.
2. **`check_code_markers_resolve`** — every `governed-by: ADR-N` marker names an ADR
   that exists. Dangling → `src/x.py claims ADR-0099 which does not exist`.
3. **`check_governs_agreement`** — the bidirectional cross-check:
   - ADR governs path `P` ⇒ `P` (its module docstring, or the directory's
     `__init__.py` / `README.md`) carries a `governed-by` marker naming that ADR.
     Missing → `0003 governs src/projections/ but nothing there is marked governed-by ADR-0003`.
   - marker `governed-by: ADR-N` at path `Q` ⇒ ADR-N's `governs` includes `Q` **or a
     parent directory of `Q`**. Mismatch → `src/x.py is marked governed-by ADR-0003 but 0003 does not govern it`.
4. **`check_governs_staleness`** — if any governed path git-changed more recently
   than the ADR file → `0003: governed code (src/projections/) changed after the ADR — revisit?`.
   Reuses the existing `git_committer_ts` helper; generalizes today's `source`
   staleness to `governs`.

## Backfill

Add `governs` + the matching `governed-by` markers to the ~8-10 existing ADRs that
map cleanly to code (human-curated, not inferred), e.g.:

| ADR | governs (illustrative — finalized in implementation) |
|---|---|
| 0001 ESDB single source of truth | `src/events/`, `src/persistence/` (event store side) |
| 0003 projection service sole writer | `src/projections/`, `src/services/projection_service.py` |
| 0005 layered mine | `src/ingestion/`, `src/enrichment/`, `src/lens/`, `src/export/` |
| 0009 lens engine generic | `src/lens/` |
| 0011 deterministic+review resolution | `src/resolution/` |
| 0013 read-side OKF exporter | `src/export/` |

Then `make adr-index` (regenerate `by-code.md`) and `make adr-check` → clean.
ADRs with no clean code mapping (e.g. 0007 focused-calls, 0015 the ADR system
itself) omit `governs`.

## Module design (extends `tools/adr/`)

- **`model.py`** — add `governs: list[str]` to `Adr`; parse it in `parse_adr`
  (default `[]`, tolerant like the other list fields).
- **`code_links.py`** (new, generic) — `scan_markers(root, paths=None) -> dict[str, list[int]]`:
  walk the repo, read module docstrings + `__init__.py` docstrings + `README.md`
  files, extract `governed-by` markers, return `{path: [adr_ids]}`. This is the
  reusable "code edge" scanner the later sub-projects (API/CLI/docs) share — keep it
  node-type-agnostic (it finds `governed-by` markers; it does not know what an ADR is).
- **`index.py`** — add `render_by_code(adrs) -> str`; `write_generated` also writes
  `by-code.md`; add `by-code.md` to `RESERVED`.
- **`check.py`** — the four checks above; each pure over `(adrs, markers)` where
  practical so they unit-test without a live tree.
- **`__main__.py` / Makefile** — `adr-index` regenerates `by-code.md`; `adr-check`
  runs the new checks. Optional `adr where <path>` → prints governing ADRs for a
  path (thin, reads `by-code` data).

## Built to extend (C/D/E — not built here)

The `governs` / `governed-by` / agreement / reverse-index machinery is generic
"knowledge-node ↔ code edge." The later graph sub-projects reuse it:

- **API surface** — endpoints/routers as nodes, `governed-by`-style markers, same
  agreement + reverse-index.
- **CLI surface** — `python -m …` / make targets as nodes.
- **Code documentation** — module/package docs as nodes.

`code_links.py` and the agreement/staleness check *shapes* are the shared substrate;
only the marker keyword and node type change. Keeping `scan_markers` node-agnostic
now is the one forward-looking constraint this spec imposes.

## Testing

- **Unit** — `governs` parse (present / absent / list); `scan_markers` over a
  synthetic tree (docstring marker, `__init__.py` marker, `README.md` marker, file
  with none); each of the four checks fires on a crafted fixture (missing path,
  dangling ADR ref, ADR-governs-but-no-marker, marker-but-ADR-doesn't-govern,
  parent-dir match succeeds, stale governed path via injected `ts_fn`); `render_by_code`
  shape. Assert **no check raises**.
- **Smoke** — `make adr-check` on the real backfilled bundle exits 0 and (after
  backfill) is clean; `make adr-index` produces a `by-code.md` listing the governed
  paths.

## Non-goals (v1)

- **Symbol-level governance** — path only. The `governs` format may later accept a
  `path#Symbol` suffix, but v1 neither emits nor parses it.
- **The API / CLI / code-documentation domains** — separate sub-projects that reuse
  this mechanism; not built here.
- **Auto-suggesting `governs` from code** — the mapping is human-curated (like the
  ADR backfill itself).
- **Surfacing governing ADRs in the read-hook when a governed file is edited** — a
  natural follow-up (the `UserPromptSubmit`/edit path could inject `by-code` hits),
  deliberately deferred to keep B focused.
- **Blocking on any finding** — non-blocking throughout, like the rest of the system.
