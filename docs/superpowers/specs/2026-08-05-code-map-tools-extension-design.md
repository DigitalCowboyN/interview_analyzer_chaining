# Code-Map Extension to `tools/` — design

**Status:** approved by owner 2026-08-05 (brainstorm dialogue).
**Program:** Round A of a two-round effort. The code map (`docs/code/`, round-1 domain)
covers only `src/` — it is blind to the entire `tools/` operations surface (the
guarded-knowledge-graph program itself). This round extends the map to `tools/` so
that (a) the code map actually covers all the repo's Python, and (b) **Round B** can
link *operations capabilities* to their implementing tool code. Round B (the
`class: product | operations` axis + the operations capability tree + file-management)
builds on this.

## Framing (locked in brainstorm)

The code map claims to answer "what will this change touch?" but silently excludes
~half the repo's Python — all 9 `tools/` packages. Extending it is correct
independent of capabilities, and it makes the operations architecture visible for the
first time: the generated `pipeline.md` will show `tools.capability → tools.code`,
`tools.knowledge → tools.code`, and tool→`src` edges (e.g. every tool that reuses
`src.ingestion.front_matter`).

**Slug convention:** unprefixed unit = product code (`src/`, unchanged); a
`tools.`-prefixed unit = operations/tooling code. So the 9 new nodes are
`tools.adr`, `tools.api`, `tools.capability`, `tools.cli`, `tools.code`,
`tools.glossary`, `tools.graphq`, `tools.knowledge`, `tools.prompts`. **No collision
with bare `src` slugs** — existing product-capability `implemented_by` lists
(`api`, `enrichment`, …) stay valid and untouched.

## Nodes

Nine new `docs/code/tools.<pkg>.md` CodeUnit nodes, **package-level only** (no tool
key-module nodes this round — the tools packages are small and uniform). Authored
frontmatter mirrors the existing code nodes:

```yaml
---
type: CodeUnit
unit: tools.capability
role: tooling
key_modules: [reader, render, check]
---
Catalogues value-framed capabilities linked to the code map; the capability domain's reader/render/check/CLI.
```

`role: tooling` (an existing, previously-unused role in the code map's taxonomy).
`depends_on` and `io` remain **derived**, never authored.

## Reader extension — `tools/code/reader.py`

The reader is `src/`-hardcoded in three spots; extend each to also see `tools/`:

- **`packages(root)`** — additionally enumerate `tools/` subdirectories, emitting each
  as `tools.<name>` (skip `__pycache__`). Returns bare `src` packages + dotted
  `tools.*` packages.
- **`_files_of(unit, root)`** — resolve a `tools.`-prefixed unit to
  `tools/<name>/**/*.py`. (Existing behaviour unchanged: a bare unit → `src/<unit>/…`;
  a dotted *src* unit like `ingestion.orchestrator` → that one `src` file. The
  `tools.` prefix disambiguates a tools *package* from a src *key-module*.)
- **`_IMPORT`** — broaden from `src\.(\w+)` to capture **both** trees, e.g.
  `(?:from|import)\s+(src|tools)\.(\w+)`, mapping a match to the dep slug `pkg`
  (src) or `tools.pkg` (tools). `dep_edges` continues to keep only edges whose target
  is a known unit, so a tool importing `src.ingestion.front_matter` yields an edge to
  `ingestion`, and `tools.capability` importing `tools.code.reader` yields an edge to
  `tools.code`.

`io_of` needs no special-casing — its heuristic already scans `_files_of` (tools
mostly read files → `files`; `tools.api` imports the app). `KEY_MODULES` is unchanged
(src-only; tools contribute package nodes only).

## Generated artifacts

`docs/code/index.md` and `docs/code/pipeline.md` regenerate to include the 9 `tools.*`
nodes under a `tooling` group and their derived edges. The Mermaid pipeline now renders
the operations architecture alongside the product pipeline.

## The code domain's own guard (unchanged mechanics, wider scope)

`make code-check` is non-blocking and already checks coverage / classification /
map-in-sync / stale. With `packages()` now returning the 9 tools packages:

- **coverage** now requires a doc node for each `tools.*` package (the backfill
  provides them).
- **map-in-sync** regenerates including the tools nodes + edges.
- **classification** requires each tools node's `role` (`tooling`, authored).

No guard code changes — the wider unit set flows through the existing checks.

## Backfill

Author the 9 `docs/code/tools.<pkg>.md` nodes: `role: tooling`, `key_modules` = the
package's real modules (`reader`, `render`, `check`, plus domain-specific ones), and a
one-line description of what that tool domain does (mine each `tools/<pkg>/` docstrings
+ the matching `docs/<domain>/` bundle). Then `make code-index`; iterate
`make code-check` to clean.

## Testing

- **Unit** — `packages()` includes `tools.capability` etc.; `_files_of("tools.code")`
  resolves to `tools/code/*.py`; `dep_edges` yields a **tool→tool** edge
  (`tools.capability → tools.code`) and a **tool→src** edge (`tools.capability →
  ingestion`, via `src.ingestion.front_matter`); a bare src unit's edges are unchanged
  (no regression). Assert no reader function raises on the wider tree.
- **Smoke** — `make code-index` writes index + pipeline including the 9 tools nodes;
  `make code-check` clean after backfill; existing `tests/code/` stay green.

## Non-goals (this round)

- **Operations capabilities / the `class: product|operations` axis / making `tooling`
  coverage-mandatory in the capability domain** — Round B.
- **The file-management product capability** — Round B.
- **Tool key-module nodes** (`tools.capability.reader`) — package-level only; a later
  drill-down if warranted.
- **Renaming `src` units to `src.`-prefixed** — src stays bare (no churn, no break to
  product-capability links).
- **A new ADR** — this is a scope extension of the existing code map, not a new
  architectural decision; Round B's classification decision gets the ADR.
- **Blocking** on any finding.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes | the domain being extended; reader + 9 backfill nodes; `code-check` clean | `src/` untouched |
| capabilities | no (this round) | — | operations caps that consume these nodes are Round B |
| cli | no | — | `code-index`/`code-check` targets already exist; no new targets |
| adr | no | — | scope extension, not a new decision (Round B carries the ADR) |
| glossary / api / prompts / graph-queries | no | — | no vocabulary/surface/prompt/query change |

**Verdict:** reconciled — the one touched domain (code) is the subject; no other live
edges. Round B's reconciliation (capabilities, adr) is tracked in its own spec.
