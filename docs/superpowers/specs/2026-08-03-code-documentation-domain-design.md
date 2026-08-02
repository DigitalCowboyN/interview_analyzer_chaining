# Code-Documentation Domain (E) — design

**Status:** approved by owner 2026-08-03 (brainstorm dialogue).
**Program:** sub-project "E" of the *guarded knowledge graph over the codebase* — the
"how" layer: the code itself, classified, with I/O, dependencies, and a generated
pipeline map. First of three rounds (E → Capabilities → Use-cases); the vertical
`fulfills: <capability>` / use-case links are **deferred** to those later rounds.

## Framing (locked in brainstorm)

Code is documented **as code, classified by role** (the deterministic counterpart to
prompts' probabilistic classification). The high-value payoff is a **dependency /
pipeline map** that makes "what will this change touch?" answerable, plus a guard that
catches an **undocumented new cross-package dependency** — real architectural drift.

- **Granularity: package + key modules.** The 16 `src/` packages are the primary nodes;
  ~15 load-bearing modules (orchestrators, engines, readers, `projection_service`,
  `main`) are named sub-nodes. Not every `.py` (that's a later drill-down).
- **Derived vs authored:** dependencies and I/O are **derived from imports** (the import
  graph is truth — never authored, never drift-prone). Each node **authors** only its
  *role* classification and a terse description. The generated map carries the derived
  edges.

## Nodes

One `docs/code/<slug>.md` per package (and per key module). Authored frontmatter:

```yaml
---
type: CodeUnit
unit: ingestion                 # package name (or dotted module for a sub-node)
role: pipeline-layer            # pipeline-layer | surface | infrastructure | model | agent | tooling
key_modules: [orchestrator, stitcher, speaker_inference, front_matter]
---
Layer 1: ingests transcripts, maps speakers, stitches utterances; emits ingestion events.
```

`role` and the prose are the human contribution. **`depends_on` and `io` are NOT in the
frontmatter** — they are derived and rendered into the generated artifacts, so they can
never lie.

## Derived facts (from imports)

- **dependencies** — cross-package / cross-module `from src.X` / `import src.X` edges
  (the DAG already extracted: `ingestion → {agents, enrichment, events, models}`, etc.).
- **I/O** — from import signatures: importing `events` → **ESDB**; `persistence`/neo4j →
  **Neo4j**; `agents` (or openai/anthropic) → **LLM**; FastAPI/uvicorn → **HTTP**;
  file ops → **files**. Heuristic; a node may add an `io_note:` to refine an obvious
  miss, but the derived set is authoritative for the guard.

## Generated artifacts

- **`docs/code/index.md`** — the catalog: each unit as `role · io · depends_on`, grouped
  by role. Generated, never hand-edited.
- **`docs/code/pipeline.md`** — a **Mermaid** dependency/pipeline graph
  (`graph LR; ingestion --> events; api --> export; …`) rendered from the real import
  DAG. The "extend/change safely" map; renders on GitHub.

## The guard — `make code-check` (non-blocking, exit 0)

1. **coverage** — every `src/` package has a `docs/code/<pkg>.md` node; missing →
   `code: package src/resolution has no doc node`.
2. **classification present** — a node with no `role` → informational.
3. **map-in-sync** — the committed `docs/code/index.md` + `pipeline.md` match a fresh
   regeneration from the current import graph. A new cross-package import without a
   regen → `code: docs/code/index.md out of sync — run make code-index (new dependency?)`.
   This is the architectural-drift catch.
4. **stale node** — a node whose `unit` is no longer a real package/module → finding.

All non-blocking.

## Module design — new `tools/code/`

Mirrors the established reader → render → check → CLI split.

- `reader.py` — `packages(root)`, `key_modules(root)` (the curated list), `dep_edges(root)`
  (package + key-module import DAG), `io_of(unit, root)` (the import-signature heuristic);
  `@dataclass CodeUnit(unit, role, key_modules, depends_on, io, description, path)`;
  `load_units(root)` (parse the authored node files + attach derived deps/io).
- `render.py` — `render_index(units)`, `render_pipeline(units)` (Mermaid). Pure.
- `check.py` — `check_coverage`, `check_classification`, `check_map_in_sync`,
  `check_stale`, `run_all(root=".")`. Non-blocking.
- `__main__.py` — `python -m tools.code {index|check}` (`index` writes both index.md +
  pipeline.md).
- **Makefile** — `code-index`, `code-check` (self-documented per the `##` convention).

`CodeUnit` / `Finding` local to `tools/code`.

## Backfill

Author one node per package (16) + the ~15 key modules: pick the `role`, list
`key_modules`, write a one/two-line description (mine the module docstrings + the
architecture docs — `data-flow.md`, `system-overview.md` — for accurate prose). Then
`make code-index`; `make code-check` clean.

## Testing

- **Unit** — `dep_edges` over a synthetic `src` tree (a cross-package import → an edge);
  `io_of` (importing `events` → ESDB); `load_units` parses a node + attaches derived
  deps/io; each check on a fixture (uncovered package; node with no role; index out of
  sync; stale unit). `render_pipeline` emits valid Mermaid. Assert **no check raises**.
- **Smoke** — `make code-index` writes index + Mermaid pipeline for the real 16 packages;
  `make code-check` clean after backfill.

## Non-goals (this round)

- **The `fulfills: <capability>` / use-case links** — Capabilities and Use-cases are the
  next two rounds; this round's nodes carry no vertical link yet.
- **Module-level (every `.py`) nodes** — package + curated key modules only.
- **Call-graph / runtime dependency analysis** — static import edges only.
- **Retrofitting capability links into prompts / CLI / API / graph-queries** — later.
- **Blocking** on any finding.
