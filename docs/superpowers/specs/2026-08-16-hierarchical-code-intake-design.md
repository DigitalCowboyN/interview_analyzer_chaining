# Hierarchical code intake — derive the real code, retire the overlay (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-16).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). This is the
deferred, and by far the largest, half of L0's intake — the **code** — done properly. It changes
the code domain from a hand-authored overlay into a structure derived from source.

## The problem

The graph says "code" is **48 `CodeUnit` nodes**, and every one is a hand-authored
`docs/code/*.md` overlay (`load_units` globs `docs/code/`, not `src/`). The actual repo is **194
`.py` files, ~1,160 functions/classes**. The corpus (`okf_records`) ingests **zero** code — only
markdown. So the subject of the whole system is represented by 48 hand-written shadows, coarse and
disconnected from the source. This also violates **ADR-0019** ("capabilities are durable authored
intent; implementation is a *derived*, replaceable link") — the implementation link was still
hand-authored.

## North star

Derive the real code into the graph, **hierarchically, from source**, and **retire the overlay**.
Code structure is a fact read from `src/`/`tools/`; intent stays on the capability/use-case layer
(ADR-0019, restored). From any code node you can `walk` **up** to the architecture that governs it
— the safeguard against isolated enhancements drifting out of an architectural decision.

## Node model (option A — one type, a `level` axis)

One `CodeUnit` node type gains an open **`level` axis** — `package | module` (with `symbol`
reserved for a later phase) — mirroring how Capability uses `kind` and UseCase uses `form`. No new
node type; minimal registry change.

- **Package** — a directory under `src/`/`tools/` containing Python (has `__init__.py` or `.py`).
  Nested packages are packages too (`api`, `api.routers`). `__init__.py` content belongs to the
  package node.
- **Module** — a non-`__init__` `.py` file. Id = its dotted path.
- **Addressing** (unchanged scheme, extended): `code:api` (package), `code:api.routers`
  (sub-package), `code:api.routers.segments` (module); `tools/` keeps its `tools.` prefix
  (`code:tools.graph`, `code:tools.graph.traverse`). Existing dotted ids (`code:ask.engine`,
  `code:export.reader`) become **module** nodes — so current edges still resolve.
- **Context** — derived from the module/package **docstring** (`ast.get_docstring`), not an
  authored description. 116/120 modules already carry one.

## Discovery (from source, not `docs/code/`)

Walk `src/` and `tools/`: every package dir → a package node; every non-`__init__` `.py` → a
module node; ids are dotted paths (strip the `src/` prefix; keep `tools.`). This replaces the
`docs/code/*.md` glob entirely. ~48 packages + ~194 modules ≈ **240 code nodes**, up from 48.

## Edges

- **`contains` / `contained_by`** (new, hierarchy) — package → its sub-packages and modules.
  This is what makes `walk` go **down** (a package's contents) and **up** (module → package →
  the ADR that `governs` it, the Capability that `implements` it). Walk-up from any point yields
  its architectural context.
- **`depends_on`** (existing edge, now derived at **module** granularity) — from AST imports
  resolved to the **full dotted module path** (today's `_IMPORT` regex stops at the top package;
  this phase resolves `from src.api.routers.segments import X` → `api.routers.segments`). Two
  modules under the same package that import each other are **siblings** related laterally — the
  sibling relationships requested. Package-level `depends_on` is rolled up from module deps.
- **Existing inbound edges are preserved** — `implements` (Capability→CodeUnit), `governs`
  (ADR→CodeUnit), `verifies` (Test→CodeUnit), `consumed_by` (Prompt/GraphQuery→CodeUnit) still
  target code nodes. Package-level targets (`api`, `enrichment`) remain nodes; dotted targets
  become module nodes. **Additive — no existing edge dangles.**

## Derived classification axes (confirmed derivable — no authoring)

Two axes are computed from the graph's edges, not authored (verified against real data):

- **`category`** (`product | operations | supporting`) — from the **implementing capabilities'
  category**: walk inbound `implements`, read each Capability's `category`. Confirmed: `api`,
  `enrichment`, `ask`, `projections`, `export` all derive to `product`. Code reached by no
  capability has no derived category — which is the L2 reachability signal, not a gap to author.
- **`determinism`** (`deterministic | probabilistic`) — **probabilistic = consumes a `Prompt`
  (inbound `consumed_by`) or `depends_on` `agents`**; else deterministic. Confirmed: `enrichment`,
  `ask`, `agents` → probabilistic; `projections`, `events` → deterministic.

These depend on **cross-domain** edges, so they are derived at the graph level (a post-harvest
annotation over the assembled edges), surfaced in the code catalog and available to `walk`.

## Retire the overlay

`docs/code/*.md` authored files are **deleted**. Their content maps out cleanly: `unit` = the
path; `key_modules` = redundant (modules are nodes); `depends_on`/`io` = already derived;
`category`/`determinism` = now derived; `description` = the module docstring (derived) and, for
"why", the Capability that implements it (reachable by walk-up). `docs/code/index.md` +
`pipeline.md` **stay as generated catalogs**, now rendered from the *derived* nodes.

Consequences for `code-check`: `check_coverage` (a src package must have an overlay),
`check_map_in_sync`, `check_stale`, `check_top_level_modules` are **obsolete or repurposed** — with
derivation there is no overlay to reconcile against; completeness is served by the derived node set
itself plus L2 reachability. `load_units` is rewritten to derive from source; the graph's
`CodeUnit` adapter uses it unchanged in shape.

## Reconciliation & risk

- Existing edges to `code:<pkg>` / `code:<dotted-module>` still resolve (package + module nodes
  cover both). The 48 old ids are a subset of the ~240 derived ids. **Verify no dangling** after
  the switch (`graph-check`).
- Retiring 48 authored descriptions discards hand-written prose. Most (46/48) code is reached by a
  capability, so its "why" survives on the intent layer. The 2 unreached units
  (`agents.agent_factory`, `io`) lose their only authored intent — the reachability check already
  flags them as "author a capability," which is where intent belongs.
- Big node-count jump (48 → ~240) enlarges the graph render; the per-edge-type Mermaid split
  (ADR-0020) keeps it readable; `contains` gets its own section.

## Scope

**This phase:** packages + modules; `contains`/`contained_by`; module-level `depends_on`; derived
`category`/`determinism`; docstring context; retire the overlay. **Deferred:** **symbols**
(functions/classes — ~1,160, a separate investigation); using walk-up for **governance** (that is
L3 — this phase builds the structure it needs); migrating any residual overlay prose to
capabilities; the other doc-reader projections (Capability/UseCase/ADR/Term over `okf_records` — a
smaller, separate cleanup).

## ADR

Capture a new ADR: **the code map is derived from source, hierarchically; the authored overlay is
retired.** It **refines ADR-0019** (implementation is now genuinely derived, not a hand-authored
link) and **ADR-0024** (the deferred code-side intake, realized and made hierarchical), and
**extends ADR-0020** (adds the `contains` edge, the `level` axis, and derived `category`/
`determinism` node axes). No dedicated ADR ever established the overlay, so nothing is superseded;
its retirement is a consequence recorded here. `source:` = this spec.

## Testing

- **Discovery:** deriving from a fixture `src/` tree yields the right package + module nodes with
  correct dotted ids and `level`; `__init__.py` maps to its package, not a module.
- **contains:** `api` contains `api.routers`; `api.routers` contains `api.routers.segments`;
  `contained_by` is the inverse — a `walk(module, "in")` reaches its package chain.
- **depends_on (module):** `from src.a.b import x` in module `c.d` yields `c.d --depends_on--> a.b`
  (full dotted resolution, not just `a`); sibling case within a package resolves.
- **Derived axes:** on the real graph, `enrichment`/`ask` derive `probabilistic`; `projections`
  deterministic; `api` category `product`; a unit with no implementing capability has no category.
- **No dangling:** after switching `load_units` to derivation, `graph-check` is clean (every
  pre-existing `implements`/`governs`/`verifies`/`consumed_by` endpoint still resolves).
- **Overlay gone:** `docs/code/*.md` unit files removed; `docs/code/index.md` regenerates from
  derived nodes; the freshness gate is clean.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | note |
| --- | --- | --- |
| code | yes — `load_units` derives packages+modules from source; overlay retired; `contains`/module-`depends_on`; docstring context | the subject |
| graph | yes — `contains` edge + `level`/`category`/`determinism` axes in registry; derived-axis post-pass; render splits `contains` | co-subject |
| capabilities | yes (read-only) | source of the derived `category`; `implements` edges unchanged |
| corpus | yes | `CodeUnit` document type is retired from `okf_records`/`OKF_HOMES` (code is now derived, not a doc); misfiled/unregistered unaffected |
| adr | yes | new ADR (refines 0019/0024, extends 0020) |
| tests / use-cases / glossary / prompts / graph-queries | no (logic) | edges into code still resolve at package+module grain |

**Verdict:** reconciled — code + graph are the subjects (code derived hierarchically from source,
overlay retired, classifications derived); a new ADR captures it; corpus drops `CodeUnit` as a
document type since code is now source-derived, not an authored markdown record.
