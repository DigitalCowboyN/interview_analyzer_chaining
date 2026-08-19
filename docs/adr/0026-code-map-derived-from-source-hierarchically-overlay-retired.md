---
type: ADR
id: 26
title: Code map derived from source, hierarchically; overlay retired
status: accepted
date: 2026-08-16
supersedes: []
superseded_by: []
governs:
  - tools/code/
tags: [adr, knowledge-management, okf, graph, code, tooling]
source: docs/superpowers/specs/2026-08-16-hierarchical-code-intake-design.md
---
## Context

The graph's `code` domain (ADR-0020) represented the codebase as 48 `CodeUnit` nodes, every one a
hand-authored `docs/code/*.md` overlay: `load_units` globbed `docs/code/`, not `src/`. The actual
repo is ~194 `.py` files across ~30 packages, and the corpus intake (ADR-0024) ingested **zero**
code — only markdown. So the subject of the whole system was represented by coarse, hand-written
shadows, disconnected from the source they described.

This also violated **ADR-0019** ("capabilities are durable authored intent; implementation is a
*derived, replaceable* link"): the implementation link was still authored by hand, in the overlay's
`unit`/`role`/`depends_on`/`io` fields, rather than derived from the code.

## Decision

The `code` domain is **derived from source, hierarchically**, and the authored overlay is **retired**.

- **One `CodeUnit` type with an open `level` axis** (`package | module`; `symbol` reserved), mirroring
  how Capability uses `kind`. A package is a directory under `src/`/`tools/` that directly contains
  a `.py`; a module is a non-`__init__` `.py`. Ids are dotted paths (`src/` stripped, `tools.` kept):
  `api`, `api.routers`, `api.routers.segments`. ~197 nodes replace the 48 overlays.
- **`contains` / `contained_by`** (new derived edge) express the hierarchy — a package contains its
  sub-packages and modules. This is what lets `walk` go **up** from any module to its package and on
  to the ADR that `governs` it and the Capability that `implements` it: the safeguard against an
  isolated enhancement drifting outside an architectural decision.
- **`depends_on` is derived at module granularity** from AST-style imports resolved to the longest
  existing node-id prefix; a module's own ancestors are excluded (that is the `contains` relation,
  not a dependency).
- **`category` and `determinism` are derived, not authored** — computed post-harvest over the
  assembled cross-domain edges. `category` comes from the implementing capability (children inherit
  their parent primary's category); `determinism` is probabilistic when a unit consumes a `Prompt`
  or depends on the `agents` package, else deterministic.
- **Context comes from docstrings** (`ast.get_docstring`), not authored descriptions. Intent stays
  on the capability/use-case layer (ADR-0019, restored).
- The `docs/code/*.md` unit files are **deleted**; `index.md` + `pipeline.md` remain as generated
  catalogs rendered from the derived nodes. Code is no longer an OKF *document* type — it is dropped
  from the corpus `OKF_HOMES` (ADR-0024), since it is source-derived, not authored markdown.

This **refines ADR-0019** (implementation is now genuinely derived, not a hand-authored link) and
**ADR-0024** (the deferred code-side intake, realized and made hierarchical), and **extends
ADR-0020** (adds the `contains` edge, the `level` axis, and the derived `category`/`determinism`
node axes to the typed-edge model). No dedicated ADR ever established the overlay, so nothing is
superseded; its retirement is a consequence recorded here.

## Consequences

- The graph explains the real code: ~197 nodes with hierarchy and module-level dependencies, every
  pre-existing `implements`/`governs`/`verifies`/`consumed_by` edge still resolving (verified — no
  dangling). Walk-up from any point yields its governing architecture.
- Completeness signals shift to the derived model: `code-check` drops the overlay-reconciliation
  checks (`check_coverage`/`check_map_in_sync`-vs-overlay/`check_stale`/`check_top_level_modules`)
  and gains `check_missing_docstring` (a module with no docstring has no derivable context). The
  capability domain's coverage check is preserved but re-expressed on `level` + an `_INFRA_PACKAGES`
  denylist, since the authored `role` it filtered on is retired.
- Retiring 48 authored descriptions discards hand-written prose; most code is reached by a capability
  so its "why" survives on the intent layer, and the L2 reachability check flags the rest.
- Deferred (not this decision): **symbols** (function/class grain); using walk-up for **governance**
  (L3); richer determinism detection inside the `agents` package (relative/in-function imports the
  import parser does not yet capture).

## Alternatives considered

- **Keep the overlay, add derivation alongside.** Rejected: two sources of truth for the same facts,
  and it would leave ADR-0019 violated (authored implementation link). Derivation is the point.
- **A separate `Package` node type instead of a `level` axis.** Rejected (YAGNI): a new type means a
  registry row, an adapter, and duplicated edge plumbing; one type with an axis mirrors the existing
  Capability `kind` / UseCase `form` pattern and keeps the registry change minimal.
- **Author `category`/`determinism` per unit.** Rejected: both are derivable from edges the graph
  already carries, and authoring them re-introduces the drift the overlay caused.
