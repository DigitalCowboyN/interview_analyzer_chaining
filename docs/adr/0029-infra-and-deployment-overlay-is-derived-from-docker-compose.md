---
type: ADR
id: 29
title: Infra and deployment overlay is derived from docker-compose
status: accepted
date: 2026-08-24
supersedes: []
superseded_by: []
governs:
  - tools/infra/
tags: [adr, knowledge-management, okf, graph, code, infra, deployment, tooling]
source: docs/superpowers/specs/2026-08-24-kg3-infra-overlay-design.md
---
## Context

The KG-1 agentic eval proved the graph is blind to the **deployment topology**: two `deployment`
scenarios (`deploy-service-topology`, `deploy-projection-service`) scored low because no edge answers
"which backing services must be up for the API" or "what does the projection-service container need."
The graph modeled Python imports and (KG-2) event/label flow, but never containers, services, their
dependency DAG, what code they run, or what backing stores that code talks to. That information is not
missing from the repo — it is fully latent in an authoritative file, `docker-compose.yml` (the only
deployment authority; no k8s/terraform/render target exists), whose `command:` entrypoints are already
`CodeUnit` nodes. This is the KG-2 shape again: derive an overlay from an authoritative source over
nodes that already exist, rather than authoring a parallel description that drifts.

## Decision

Add a **derived** infra/deployment overlay — two node types and four edges parsed from
`docker-compose.yml` (+ a client-lib map and a `# talks-to:` marker fallback). Nothing is authored as
prose; `Service`/`EnvVar` are graph-derived, not OKF corpus docs (like `CodeUnit`, ADR-0026).

**Node types** (`tools/graph/registry.py::NODE_DOMAINS`, sourced by `reader._ADAPTERS`):

- **`Service`** (`service:`) — one per compose service, with a derived **`kind`** axis: `code` (has a
  `build:` + `command:`) vs `backing` (image-only external). An attribute, not a separate type
  (mirrors `code`'s `level`/`category`).
- **`EnvVar`** (`env:`) — one per **inline** `environment:` var. `.env` / `env_file` contents are
  **never** read (secret-safety); a service records that it loads `.env` as an opaque boolean.

**Edges** (all harvest-grain; `registry.EDGES` + `reader._DERIVED`, built in `tools/infra/reader.py`):

- **`requires` / `required_by`** (Service → Service) — from compose `depends_on:`. A *distinct* verb,
  **not** a second `depends_on` (the `CodeUnit→CodeUnit` one): `render.py` keys the catalog by edge
  name, so a duplicate name would double-count/double-list.
- **`runs` / `run_by`** (Service → CodeUnit) — the `command:` parsed to the `src.*` module it launches
  (`uvicorn src.main:app` → `code:main`).
- **`talks_to` / `talked_to_by`** (CodeUnit → Service) — a module's backing-service client-library
  import (`neo4j`/`esdbclient`/`celery` → the neo4j/eventstore/redis service), with a `# talks-to:`
  marker fallback (the ADR-0028 `# emits:` precedent).
- **`configured_by` / `configures`** (Service → EnvVar) — from inline `environment:`.

Result: `service:app —requires→ {neo4j, eventstore, redis}` + `—runs→ code:main`; walking inbound from
a backing service reaches both its `required_by` services and its `talks_to` code. The two deployment
scenarios become traversable end to end.

This **extends ADR-0020** (adds `Service`/`EnvVar` node types + four edges to the typed-edge model),
is **consistent with ADR-0025/0026** (rebuilt-from-source; derived, not authored), and reuses the
**ADR-0028** marker-fallback pattern for the one non-compose edge.

## Consequences

- The KG-1 `deploy-service-topology` / `deploy-projection-service` (`gap`) scenarios become traversable;
  the eval re-run measures the lift (recorded in `evals/graph/RESULTS.md`).
- New `tools/infra/` domain (reader → render → check → CLI), a generated `docs/infra/index.md` catalog,
  and `make infra-check` (advisory) + `make infra-index` (in `regen-derived`, freshness-gated) — the
  per-domain pattern (ADR-0016/0023). This ADR `governs` `tools/infra/`.
- **Fidelity ceilings, documented + guarded** (`tools.infra.check.check_infra`, advisory):
  a single compose file (overlays/profiles/other deploy targets out of scope — none exist today);
  `runs` needs an exec-form `command:` naming `src.*` (a shell-form command or Makefile indirection is
  **flagged**); `talks_to` covers direct client-lib imports (wrapped clients need `# talks-to:`);
  `EnvVar` is inline-only (`.env` excluded); `requires` reflects compose declarations, not runtime.
- The `# talks-to:` marker (and the check) scan **real comment tokens only** (via `tokenize`), so a
  marker-syntax string inside a docstring/description never mints a false edge or finding — a
  correctness fix the check itself surfaced during the build.
- `talks_to`'s import heuristic found the true infra boundary (5 modules, incl. the
  `subscription_manager` gold node) with zero false positives, so **no source markers were needed**.

## Alternatives considered

- **Authored infra domain** (`docs/infra/` prose service nodes, drift-guarded). Rejected: the topology
  is already latent in an authoritative file, so authoring re-introduces drift for derivable facts.
- **A second `depends_on` row for Service→Service.** Rejected in spec self-review: `render.py` keys the
  catalog by edge name; a duplicate would double-count. `requires` is distinct and reads naturally.
- **Model every `.env`/`env_file` variable.** Rejected: secrets must not enter the graph; inline
  `environment:` is the safe, authoritative subset.
- **Infer `talks_to` from connection-string env vars.** Rejected as fuzziest on the code side (all
  config flows through `code:config`); the client-lib import is precise and the marker covers the rest.
