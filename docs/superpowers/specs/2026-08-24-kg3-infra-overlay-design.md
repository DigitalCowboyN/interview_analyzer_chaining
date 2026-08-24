# KG-3: Infra / deployment overlay — design

**Status:** approved (brainstx 2026-08-24) — ready for plan
**Milestone:** KG-3 in `docs/superpowers/kg-program-roadmap.md`
**Extends:** ADR-0020 (typed-edge graph model). Consistent with ADR-0025 (rebuilt-from-source),
ADR-0026 (derived, not authored — like the code map), ADR-0028 (marker-fallback precedent, `# emits:`).
**Locks:** a new ADR-0029 ("Infra/deployment overlay is derived from compose").

## Problem

The KG-1 agentic eval proved the graph is blind to the **deployment topology**. Two scenarios are
flagged `gap` and score low because no edge answers them:

- `deploy-service-topology` — *"Which backing services must be running for the API to serve reads?"*
  (gold: `code:api`, `code:persistence`, `code:events.store`)
- `deploy-projection-service` — *"What does the projection-service container need in order to run —
  its configuration, dependencies, and the services it talks to?"* (gold:
  `code:projections.projection_service`, `code:config`, `code:events.store`,
  `code:projections.subscription_manager`)

The graph models Python imports and (since KG-2) event/label flow, but never **containers, services,
their dependency DAG, what code they run, or what backing stores that code talks to**. That
information is not missing from the repo — it is fully latent in an authoritative file,
`docker-compose.yml`, whose endpoints (the `command:` entrypoints) are already `CodeUnit` nodes.

This is the same shape KG-2 exploited: derive an overlay from an authoritative source over nodes that
already exist, rather than authoring a parallel description that drifts.

## Source of truth

`docker-compose.yml` is the **only** deployment authority in the repo (verified: no k8s / terraform /
helm / render / fly manifests; the sole GitHub workflow `health.yml` is CI, and its "render" is
`make regen-derived`, not Render.com). The overlay is therefore derived from:

1. `docker-compose.yml` — services, `depends_on`, `command:`, inline `environment:`. Parsed with
   **PyYAML** (6.0.3, already in `requirements.txt`).
2. A small **client-library → service** map, for `talks_to` (see §Edges).
3. `# talks-to:` **markers** in code, as the escape hatch for the `talks_to` ceiling.

Nothing is authored as prose. Service/EnvVar nodes are graph-derived and are **not** OKF corpus docs
(same treatment as `CodeUnit` since ADR-0026).

## Node types (two new)

Registered in `tools/graph/registry.py::NODE_DOMAINS` and sourced by adapters in
`tools/graph/reader.py::_ADAPTERS`.

### `Service` — domain slug `service:`
One per compose service (7 today: `app`, `worker`, `projection-service`, `redis`, `neo4j`,
`neo4j-test`, `eventstore`). Id = the compose service name.

Derived **`kind`** axis (an attribute, not a separate node type — mirrors `code`'s `level`/`category`):

- `kind = "code"` — the service has a `build:` block **and** a `command:` (→ `app`, `worker`,
  `projection-service`). These run our code.
- `kind = "backing"` — image-only external (`image:` present, no `build:`) (→ `redis`, `neo4j`,
  `neo4j-test`, `eventstore`). These are the "what must be up" answers.

Service context (for `walk`/`context` output): `kind`, `image` (backing) or `command` (code),
exposed `ports`, and the count of `depends_on`.

### `EnvVar` — domain slug `env:`
One per **inline** `environment:` entry across services (e.g. `ESDB_CONNECTION_STRING`,
`PROJECTION_LANE_COUNT`, `ENABLE_PROJECTION_SERVICE`, `PYTHONUNBUFFERED`). Id = the variable name.

**Ceiling (deliberate):** only inline `environment:` vars are modeled. `env_file: [.env]` contents
are **never** read or enumerated — secrets must not enter the graph. Documented, and the `Service`
node records that it also loads `.env` as an opaque fact (a boolean, not the contents).

## Edges (four new + inverses)

All **harvest-grain** (compose is tiny — no lazy/symbol machinery needed; these behave like KG-2's
`reads`/`writes`). Registered in `registry.EDGES`; built by functions in a new `tools/infra/reader.py`,
wired into `tools/graph/reader.py::_DERIVED`.

| edge | inverse | from → to | derivation |
| --- | --- | --- | --- |
| `requires` | `required_by` | Service → Service | compose `depends_on:` keys |
| `runs` | `run_by` | Service → CodeUnit | parse `command:` → the `src.*` module it launches |
| `talks_to` | `talked_to_by` | CodeUnit → Service | client-lib import + `# talks-to:` marker |
| `configured_by` | `configures` | Service → EnvVar | compose inline `environment:` |

### `requires` (Service → Service)
Directly from each service's compose `depends_on:` map (the condition — `service_healthy` /
`service_started` — is carried as an edge property, not a separate node). **Naming (decided in spec
self-review):** the service→service edge uses a *distinct* verb `requires`, **not** the existing
`depends_on` (which is `CodeUnit → CodeUnit`). `tools/graph/render.py` keys the generated catalog by
`et.name` (`counts.get(et.name)`, `by_type.get(et.name)`) — two `EdgeType` rows sharing a name would
double-count and double-list in `docs/graph/graph.md`. A distinct verb avoids the collision and reads
naturally ("service `app` requires `neo4j`"). The other three verbs (`runs`, `talks_to`,
`configured_by`) are likewise new — none collide with an existing edge name.

### `runs` (Service → CodeUnit)
Parse the code-service `command:` list to the module it launches:

- `["uvicorn", "src.main:app", ...]` → `code:main`
- `["celery", "-A", "src.celery_app", "worker", ...]` → `code:celery_app`
- `["python", "-m", "src.run_projection_service", ...]` → `code:run_projection_service`

Resolution rule: find the first token of form `src.<dotted>` (optionally with a `:attr` suffix, as in
`src.main:app`), strip `src.` and any `:attr`, and match against the `CodeUnit` id set. A `command:`
that names no resolvable `src.*` module (shell-form string, a `make` target, a bare binary) yields no
`runs` edge and is **flagged** by `check_infra` (a real ceiling, surfaced not hidden).

### `talks_to` (CodeUnit → Service) — the only non-compose edge
Derived, with a marker fallback (the KG-2 `emits` pattern):

- **Derived:** a small map `{client-lib import → backing-service kind}` keyed on the compose images:
  `neo4j` → the `neo4j` service, `esdbclient` → the `eventstore` service, `celery` → the `redis`
  service. A module whose AST imports one of these libs gets a `talks_to` edge to that service.
  Verified endpoints today: `code:utils.neo4j_driver → neo4j`, `code:events.store → eventstore`,
  `code:celery_app → redis` (each imports exactly one client lib — clean, no ambiguity).
- **Marker fallback:** `# talks-to: <service>` on a module adds an explicit edge, for wrapped or
  indirect clients the import heuristic misses. Same shape as `# emits:` / `# calls:` (ADR-0028/0027).

**Bridging to the code-centric gold:** the gold expects `code:persistence` and
`code:projections.subscription_manager`, which talk to their stores *through* `utils.neo4j_driver` /
`events.store`. The walk bridges via existing `depends_on` code edges: `Service(neo4j) ← talks_to —
code:utils.neo4j_driver ← depends_on — code:persistence`. If a first-class link is wanted, a
`# talks-to:` marker on `persistence` / `subscription_manager` is the honest one-line addition; the
plan will add markers only where the import heuristic leaves a gold node unreachable, never
speculatively.

### `configured_by` (Service → EnvVar)
From each service's inline `environment:` list. Enables "what configures projection-service?" →
`PROJECTION_LANE_COUNT`, `ESDB_CONNECTION_STRING`, `ENABLE_PROJECTION_SERVICE`, `PYTHONUNBUFFERED`.

## Where it lives (per-domain pattern, ADR-0016/0023)

New `tools/infra/` domain, following reader → render → check → CLI:

- `tools/infra/reader.py` — `load_services(root)`, `load_env_vars(root)`, and the four derived-edge
  builders (`service_dep_edges`, `runs_edges`, `talks_to_edges`, `configured_by_edges`). Owns the
  compose parse + the client-lib map + the `command:` resolver.
- `tools/infra/render.py` — renders `docs/infra/index.md` (a browsable catalog: services by kind, the
  dependency DAG, each code-service's entrypoint + talks_to, env vars).
- `tools/infra/check.py` — `check_infra(root)`: a code-service `command:` that resolves to no
  `CodeUnit`; a `talks_to` marker naming an unknown service; an `EnvVar`/service referenced by an
  edge that doesn't resolve. Non-blocking (`List[Finding]`, CLI returns 0), guarded in `run_all`.
- `tools/infra/__main__.py` — `python -m tools.infra check|list|render`.
- `make infra-check` (advisory) + `docs/infra/index.md` added to `regen-derived` and the freshness
  gate (like `docs/code`, `docs/graph`).

Graph wiring (the "add a node/edge type" surface, all small):

- `registry.NODE_DOMAINS`: `Service: "service"`, `EnvVar: "env"`.
- `registry.EDGES`: four `EdgeType` rows (derived; `field` = the `_DERIVED` key).
- `reader._ADAPTERS`: `Service: (load_services, "id")`, `EnvVar: (load_env_vars, "name")`.
- `reader._DERIVED`: four field → builder entries.
- `docs/graph/{index,graph}.md` regenerate to include the new node/edge counts.

Node/edge harvesting stays module-grain and eager (harvest-equivalence, ADR-0027, is unaffected —
these are new node types, not symbol-lazy code edges; `test_lazy_walk.py` still holds because infra
edges appear identically in harvest and walk).

## What it enables (the close = the eval re-run)

- *"Which backing services must run for the API to serve reads?"* →
  `code:api ← run_by — Service(app) — requires → {Service(neo4j), Service(eventstore), Service(redis)}`;
  and `Service(neo4j) ← talks_to — code:utils.neo4j_driver`, `Service(eventstore) ← talks_to —
  code:events.store`. Both the service DAG and the code side are reachable.
- *"What does projection-service need?"* → `Service(projection-service)` → `requires`
  {eventstore, neo4j} + `runs` → `code:run_projection_service` + `configured_by` →
  {`PROJECTION_LANE_COUNT`, `ESDB_CONNECTION_STRING`, `ENABLE_PROJECTION_SERVICE`}; and the entrypoint
  code reaches `code:config` / `code:projections.subscription_manager` via existing (CodeUnit→CodeUnit)
  `depends_on`.

## Ceilings (documented, and where possible guarded)

1. **Single compose file.** Overlays (`docker-compose.override.yml`), profiles, and non-compose
   deploy targets are out of scope (none exist today). If a second compose file appears, the reader
   parses only `docker-compose.yml` until extended — stated, not silently partial.
2. **`command:` must name a `src.*` module** for `runs` to resolve. Shell-form commands and Makefile
   indirection need a future marker; today all three code services use exec-form `src.*` commands.
   Unresolved commands are **flagged**.
3. **`talks_to` import heuristic** covers direct client-lib imports; wrapped/indirect clients need
   `# talks-to:`. Same ceiling as `emits`.
4. **EnvVar = inline `environment:` only.** `.env` contents are excluded by design (secrets). A
   service records that it loads `.env` as an opaque boolean.
5. **`requires` reflects compose `depends_on:` declarations, not runtime reality** — a service that
   talks to a store without declaring `depends_on:` would be under-linked; the `talks_to` edge partly
   compensates on the code side.

## Deliverables

1. `tools/infra/` (reader, render, check, `__main__`) + `docs/infra/index.md`.
2. Two node types + four edges wired into the graph; `docs/graph` regenerated.
3. `make infra-check`; `docs/infra/index.md` in `regen-derived` + freshness gate.
4. Tests: compose parse, `kind` derivation, each edge builder, the `command:` resolver, the
   `talks_to` import+marker derivation, `check_infra` (clean on real repo + flags a synthetic
   unresolvable command), a `walk`/`context` end-to-end for both eval questions.
5. Eval re-run of `deploy-service-topology` + `deploy-projection-service`; lift recorded in
   `evals/graph/RESULTS.md`.
6. **ADR-0029**; whole-branch review; PR.

## Knowledge-graph check

This spec changes surfaces the knowledge graph covers (adds node/edge types, a new `tools/infra/`
domain, a generated `docs/infra/` catalog). On implementation: run `make graph-check`, `make
infra-check` (new), and the freshness gate; regenerate `docs/graph/{index,graph}.md`,
`docs/cli/index.md` (new make target), `docs/code/index.md` + `docs/tests/index.md` (new modules/tests).
ADR-0029 records the decision; `make adr-index` + `make knowledge-check`.
