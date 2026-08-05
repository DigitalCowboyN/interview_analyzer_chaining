# CLAUDE.md

Guidance for agents working in this repository.

## What this is
An event-sourced transcript-mining system: EventStoreDB is the source of truth;
a projection service builds a Neo4j read model; a lens engine extracts
insights; an OKF exporter reads the Neo4j side to produce a portable bundle.
FastAPI serves the read side; a Next.js frontend (`frontend/`) renders the
workbench + gallery UI.

## Dev commands
- Lint / format: `make lint` (flake8), `make format` (black)
- Tests: `make test` · unit only: `make test-unit` · integration:
  `make test-integration` (assumes services already running; use
  `make test-integration-full` to also start/stop them)
- Run API / worker: `make run-api`, `make run-worker` · UI: `make ui-dev`
- ADRs: `make adr-check` (validate), `make adr-index` (regenerate index/log)

## Layout
`src/` — application code (ingestion, enrichment, resolution, projections,
lens, export, api, ui). `frontend/` — the Next.js UI.
`tools/adr/` — the ADR knowledge tooling. `docs/adr/` — the decision corpus.
`docs/superpowers/{specs,plans}/` — design specs and implementation plans.

## Knowledge map
This repo keeps guarded knowledge domains under `docs/` — see
[`docs/index.md`](docs/index.md) for the map. Each has a non-blocking
`make <domain>-check`. When you change a surface one covers, consult its bundle and
run its check. When you write a spec/plan, record a `## Knowledge-graph check`
addendum (`make knowledge-check` flags a new one that skipped it).

## Architecture Decision Records (policy)
- **Before locking any architectural decision, consult `docs/adr/index.md`.**
  If your decision changes an existing one, write a new ADR and set
  `supersedes` (and the old ADR's `superseded_by`) — never silently override
  a decision in prose.
- **After a brainstorm locks decisions, capture them:**
  `python -m tools.adr new "<title>"`, fill in the scaffold, set `source:` to
  the spec that locked it, then run `make adr-index`.
- ADRs are durable (what/why); specs are disposable (how, this milestone).
  ADRs link out to specs — don't duplicate spec content into an ADR.
- `make adr-check` reports drift (schema, id uniqueness, bidirectional
  supersede edges, specs that lock decisions without an ADR, and ADRs whose
  `source` changed after them) — it never blocks a build or commit.
