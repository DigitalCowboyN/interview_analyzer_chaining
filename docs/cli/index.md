# CLI surface

## Make targets

| command | visibility | description |
| --- | --- | --- |
| adr-check | everyday | Validate docs/adr (non-blocking) |
| adr-index | everyday | Regenerate docs/adr generated files |
| all | everyday | Lint, format, and test |
| api-check | everyday | Reconcile the API surface + openapi.json freshness (non-blocking) |
| api-index | everyday | Regenerate docs/api/index.md (the API catalog) |
| build | everyday | Build Docker images |
| capability-check | everyday | Reconcile capabilities vs the code map + coverage (non-blocking) |
| capability-index | everyday | Regenerate docs/capabilities/index.md (the capability catalogue) |
| clean | everyday | Remove __pycache__ and .pyc files |
| clean-coverage | everyday | Remove coverage data |
| cli-check | everyday | Reconcile docs against the real CLI surface (non-blocking) |
| cli-index | everyday | Regenerate docs/cli/index.md (the CLI catalog) |
| code-check | everyday | Reconcile the code map vs the import graph (non-blocking) |
| code-index | everyday | Regenerate docs/code/index.md + pipeline.md (code map) |
| coverage | internal | Coverage report (terminal) |
| coverage-html | internal | Coverage report (HTML) |
| coverage-xml | internal | Coverage report (XML, for CI) |
| db-test-clear | internal | Clear the test Neo4j database |
| db-test-down | internal | Stop and remove test Neo4j |
| db-test-up | internal | Start test Neo4j (no wait) |
| deployed-smoke | everyday | Prove the dockerized projection path end-to-end |
| es-down | internal | Stop the event sourcing system |
| es-logs | internal | Tail event sourcing logs |
| es-status | internal | Show event sourcing system status |
| es-up | internal | Start EventStore + projection service |
| eventstore-clear | internal | Delete all EventStoreDB data (destructive) |
| eventstore-down | internal | Stop EventStoreDB |
| eventstore-health | internal | Check EventStoreDB health |
| eventstore-logs | internal | Tail EventStoreDB logs |
| eventstore-restart | internal | Restart EventStoreDB |
| eventstore-up | internal | Start EventStoreDB |
| format | everyday | Run black formatter |
| glossary-check | everyday | Reconcile the glossary against code vocabulary (non-blocking) |
| glossary-index | everyday | Regenerate docs/glossary/index.md |
| graph-check | everyday | Reconcile the edge graph vs its sources (non-blocking) |
| graph-index | everyday | Regenerate docs/graph/index.md + graph.md (cross-domain edge graph) |
| graphq-check | everyday | Reconcile graph queries vs schema + consumers (non-blocking) |
| graphq-index | everyday | Regenerate docs/graph-queries/index.md (graph-query registry) |
| health | everyday | Run every domain check + the cross-domain graph check (full sweep) |
| help | everyday | Show the everyday commands |
| hooks-install | everyday | Install the shared project git hooks |
| ingest | everyday | Ingest + enrich a transcript (FILE=<path>) |
| knowledge-check | everyday | Reconcile specs/plans + cascade root vs the knowledge domains (non-blocking) |
| lint | everyday | Run flake8 linter |
| live-feed-smoke | everyday | Prove the SSE live-feed bridge delivers a real ESDB event |
| projection-down | internal | Stop the projection service |
| projection-logs | internal | Tail projection service logs |
| projection-restart | internal | Restart the projection service |
| projection-smoke | everyday | Prove the projection-ordering fix (M4.9) |
| projection-status | internal | Show projection service status |
| projection-up | internal | Start the projection service (docker) |
| prompt-check | everyday | Reconcile the prompt registry vs glossary + code consumers (non-blocking) |
| prompt-index | everyday | Regenerate docs/prompts/index.md (probabilistic-components catalog) |
| run | everyday | Run application container (API) |
| run-api | everyday | Run FastAPI server (local, dev) |
| run-projection | everyday | Run the projection service (standalone) |
| run-worker | everyday | Run Celery worker (local) |
| test | everyday | Run pytest tests (local) |
| test-all-full | everyday | Start services, run ALL tests with coverage, stop |
| test-cov | everyday | Run tests with coverage report |
| test-e2e | internal | Run end-to-end integration tests |
| test-eventstore | internal | Run EventStoreDB-dependent tests |
| test-full-system | internal | Run the full system test suite |
| test-infra-down | internal | Stop test infrastructure |
| test-infra-up | internal | Start test infra (neo4j-test + eventstore) |
| test-integration | everyday | Run integration tests (assumes services running) |
| test-integration-full | everyday | Start services, run integration tests, stop |
| test-projections | internal | Run projection tests |
| test-rebuild | everyday | Run projection rebuild test (validates event sourcing) |
| test-unit | everyday | Run unit tests only (no integration markers) |
| ui-build | everyday | Production build of the frontend |
| ui-dev | everyday | Run the frontend dev server |
| ui-smoke | everyday | Playwright smoke: seeded interview to transcript text-edit settle |
| ui-test | everyday | Frontend gates: lint + typecheck + vitest |
| ui-typegen | everyday | Regenerate OpenAPI types from the backend app object |
| usecase-check | everyday | Reconcile use-cases vs forms/categories/criteria + coverage (non-blocking) |
| usecase-index | everyday | Regenerate docs/use-cases/index.md (the use-case corpus + derived coverage) |
| wait-eventstore | internal | Wait for EventStoreDB to be healthy |
| wait-neo4j-test | internal | Wait for the Neo4j test DB to be healthy |

## Module entry points

| command | description |
| --- | --- |
| python -m src.ask | CLI: python -m src.ask <project_id> "<question>" [--top-k 12] |
| python -m src.enrichment | Enrichment layer. |
| python -m src.export | Export layer (OKF). |
| python -m src.ingestion | Ingestion layer. |
| python -m src.lens | Lens engine. |
| python -m src.resolution | Resolution layer. |
| python -m tools.adr |  |
| python -m tools.api |  |
| python -m tools.capability |  |
| python -m tools.cli |  |
| python -m tools.code |  |
| python -m tools.glossary |  |
| python -m tools.graph |  |
| python -m tools.graphq |  |
| python -m tools.knowledge |  |
| python -m tools.prompts |  |
| python -m tools.usecase |  |
