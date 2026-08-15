# Makefile

# Variables
# Python detection: allow override, else resolve the pyenv-pinned interpreter (see
# .python-version) even when shims aren't on PATH, then fall back to python/python3.
# The generic fallbacks may pick a system python WITHOUT the project deps (pyyaml,
# pytest) — run inside the project's pyenv env, or set PYTHON=/path/to/python.
PYTHON ?= $(shell pyenv which python 2>/dev/null || command -v python 2>/dev/null || command -v python3 2>/dev/null)
MODULE_NAME = src
TEST_DIR = tests

# Default target
.PHONY: all
all: lint format test ## Lint, format, and test

# Linting
.PHONY: lint
lint: ## Run flake8 linter
	@echo "Linting code..."
	$(PYTHON) -m flake8 $(MODULE_NAME) $(TEST_DIR)

# ADR checks (non-blocking): validates docs/adr/ structure + spec-decision cross-refs
.PHONY: adr-check
adr-check: ## Validate docs/adr (non-blocking)
	@$(PYTHON) -m tools.adr check

# Regenerate docs/adr/index.md + log.md from the ADR bundle
.PHONY: adr-index
adr-index: ## Regenerate docs/adr generated files
	@$(PYTHON) -m tools.adr index

# CLI-surface catalog + reconciliation (non-blocking)
.PHONY: cli-index
cli-index: ## Regenerate docs/cli/index.md (the CLI catalog)
	@$(PYTHON) -m tools.cli index

.PHONY: cli-check
cli-check: ## Reconcile docs against the real CLI surface (non-blocking)
	@$(PYTHON) -m tools.cli check

# API-surface catalog + openapi.json freshness (non-blocking; imports the app)
.PHONY: api-index
api-index: ## Regenerate docs/api/index.md (the API catalog)
	@$(PYTHON) -m tools.api index

.PHONY: api-check
api-check: ## Reconcile the API surface + openapi.json freshness (non-blocking)
	@$(PYTHON) -m tools.api check

# Glossary/taxonomy vocabulary catalog + reconciliation (non-blocking)
.PHONY: glossary-index
glossary-index: ## Regenerate docs/glossary/index.md
	@$(PYTHON) -m tools.glossary index

.PHONY: glossary-check
glossary-check: ## Reconcile the glossary against code vocabulary (non-blocking)
	@$(PYTHON) -m tools.glossary check

.PHONY: prompt-index
prompt-index: ## Regenerate docs/prompts/index.md (probabilistic-components catalog)
	@$(PYTHON) -m tools.prompts index

.PHONY: prompt-check
prompt-check: ## Reconcile the prompt registry vs glossary + code consumers (non-blocking)
	@$(PYTHON) -m tools.prompts check

# Graph-query registry: schema-drift + output-contract reconciliation (non-blocking)
.PHONY: graphq-index
graphq-index: ## Regenerate docs/graph-queries/index.md (graph-query registry)
	@$(PYTHON) -m tools.graphq index

.PHONY: graphq-check
graphq-check: ## Reconcile graph queries vs schema + consumers (non-blocking)
	@$(PYTHON) -m tools.graphq check

.PHONY: code-index
code-index: ## Regenerate docs/code/index.md + pipeline.md (code map)
	@$(PYTHON) -m tools.code index

.PHONY: code-check
code-check: ## Reconcile the code map vs the import graph (non-blocking)
	@$(PYTHON) -m tools.code check

# Cross-domain edge graph: typed links between nodes across all domains (non-blocking)
.PHONY: graph-index
graph-index: ## Regenerate docs/graph/index.md + graph.md (cross-domain edge graph)
	@$(PYTHON) -m tools.graph index

.PHONY: graph-check
graph-check: ## Reconcile the edge graph vs its sources (non-blocking)
	@$(PYTHON) -m tools.graph check

.PHONY: health
health: ## Run every domain check + the cross-domain graph check (full sweep)
	@for d in adr cli api glossary prompts graphq code capability knowledge graph usecase testmap; do $(PYTHON) -m tools.$$d check || true; done

.PHONY: regen-derived
regen-derived: ## Regenerate the source-derived indexes (env-independent; the CI freshness gate's set)
	@$(MAKE) code-index capability-index usecase-index testmap-index glossary-index \
	         graphq-index prompt-index adr-index cli-index graph-index

.PHONY: regen-all
regen-all: ## Regenerate every generated index/doc (regen-derived + the app-derived api index)
	@$(MAKE) regen-derived api-index

.PHONY: knowledge-check
knowledge-check: ## Reconcile specs/plans + cascade root vs the knowledge domains (non-blocking)
	@$(PYTHON) -m tools.knowledge check

.PHONY: capability-index
capability-index: ## Regenerate docs/capabilities/index.md (the capability catalogue)
	@$(PYTHON) -m tools.capability index

.PHONY: capability-check
capability-check: ## Reconcile capabilities vs the code map + coverage (non-blocking)
	@$(PYTHON) -m tools.capability check

.PHONY: usecase-index
usecase-index: ## Regenerate docs/use-cases/index.md (the use-case corpus + derived coverage)
	@$(PYTHON) -m tools.usecase index

.PHONY: usecase-check
usecase-check: ## Reconcile use-cases vs forms/categories/criteria + coverage (non-blocking)
	@$(PYTHON) -m tools.usecase check

.PHONY: testmap-index
testmap-index: ## Regenerate docs/tests/index.md (test suite nodes + verification rollup)
	@$(PYTHON) -m tools.testmap index

.PHONY: testmap-check
testmap-check: ## Reconcile tests vs code/intent + verification coverage (non-blocking)
	@$(PYTHON) -m tools.testmap check

# Install the shared project git hooks (non-blocking ADR drift report on commit)
.PHONY: hooks-install
hooks-install: ## Install the shared project git hooks
	@git config core.hooksPath .githooks
	@echo "git hooks installed (core.hooksPath=.githooks)"

# Formatting
.PHONY: format
format: ## Run black formatter
	@echo "Formatting code..."
	$(PYTHON) -m black $(MODULE_NAME) $(TEST_DIR)

# Testing
.PHONY: test
test: ## Run pytest tests (local)
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST_DIR)

# Testing with coverage
.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest --cov=$(MODULE_NAME) --cov-report=term-missing --cov-report=html

# Coverage report only (terminal)
.PHONY: coverage
coverage: ##@ Coverage report (terminal)
	@echo "Generating coverage report..."
	$(PYTHON) -m coverage report --show-missing

# Coverage report HTML
.PHONY: coverage-html
coverage-html: ##@ Coverage report (HTML)
	@echo "Generating HTML coverage report..."
	$(PYTHON) -m coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Coverage report XML (for CI/CD)
.PHONY: coverage-xml
coverage-xml: ##@ Coverage report (XML, for CI)
	@echo "Generating XML coverage report..."
	$(PYTHON) -m coverage xml

# Quick unit tests (exclude integration tests)
.PHONY: test-unit
test-unit: ## Run unit tests only (no integration markers)
	@echo "Running unit tests..."
	$(PYTHON) -m pytest -m "not integration" --cov=$(MODULE_NAME) --cov-report=term-missing

# Integration tests only (assumes services running; auto-detects environment)
.PHONY: test-integration
test-integration: ## Run integration tests (assumes services running)
	@echo "Running integration tests..."
	$(PYTHON) -m pytest -m integration

# Clean coverage data
.PHONY: clean-coverage
clean-coverage: ## Remove coverage data
	@echo "Cleaning coverage data..."
	rm -rf htmlcov/
	rm -f .coverage coverage.xml

# Run API (Development)
.PHONY: run-api
run-api: ## Run FastAPI server (local, dev)
	@echo "Starting API server (dev mode)..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery Worker (Development)
.PHONY: run-worker
run-worker: ## Run Celery worker (local)
	@echo "Starting Celery worker..."
	celery -A src.celery_app worker --loglevel=info

# --- Frontend (Next.js UI in frontend/) --- #

# Run the frontend dev server
.PHONY: ui-dev
ui-dev: ## Run the frontend dev server
	@echo "Starting frontend dev server..."
	cd frontend && npm run dev

# Production build of the frontend
.PHONY: ui-build
ui-build: ## Production build of the frontend
	@echo "Building frontend for production..."
	cd frontend && npm run build

# Frontend gates: lint + typecheck + vitest
.PHONY: ui-test
ui-test: ## Frontend gates: lint + typecheck + vitest
	@echo "Running frontend lint, typecheck, and tests..."
	cd frontend && npm run lint && npm run typecheck && npm test

# Regenerate frontend/openapi.json + src/api/schema.d.ts from the backend
# app object (no running server needed) — commit both after backend
# contract changes.
.PHONY: ui-typegen
ui-typegen: ## Regenerate OpenAPI types from the backend app object
	@echo "Regenerating frontend OpenAPI types..."
	cd frontend && npm run typegen

# UI Playwright smoke (M5.0 Task 9): proves a real ingest is navigable in the
# workbench AND that a UI-driven text edit round-trips through the real
# event-sourced write path (command -> ESDB -> dockerized projection-service
# -> Neo4j -> refetch). Mirrors `deployed-smoke`'s structure: same dev-stack
# containers (the test-infra-up Neo4j/ESDB have no projection consumer), same
# "don't rely on $(PYTHON)" pyenv pin. Playwright itself starts uvicorn + next
# dev via its `webServer` config (frontend/playwright.config.ts); seeding is
# a Python helper the spec shells out to (frontend/e2e/seed_smoke.py) — see
# frontend/e2e/smoke.spec.ts's header for the full required-services list.
# UI_SMOKE=1 gates the spec so a bare `npx playwright test` (or `npm test`,
# which vitest.config.ts excludes e2e/ from entirely) never runs it.
.PHONY: ui-smoke
ui-smoke: ## Playwright smoke: seeded interview to transcript text-edit settle
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	cd frontend && UI_SMOKE=1 npx playwright test

# --- End Frontend --- #

# Clean (optional)
.PHONY: clean
clean: ## Remove __pycache__ and .pyc files
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


# Default target (shows help)
.PHONY: help
help: ## Show the everyday commands
	@$(PYTHON) -m tools.cli help

# Build the Docker image
.PHONY: build
build: ## Build Docker images
	@echo "Building Docker image..."
	docker compose build app worker

# Run the application container (defaults to running the API)
.PHONY: run
run: ## Run application container (API)
	@echo "Running application container (API)..."
	docker compose up -d app


# --- Test Database Management --- #

.PHONY: db-test-up
db-test-up: ##@ Start test Neo4j (no wait)
	@echo "Starting TEST Neo4j database service (without waiting)..."
	# Start only the test database, DON'T wait for healthcheck
	docker compose up -d neo4j-test

.PHONY: db-test-down
db-test-down: ##@ Stop and remove test Neo4j
	@echo "Stopping and removing TEST Neo4j database service..."
	# Stop and remove the container and its volume
	docker compose down -v neo4j-test

.PHONY: db-test-clear
db-test-clear: ##@ Clear the test Neo4j database
	@echo "Clearing TEST Neo4j database..."
	# Execute cypher command inside the test container to delete all nodes/relationships
	docker compose exec neo4j-test cypher-shell -u neo4j -p testpassword -d neo4j "MATCH (n) DETACH DELETE n;"
	@echo "TEST Neo4j database cleared."

# --- End Test Database Management --- #

# --- EventStoreDB Management --- #

.PHONY: eventstore-up
eventstore-up: ##@ Start EventStoreDB
	@echo "Starting EventStoreDB service..."
	docker compose up -d eventstore
	@echo "Waiting for EventStoreDB to be healthy (this may take 60+ seconds)..."
	@sleep 5

.PHONY: eventstore-down
eventstore-down: ##@ Stop EventStoreDB
	@echo "Stopping EventStoreDB service..."
	docker compose stop eventstore

.PHONY: eventstore-health
eventstore-health: ##@ Check EventStoreDB health
	@echo "Checking EventStoreDB health..."
	@docker compose exec eventstore curl -f http://localhost:2113/health/live 2>/dev/null || echo "EventStoreDB not healthy yet. Try: make eventstore-logs"

.PHONY: eventstore-logs
eventstore-logs: ##@ Tail EventStoreDB logs
	@echo "Tailing EventStoreDB logs..."
	@docker logs -f interview_analyzer_eventstore

.PHONY: eventstore-restart
eventstore-restart: eventstore-down eventstore-up ##@ Restart EventStoreDB
	@echo "EventStoreDB restarted"

.PHONY: eventstore-clear
eventstore-clear: ##@ Delete all EventStoreDB data (destructive)
	@echo "WARNING: This will delete all EventStoreDB data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down eventstore
	docker volume rm interview_analyzer_chaining_eventstore_data || true
	@echo "EventStoreDB data cleared. Run 'make eventstore-up' to start fresh."

# --- End EventStoreDB Management --- #

# --- Projection Service Management --- #

.PHONY: run-projection
run-projection: ## Run the projection service (standalone)
	@echo "Starting projection service (standalone)..."
	$(PYTHON) -m src.run_projection_service

.PHONY: ingest
ingest: ## Ingest + enrich a transcript (FILE=<path>)
	@echo "Ingesting + enriching $(FILE)..."
	$(PYTHON) -m src.ingestion $(FILE) --enrich

.PHONY: projection-up
projection-up: ##@ Start the projection service (docker)
	@echo "Starting projection service via docker-compose..."
	docker compose up -d projection-service

.PHONY: projection-down
projection-down: ##@ Stop the projection service
	@echo "Stopping projection service..."
	docker compose stop projection-service

.PHONY: projection-logs
projection-logs: ##@ Tail projection service logs
	@echo "Tailing projection service logs..."
	@docker logs -f interview_analyzer_projection_service

.PHONY: projection-restart
projection-restart: projection-down projection-up ##@ Restart the projection service
	@echo "Projection service restarted"

.PHONY: projection-status
projection-status: ##@ Show projection service status
	@echo "Checking projection service status..."
	@docker ps --filter name=interview_analyzer_projection_service --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# --- End Projection Service Management --- #

# --- Deployed-Path Smoke --- #
# Proves the dockerized projection service delivers events end-to-end against
# the DEV neo4j/eventstore containers (not the neo4j-test used by test-infra-up).
# Invokes pytest directly with the pyenv interpreter rather than
# scripts/test-integration.sh: that script overrides NEO4J_URI to the test
# instance, but this test constructs its own dev-Neo4j driver regardless — the
# direct invocation just keeps the intent (dev stack, not test stack) obvious.
# NOTE: deliberately NOT using $(PYTHON) here. $(PYTHON) resolves via
# `command -v python` in the invoking shell, which on this machine (and in
# non-interactive Bash generally) can resolve to a Homebrew python without
# pytest installed. scripts/test.sh and scripts/test-integration.sh pin
# $$HOME/.pyenv/versions/3.10.7/bin/python directly for the same reason —
# mirror that convention here so `make deployed-smoke` works standalone.
.PHONY: deployed-smoke
deployed-smoke: ## Prove the dockerized projection path end-to-end
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	DEPLOYED_SMOKE=1 $$HOME/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_deployed_projection_smoke.py -q --no-cov

# --- End Deployed-Path Smoke --- #

# --- Projection-Ordering Smoke (M4.9) --- #
# Proves the per-lane commit_position reorder buffer fixed the cross-lane
# ordering race: seeds several interviews through the real command path and
# asserts each projects EVERY fragment with a non-null speaker (the
# completeness the race used to break intermittently). See
# tests/integration/test_projection_ordering_smoke.py's header.
# ESDB_CONNECTION_STRING is overridden for the same reason as deployed-smoke:
# the committed .env points ESDB at the docker-internal "eventstore"
# hostname, unresolvable from this host-run pytest process. Pyenv interpreter
# pinned directly (NOT $(PYTHON)) — same rationale as deployed-smoke above.
.PHONY: projection-smoke
projection-smoke: ## Prove the projection-ordering fix (M4.9)
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	PROJECTION_SMOKE=1 ESDB_CONNECTION_STRING=esdb://localhost:2113?tls=false $$HOME/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_projection_ordering_smoke.py -q --no-cov

# --- End Projection-Ordering Smoke --- #

# --- Live-Feed Smoke (M5.1 Task 6) --- #
# Proves the SSE bridge (src/ui/notifications.py's EsdbWatcher/NotificationHub
# feeding src/api/routers/ui.py's `/ui/streams/events` route) delivers a real
# ESDB event to a live subscriber -- see tests/integration/
# test_live_feed_smoke.py's header for exactly what is and isn't exercised
# (only ESDB directly; Neo4j/projection-service are brought up for a single
# consistent dev-stack recipe across all three smokes, not because this
# test's assertions touch them).
# ESDB_CONNECTION_STRING is overridden here for the same reason as
# frontend/e2e/seed_smoke.py's identical override: the committed .env points
# ESDB at the docker-internal "eventstore" hostname, unresolvable from this
# host-run pytest process. get_event_store_client() reads the env var lazily
# on first use (not at import time), and tests/conftest.py's .env loader only
# sets a key if it ISN'T already present -- so setting it here on the command
# line (already in the environment before pytest/conftest ever runs) wins.
# NOTE: deliberately NOT using $(PYTHON) here -- same rationale as
# deployed-smoke above: $(PYTHON) can resolve to a pytest-less Homebrew
# python in non-interactive shells, so the pyenv interpreter is pinned
# directly, mirroring scripts/test.sh's convention.
.PHONY: live-feed-smoke
live-feed-smoke: ## Prove the SSE live-feed bridge delivers a real ESDB event
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	LIVE_FEED_SMOKE=1 ESDB_CONNECTION_STRING=esdb://localhost:2113?tls=false $$HOME/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_live_feed_smoke.py -q --no-cov

# --- End Live-Feed Smoke --- #

# --- Event Sourcing System Management --- #

.PHONY: es-up
es-up: eventstore-up projection-up ##@ Start EventStore + projection service
	@echo "Event sourcing system (EventStore + Projection Service) started"

.PHONY: es-down
es-down: projection-down eventstore-down ##@ Stop the event sourcing system
	@echo "Event sourcing system stopped"

.PHONY: es-status
es-status: ##@ Show event sourcing system status
	@echo "=== Event Sourcing System Status ==="
	@echo ""
	@echo "EventStoreDB:"
	@docker ps --filter name=interview_analyzer_eventstore --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "  Not running"
	@echo ""
	@echo "Projection Service:"
	@docker ps --filter name=interview_analyzer_projection_service --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "  Not running"
	@echo ""

.PHONY: es-logs
es-logs: ##@ Tail event sourcing logs
	@echo "=== Tailing Event Sourcing Logs ==="
	@docker compose logs -f eventstore projection-service

# --- End Event Sourcing System Management --- #

# --- Testing with EventStore --- #

.PHONY: test-eventstore
test-eventstore: ##@ Run EventStoreDB-dependent tests
	@echo "Running EventStoreDB-dependent tests..."
	$(PYTHON) -m pytest tests/commands/test_command_handlers.py -v

.PHONY: test-e2e
test-e2e: ##@ Run end-to-end integration tests
	@echo "Running end-to-end integration tests..."
	$(PYTHON) -m pytest tests/integration/test_e2e_file_processing.py tests/integration/test_e2e_user_edits.py -v -m eventstore

.PHONY: test-projections
test-projections: ##@ Run projection tests
	@echo "Running projection-related tests..."
	$(PYTHON) -m pytest tests/projections/ -v

.PHONY: test-full-system
test-full-system: ##@ Run the full system test suite
	@echo "Running full system test suite..."
	$(PYTHON) -m pytest tests/ -v --ignore=tests/integration/test_projection_replay.py --ignore=tests/integration/test_idempotency.py --ignore=tests/integration/test_performance.py

# Projection rebuild test - validates event sourcing architecture
# Requires: EventStoreDB + Neo4j running, valid OpenAI API key
# Usage: make test-rebuild
#        make test-rebuild KEEP_SERVICES=1  (don't stop services after)
.PHONY: test-rebuild
test-rebuild: test-infra-up ## Run projection rebuild test (validates event sourcing)
	@echo ""
	@echo "=== Running Projection Rebuild Test ==="
	@echo "This test validates that Neo4j can be rebuilt from events."
	@echo ""
	-$(PYTHON) -m pytest tests/integration/test_projection_rebuild.py -v --no-cov $(PYTEST_ARGS); \
	TEST_EXIT=$$?; \
	echo ""; \
	if [ "$(KEEP_SERVICES)" = "0" ]; then \
		$(MAKE) test-infra-down; \
	else \
		echo "KEEP_SERVICES=1: Test infrastructure left running"; \
	fi; \
	exit $$TEST_EXIT

# --- End Testing --- #

# --- Integration Test Orchestration --- #
# These targets properly orchestrate service startup, health checks, and test execution

# Configuration for health check retries
HEALTH_RETRIES ?= 30
HEALTH_INTERVAL ?= 2

# Wait for Neo4j test database to be healthy
.PHONY: wait-neo4j-test
wait-neo4j-test: ##@ Wait for the Neo4j test DB to be healthy
	@echo "Waiting for Neo4j test database to be healthy..."
	@for i in $$(seq 1 $(HEALTH_RETRIES)); do \
		if docker exec interview_analyzer_neo4j_test cypher-shell -u neo4j -p testpassword "RETURN 1" >/dev/null 2>&1; then \
			echo "✓ Neo4j test database is healthy"; \
			exit 0; \
		fi; \
		echo "  Attempt $$i/$(HEALTH_RETRIES) - waiting $(HEALTH_INTERVAL)s..."; \
		sleep $(HEALTH_INTERVAL); \
	done; \
	echo "✗ Neo4j test database failed health check after $(HEALTH_RETRIES) attempts"; \
	exit 1

# Wait for EventStoreDB to be healthy
.PHONY: wait-eventstore
wait-eventstore: ##@ Wait for EventStoreDB to be healthy
	@echo "Waiting for EventStoreDB to be healthy..."
	@for i in $$(seq 1 $(HEALTH_RETRIES)); do \
		if docker exec interview_analyzer_eventstore curl -sf http://localhost:2113/health/live >/dev/null 2>&1; then \
			echo "✓ EventStoreDB is healthy"; \
			exit 0; \
		fi; \
		echo "  Attempt $$i/$(HEALTH_RETRIES) - waiting $(HEALTH_INTERVAL)s..."; \
		sleep $(HEALTH_INTERVAL); \
	done; \
	echo "✗ EventStoreDB failed health check after $(HEALTH_RETRIES) attempts"; \
	exit 1

# Start test infrastructure (neo4j-test + eventstore)
.PHONY: test-infra-up
test-infra-up: ##@ Start test infra (neo4j-test + eventstore)
	@echo "Starting test infrastructure..."
	docker compose up -d neo4j-test eventstore
	@$(MAKE) wait-neo4j-test
	@$(MAKE) wait-eventstore
	@echo "✓ Test infrastructure ready"

# Stop test infrastructure
.PHONY: test-infra-down
test-infra-down: ##@ Stop test infrastructure
	@echo "Stopping test infrastructure..."
	docker compose stop neo4j-test eventstore
	@echo "✓ Test infrastructure stopped"

# Full integration test run: start services → run tests → report
# Usage: make test-integration-full
#        make test-integration-full PYTEST_ARGS="-v -x"
#        make test-integration-full KEEP_SERVICES=1  (don't stop services after)
PYTEST_ARGS ?= -v
KEEP_SERVICES ?= 0

.PHONY: test-integration-full
test-integration-full: test-infra-up ## Start services, run integration tests, stop
	@echo ""
	@echo "=== Running Integration Tests ==="
	@echo ""
	-$(PYTHON) -m pytest -m "integration or eventstore or neo4j" $(PYTEST_ARGS); \
	TEST_EXIT=$$?; \
	echo ""; \
	if [ "$(KEEP_SERVICES)" = "0" ]; then \
		$(MAKE) test-infra-down; \
	else \
		echo "KEEP_SERVICES=1: Test infrastructure left running"; \
	fi; \
	exit $$TEST_EXIT

# Full test suite with all markers
.PHONY: test-all-full
test-all-full: test-infra-up ## Start services, run ALL tests with coverage, stop
	@echo ""
	@echo "=== Running Full Test Suite ==="
	@echo ""
	-$(PYTHON) -m pytest --cov=$(MODULE_NAME) --cov-report=term-missing $(PYTEST_ARGS); \
	TEST_EXIT=$$?; \
	echo ""; \
	if [ "$(KEEP_SERVICES)" = "0" ]; then \
		$(MAKE) test-infra-down; \
	else \
		echo "KEEP_SERVICES=1: Test infrastructure left running"; \
	fi; \
	exit $$TEST_EXIT

# --- End Integration Test Orchestration --- #