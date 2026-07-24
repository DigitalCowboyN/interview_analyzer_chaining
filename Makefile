# Makefile

# Variables
# Python detection: Allow override, else prefer 'python' (pyenv), fallback to 'python3'
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
MODULE_NAME = src
TEST_DIR = tests

# Default target
.PHONY: all
all: lint format test

# Linting
.PHONY: lint
lint:
	@echo "Linting code..."
	$(PYTHON) -m flake8 $(MODULE_NAME) $(TEST_DIR)

# Formatting
.PHONY: format
format:
	@echo "Formatting code..."
	$(PYTHON) -m black $(MODULE_NAME) $(TEST_DIR)

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST_DIR)

# Testing with coverage
.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest --cov=$(MODULE_NAME) --cov-report=term-missing --cov-report=html

# Coverage report only (terminal)
.PHONY: coverage
coverage:
	@echo "Generating coverage report..."
	$(PYTHON) -m coverage report --show-missing

# Coverage report HTML
.PHONY: coverage-html
coverage-html:
	@echo "Generating HTML coverage report..."
	$(PYTHON) -m coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Coverage report XML (for CI/CD)
.PHONY: coverage-xml
coverage-xml:
	@echo "Generating XML coverage report..."
	$(PYTHON) -m coverage xml

# Quick unit tests (exclude integration tests)
.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest -m "not integration" --cov=$(MODULE_NAME) --cov-report=term-missing

# Integration tests only (assumes services running; auto-detects environment)
.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(PYTHON) -m pytest -m integration

# Clean coverage data
.PHONY: clean-coverage
clean-coverage:
	@echo "Cleaning coverage data..."
	rm -rf htmlcov/
	rm -f .coverage coverage.xml

# Run API (Development)
.PHONY: run-api
run-api:
	@echo "Starting API server (dev mode)..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery Worker (Development)
.PHONY: run-worker
run-worker:
	@echo "Starting Celery worker..."
	celery -A src.celery_app worker --loglevel=info

# --- Frontend (Next.js UI in frontend/) --- #

# Run the frontend dev server
.PHONY: ui-dev
ui-dev:
	@echo "Starting frontend dev server..."
	cd frontend && npm run dev

# Production build of the frontend
.PHONY: ui-build
ui-build:
	@echo "Building frontend for production..."
	cd frontend && npm run build

# Frontend gates: lint + typecheck + vitest
.PHONY: ui-test
ui-test:
	@echo "Running frontend lint, typecheck, and tests..."
	cd frontend && npm run lint && npm run typecheck && npm test

# Regenerate frontend/openapi.json + src/api/schema.d.ts from the backend
# app object (no running server needed) — commit both after backend
# contract changes.
.PHONY: ui-typegen
ui-typegen:
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
ui-smoke:
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	cd frontend && UI_SMOKE=1 npx playwright test

# --- End Frontend --- #

# Clean (optional)
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


# Default target (shows help)
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Testing:"
	@echo "  test                 Run pytest tests (local)"
	@echo "  test-unit            Run unit tests only (no integration markers)"
	@echo "  test-integration     Run integration tests (assumes services running)"
	@echo "  test-integration-full  Start services → run integration tests → stop services"
	@echo "  test-all-full        Start services → run ALL tests with coverage → stop"
	@echo "  test-rebuild         Run projection rebuild test (validates event sourcing)"
	@echo "  test-cov             Run tests with coverage report"
	@echo ""
	@echo "  Options for test-integration-full and test-all-full:"
	@echo "    PYTEST_ARGS=\"-v -x\"   Pass extra pytest arguments"
	@echo "    KEEP_SERVICES=1       Don't stop services after tests"
	@echo ""
	@echo "Test Infrastructure:"
	@echo "  test-infra-up        Start neo4j-test + eventstore with health checks"
	@echo "  test-infra-down      Stop test infrastructure"
	@echo "  wait-neo4j-test      Wait for Neo4j test database to be healthy"
	@echo "  wait-eventstore      Wait for EventStoreDB to be healthy"
	@echo ""
	@echo "Database Services:"
	@echo "  db-up                Start Neo4j + Redis"
	@echo "  db-down              Stop Neo4j + Redis"
	@echo "  db-test-up           Start test Neo4j database"
	@echo "  db-test-down         Stop test Neo4j database"
	@echo "  db-test-clear        Clear test Neo4j database"
	@echo ""
	@echo "Event Sourcing:"
	@echo "  eventstore-up        Start EventStoreDB"
	@echo "  eventstore-down      Stop EventStoreDB"
	@echo "  eventstore-health    Check EventStoreDB health"
	@echo "  es-up                Start EventStoreDB + Projection Service"
	@echo "  es-down              Stop event sourcing system"
	@echo "  es-status            Show event sourcing system status"
	@echo ""
	@echo "Deployed-Path Smoke:"
	@echo "  deployed-smoke       Prove the dockerized projection path end-to-end (real containers)"
	@echo ""
	@echo "Application:"
	@echo "  build                Build Docker images"
	@echo "  run                  Run application (API)"
	@echo "  run-api              Run FastAPI server (local)"
	@echo "  run-worker           Run Celery worker (local)"
	@echo "  ingest FILE=<path>   Ingest + enrich a transcript (Layer 1+2)"
	@echo ""
	@echo "Frontend (Next.js UI, in frontend/):"
	@echo "  ui-dev               Run the frontend dev server"
	@echo "  ui-build             Production build of the frontend"
	@echo "  ui-test              Frontend gates: lint + typecheck + vitest"
	@echo "  ui-typegen           Regenerate OpenAPI types from the backend app object"
	@echo "  ui-smoke             Playwright smoke: seeded interview -> transcript -> text-edit settle"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint                 Run flake8 linter"
	@echo "  format               Run black formatter"
	@echo "  clean                Remove __pycache__ and .pyc files"
	@echo "  clean-coverage       Remove coverage data"

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker compose build app worker

# Run the application container (defaults to running the API)
.PHONY: run
run:
	@echo "Running application container (API)..."
	docker compose up -d app


# --- Test Database Management --- #

.PHONY: db-test-up
db-test-up:
	@echo "Starting TEST Neo4j database service (without waiting)..."
	# Start only the test database, DON'T wait for healthcheck
	docker compose up -d neo4j-test

.PHONY: db-test-down
db-test-down:
	@echo "Stopping and removing TEST Neo4j database service..."
	# Stop and remove the container and its volume
	docker compose down -v neo4j-test

.PHONY: db-test-clear
db-test-clear:
	@echo "Clearing TEST Neo4j database..."
	# Execute cypher command inside the test container to delete all nodes/relationships
	docker compose exec neo4j-test cypher-shell -u neo4j -p testpassword -d neo4j "MATCH (n) DETACH DELETE n;"
	@echo "TEST Neo4j database cleared."

# --- End Test Database Management --- #

# --- EventStoreDB Management --- #

.PHONY: eventstore-up
eventstore-up:
	@echo "Starting EventStoreDB service..."
	docker compose up -d eventstore
	@echo "Waiting for EventStoreDB to be healthy (this may take 60+ seconds)..."
	@sleep 5

.PHONY: eventstore-down
eventstore-down:
	@echo "Stopping EventStoreDB service..."
	docker compose stop eventstore

.PHONY: eventstore-health
eventstore-health:
	@echo "Checking EventStoreDB health..."
	@docker compose exec eventstore curl -f http://localhost:2113/health/live 2>/dev/null || echo "EventStoreDB not healthy yet. Try: make eventstore-logs"

.PHONY: eventstore-logs
eventstore-logs:
	@echo "Tailing EventStoreDB logs..."
	@docker logs -f interview_analyzer_eventstore

.PHONY: eventstore-restart
eventstore-restart: eventstore-down eventstore-up
	@echo "EventStoreDB restarted"

.PHONY: eventstore-clear
eventstore-clear:
	@echo "WARNING: This will delete all EventStoreDB data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down eventstore
	docker volume rm interview_analyzer_chaining_eventstore_data || true
	@echo "EventStoreDB data cleared. Run 'make eventstore-up' to start fresh."

# --- End EventStoreDB Management --- #

# --- Projection Service Management --- #

.PHONY: run-projection
run-projection:
	@echo "Starting projection service (standalone)..."
	$(PYTHON) -m src.run_projection_service

.PHONY: ingest
ingest:
	@echo "Ingesting + enriching $(FILE)..."
	$(PYTHON) -m src.ingestion $(FILE) --enrich

.PHONY: projection-up
projection-up:
	@echo "Starting projection service via docker-compose..."
	docker compose up -d projection-service

.PHONY: projection-down
projection-down:
	@echo "Stopping projection service..."
	docker compose stop projection-service

.PHONY: projection-logs
projection-logs:
	@echo "Tailing projection service logs..."
	@docker logs -f interview_analyzer_projection_service

.PHONY: projection-restart
projection-restart: projection-down projection-up
	@echo "Projection service restarted"

.PHONY: projection-status
projection-status:
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
deployed-smoke:
	@echo "Building + starting neo4j, eventstore, projection-service (dev stack)..."
	docker compose up -d --build neo4j eventstore projection-service
	@echo "Waiting for services..."
	docker compose ps
	DEPLOYED_SMOKE=1 $$HOME/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_deployed_projection_smoke.py -q --no-cov

# --- End Deployed-Path Smoke --- #

# --- Event Sourcing System Management --- #

.PHONY: es-up
es-up: eventstore-up projection-up
	@echo "Event sourcing system (EventStore + Projection Service) started"

.PHONY: es-down
es-down: projection-down eventstore-down
	@echo "Event sourcing system stopped"

.PHONY: es-status
es-status:
	@echo "=== Event Sourcing System Status ==="
	@echo ""
	@echo "EventStoreDB:"
	@docker ps --filter name=interview_analyzer_eventstore --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "  Not running"
	@echo ""
	@echo "Projection Service:"
	@docker ps --filter name=interview_analyzer_projection_service --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "  Not running"
	@echo ""

.PHONY: es-logs
es-logs:
	@echo "=== Tailing Event Sourcing Logs ==="
	@docker compose logs -f eventstore projection-service

# --- End Event Sourcing System Management --- #

# --- Testing with EventStore --- #

.PHONY: test-eventstore
test-eventstore:
	@echo "Running EventStoreDB-dependent tests..."
	$(PYTHON) -m pytest tests/commands/test_command_handlers.py -v

.PHONY: test-e2e
test-e2e:
	@echo "Running end-to-end integration tests..."
	$(PYTHON) -m pytest tests/integration/test_e2e_file_processing.py tests/integration/test_e2e_user_edits.py -v -m eventstore

.PHONY: test-projections
test-projections:
	@echo "Running projection-related tests..."
	$(PYTHON) -m pytest tests/projections/ -v

.PHONY: test-full-system
test-full-system:
	@echo "Running full system test suite..."
	$(PYTHON) -m pytest tests/ -v --ignore=tests/integration/test_projection_replay.py --ignore=tests/integration/test_idempotency.py --ignore=tests/integration/test_performance.py

# Projection rebuild test - validates event sourcing architecture
# Requires: EventStoreDB + Neo4j running, valid OpenAI API key
# Usage: make test-rebuild
#        make test-rebuild KEEP_SERVICES=1  (don't stop services after)
.PHONY: test-rebuild
test-rebuild: test-infra-up
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
wait-neo4j-test:
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
wait-eventstore:
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
test-infra-up:
	@echo "Starting test infrastructure..."
	docker compose up -d neo4j-test eventstore
	@$(MAKE) wait-neo4j-test
	@$(MAKE) wait-eventstore
	@echo "✓ Test infrastructure ready"

# Stop test infrastructure
.PHONY: test-infra-down
test-infra-down:
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
test-integration-full: test-infra-up
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
test-all-full: test-infra-up
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