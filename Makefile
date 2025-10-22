# Makefile

# Variables
PYTHON = python3
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

# Integration tests only
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

# Clean (optional)
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run API server (explicit target)
.PHONY: run-api-explicit
run-api-explicit:
	@echo "Running API server..."
	docker compose up -d app

# Run tests
.PHONY: test-explicit
test-explicit:
	@echo "Running tests..."
	docker compose run --rm app pytest $(ARGS)

# Run linter
.PHONY: lint-explicit
lint-explicit:
	@echo "Running linter..."
	docker compose run --rm app flake8 src tests

# Run formatter
.PHONY: format-explicit
format-explicit:
	@echo "Running formatter..."
	docker compose run --rm app black src tests

# Stop and remove containers and potentially volumes
.PHONY: clean-explicit
clean-explicit:
	@echo "Stopping and removing containers..."
	docker compose down -v
	@echo "Cleaning complete."

# Start only database services
.PHONY: db-up
db-up:
	@echo "Starting database services (Neo4j, Redis)..."
	docker compose up -d neo4j redis

# Stop only database services
.PHONY: db-down
db-down:
	@echo "Stopping database services..."
	docker compose stop neo4j redis

# Run the pipeline
.PHONY: run-pipeline
run-pipeline:
	@echo "Running pipeline..."
	docker compose run --rm app python src/main.py --run-pipeline $(ARGS)

# Default target (shows help)
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build the Docker image for the application."
	@echo "  run            Run the application container (API server by default)."
	@echo "  run-pipeline   Run the processing pipeline within a container."
	@echo "  run-api        Run the FastAPI server within a container (same as run)."
	@echo "  test           Run pytest tests within a container."
	@echo "  test-cov       Run tests with coverage reporting."
	@echo "  test-unit      Run unit tests only (exclude integration tests)."
	@echo "  test-integration Run integration tests only."
	@echo "  coverage       Generate terminal coverage report."
	@echo "  coverage-html  Generate HTML coverage report."
	@echo "  coverage-xml   Generate XML coverage report (for CI/CD)."
	@echo "  lint           Run flake8 linter within a container."
	@echo "  format         Run black code formatter within a container."
	@echo "  clean          Stop and remove containers, remove volumes."
	@echo "  db-up          Start database services (Neo4j, Redis) using Docker Compose."
	@echo "  db-down        Stop database services."

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

# Run tests
.PHONY: test-explicit
test-explicit:
	@echo "Running tests..."
	docker compose run --rm app pytest $(ARGS)

# Run linter
.PHONY: lint-explicit
lint-explicit:
	@echo "Running linter..."
	docker compose run --rm app flake8 src tests

# Run formatter
.PHONY: format-explicit
format-explicit:
	@echo "Running formatter..."
	docker compose run --rm app black src tests

# Stop and remove containers and potentially volumes
.PHONY: clean-explicit
clean-explicit:
	@echo "Stopping and removing containers..."
	docker compose down -v
	@echo "Cleaning complete."

# Start only database services
.PHONY: db-up
db-up:
	@echo "Starting database services (Neo4j, Redis)..."
	docker compose up -d neo4j redis

# Stop only database services
.PHONY: db-down
db-down:
	@echo "Stopping database services..."
	docker compose stop neo4j redis

# Run the pipeline
.PHONY: run-pipeline
run-pipeline:
	@echo "Running pipeline..."
	docker compose run --rm app python src/main.py --run-pipeline $(ARGS)

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

# --- End Testing --- # 