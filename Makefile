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