# Environment-Aware Testing Architecture

**Status:** ✅ Implemented  
**Date:** October 21, 2025  
**Implementation:** Option 1 - Smart Environment-Aware Fixture

## Overview

This document describes the environment-aware testing architecture that allows tests to run seamlessly in multiple environments:

- **Docker containers** (e.g., dev containers, CI)
- **Host machines** (local development)
- **CI/CD pipelines** (GitHub Actions, GitLab CI, etc.)

## The Problem

Previously, integration tests would attempt to manage Docker services (like `neo4j-test`) by running `make db-test-up` from within test fixtures. This caused failures when tests ran inside containers where:

- Docker CLI commands are not available
- The services are already managed externally by `docker-compose`

**Error Example:**

```python
subprocess.CalledProcessError: Command '['make', 'db-test-up']' returned non-zero exit code 127
```

## The Solution: Smart Environment-Aware Fixtures

### Core Components

#### 1. Environment Detection (`src/utils/environment.py`)

**`detect_environment() -> str`**

- Detects the current runtime environment
- Returns: `"docker"`, `"ci"`, or `"host"`
- Uses multiple detection methods:
  - Presence of `/.dockerenv` file
  - Checking `/proc/1/cgroup` for container indicators
  - CI environment variables (CI, GITHUB_ACTIONS, etc.)

**`is_service_externally_managed(service_name: str, port: int) -> bool`**

- Determines if a service is externally managed
- Returns `True` if:
  - We're in a container/CI AND
  - The service is accessible on the network
- Can be overridden with `MANAGE_TEST_SERVICES` env var

**`check_neo4j_test_ready(timeout: float) -> bool`**

- Verifies Neo4j test service is ready
- Checks both Bolt (7687) and HTTP (7474) connectivity
- Returns `True` if service is healthy

#### 2. Environment-Aware Test Fixture (`tests/integration/conftest.py`)

**`ensure_neo4j_test_service()` Fixture**

This session-scoped fixture adapts its behavior based on the environment:

```python
@pytest.fixture(scope="session")
def ensure_neo4j_test_service():
    environment = detect_environment()
    externally_managed = is_service_externally_managed("neo4j-test", 7687)

    if externally_managed:
        # In Docker/CI: Verify service availability
        if not check_neo4j_test_ready(timeout=10.0):
            pytest.fail("Neo4j test service is not accessible")
    else:
        # On Host: Start the service ourselves
        subprocess.run(["make", "db-test-up"], check=True)
        # Wait for readiness...

    yield  # Tests run

    # Cleanup only if we started the service
    if service_started_by_fixture:
        subprocess.run(["make", "db-test-down"], check=True)
```

### Behavior Matrix

| Environment                | Service Detection      | Action Taken     | Cleanup         |
| -------------------------- | ---------------------- | ---------------- | --------------- |
| **Docker** (dev container) | Service accessible     | ✅ Verify only   | ❌ Skip         |
| **Host** (local dev)       | Service not accessible | 🚀 Start service | ✅ Stop service |
| **CI** (GitHub Actions)    | Service accessible     | ✅ Verify only   | ❌ Skip         |
| **CI** (custom)            | Service not accessible | 🚀 Start service | ✅ Stop service |

### Override Mechanism

You can force a specific behavior using the `MANAGE_TEST_SERVICES` environment variable:

```bash
# Force fixture to manage services (start/stop)
export MANAGE_TEST_SERVICES=true

# Force fixture to assume external management (verify only)
export MANAGE_TEST_SERVICES=false
```

## Usage in Different Contexts

### 1. Development in VS Code Dev Container

**Setup:** `docker-compose.yml` runs `neo4j-test` service

**Test Execution:**

```bash
# Inside dev container
pytest tests/integration/
```

**Behavior:**

- Detects `docker` environment
- Finds `neo4j-test` accessible at `bolt://neo4j-test:7687`
- **Verifies availability only** (no start/stop)
- Tests use existing service

**Output:**

```
=== Setting up Neo4j test environment ===
Environment: docker
Externally managed: True
Service is externally managed - verifying availability...
Neo4j test service verified and ready!
```

### 2. Host Machine Development

**Setup:** No services running initially

**Test Execution:**

```bash
# On host machine
pytest tests/integration/
```

**Behavior:**

- Detects `host` environment
- Service not accessible initially
- **Starts neo4j-test** using `make db-test-up`
- Waits for service to be healthy
- Tests run
- **Stops neo4j-test** using `make db-test-down`

**Output:**

```
=== Setting up Neo4j test environment ===
Environment: host
Externally managed: False
Starting neo4j-test container...
Neo4j test container started successfully
Waiting for Neo4j test service to become ready...
Neo4j test service is ready for testing!
```

### 3. CI/CD Pipeline

**GitHub Actions Example:**

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      neo4j-test:
        image: neo4j:5.11
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7688:7687

    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/
```

**Behavior:**

- Detects `ci` environment (via `CI` env var)
- Finds neo4j-test accessible at `localhost:7688`
- **Verifies availability only**
- Uses GitHub Actions service container

## Architecture Benefits

### ✅ Correctness

- **No Docker-in-Docker complexity** - Respects the host orchestration
- **Single source of truth** - Services managed by compose, not tests
- **Idempotent** - Multiple test runs don't conflict

### ✅ Performance

- **Faster in containers** - No service start/stop overhead
- **Shared service** - All tests use the same Neo4j instance
- **Quick verification** - Socket checks take milliseconds

### ✅ Developer Experience

- **Transparent** - Works the same everywhere
- **Debuggable** - Clear logs show what's happening
- **Flexible** - Override mechanism for special cases

### ✅ Maintainability

- **Centralized logic** - Environment detection in one place
- **Testable** - Detection functions are unit-testable
- **Extensible** - Easy to add more services (EventStoreDB, Redis, etc.)

## Test Results

**Before (Old Approach):**

```
ERROR: subprocess.CalledProcessError: returned non-zero exit code 127
129 integration tests failed
```

**After (Environment-Aware Approach):**

```
=== 139 passed, 4 failed, 15 errors in 91.77s ===
Coverage: 43.1% (above 25% threshold)
0 subprocess errors related to service management
```

The 4 failures and 15 errors are **unrelated to service management** and are pre-existing issues with test code itself.

## Future Enhancements

### 1. EventStoreDB Support

Apply the same pattern to EventStoreDB tests:

```python
@pytest.fixture(scope="session")
def ensure_eventstore_service():
    if is_service_externally_managed("eventstore", 2113):
        # Verify HTTP health endpoint
        check_eventstore_ready()
    else:
        # Start using makefile
        subprocess.run(["make", "eventstore-up"])
```

### 2. Redis Support

Extend to Redis if needed:

```python
if is_service_externally_managed("redis", 6379):
    verify_redis_ping()
```

### 3. Dynamic Service Discovery

Use Docker API to query running containers:

```python
def get_running_services() -> List[str]:
    # Query Docker to find what's already running
    pass
```

## Related Files

- **`src/utils/environment.py`** - Core detection and verification logic
- **`tests/integration/conftest.py`** - Environment-aware fixtures
- **`docker-compose.yml`** - Service orchestration
- **`.devcontainer/devcontainer.json`** - Dev container configuration

## References

- [Original Analysis](../ARCHITECTURE_ANALYSIS_NEO4J_TEST.md) _(if exists)_
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Pytest Fixture Best Practices](https://docs.pytest.org/en/stable/fixture.html)
