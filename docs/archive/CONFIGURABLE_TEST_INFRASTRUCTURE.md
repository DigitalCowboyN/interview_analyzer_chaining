# Configurable Test Infrastructure

**Status:** ✅ Implemented  
**Date:** 2025-10-22  
**Architectural Principle:** Configuration over Hardcoding

---

## Overview

All test infrastructure connection details (hosts, ports, URIs) are now **configurable via environment variables** rather than hardcoded. This follows the 12-Factor App principle of storing configuration in the environment.

---

## Why This Matters

### ❌ **Before (Hardcoded)**

```python
# Tests ONLY work with these exact values:
neo4j_uri = "bolt://neo4j-test:7687"  # Can't change
eventstore = "esdb://eventstore:2113?tls=false"  # Can't change
```

**Problems:**

- Can't test against different infrastructure
- Port conflicts block testing
- Can't use managed services
- CI/CD must match exactly
- Changes to docker-compose.yml break tests

### ✅ **After (Configurable)**

```python
# Tests work with any infrastructure:
neo4j_uri = os.getenv("NEO4J_TEST_URI", default_uri)  # Override anytime
eventstore = os.getenv("EVENTSTORE_HOST", default_host)  # Override anytime
```

**Benefits:**

- Test against any Neo4j/EventStore instance
- CI/CD can use different ports
- Developers can customize per environment
- Managed services work out of the box
- docker-compose changes don't break tests

---

## Configuration Reference

### Neo4j Test Database

#### Method 1: Full URI Override

```bash
export NEO4J_TEST_URI="bolt://my-neo4j-server:17687"
pytest tests/integration/
```

#### Method 2: Host + Port

```bash
export NEO4J_TEST_HOST="my-neo4j-server"
export NEO4J_TEST_PORT="17687"
pytest tests/integration/
```

#### Method 3: Use Defaults

```bash
# Docker environment → bolt://neo4j-test:7687
# Host environment → bolt://localhost:7688 (from docker-compose.yml)
pytest tests/integration/
```

**All Neo4j Variables:**

- `NEO4J_TEST_URI` - Full connection URI (highest priority)
- `NEO4J_TEST_HOST` - Hostname/service name (default: `neo4j-test` in Docker, `localhost` on host)
- `NEO4J_TEST_PORT` - Port number (default: `7687` in Docker, `7688` on host)
- `NEO4J_TEST_USER` - Username (default: `neo4j`)
- `NEO4J_TEST_PASSWORD` - Password (default: `testpassword`)

---

### EventStoreDB

#### Method 1: Full Connection String Override

```bash
export EVENTSTORE_TEST_CONNECTION_STRING="esdb://my-eventstore:12113?tls=false"
pytest tests/integration/
```

#### Method 2: Host + Port

```bash
export EVENTSTORE_HOST="my-eventstore"
export EVENTSTORE_PORT="12113"
pytest tests/integration/
```

#### Method 3: Use Defaults

```bash
# Docker/CI → esdb://eventstore:2113?tls=false
# Host → esdb://localhost:2113?tls=false
pytest tests/integration/
```

**All EventStore Variables:**

- `EVENTSTORE_TEST_CONNECTION_STRING` - Full connection string (highest priority)
- `EVENTSTORE_HOST` - Hostname/service name (default: `eventstore` in Docker/CI, `localhost` on host)
- `EVENTSTORE_PORT` - Port number (default: `2113`)

---

## Common Use Cases

### 1. Default Development (No Config Needed)

```bash
# Start docker-compose services
make db-test-up
make eventstore-up

# Run tests (uses docker-compose ports automatically)
pytest tests/integration/
```

**What happens:**

- Tests detect they're on host
- Connect to `localhost:7688` (Neo4j test)
- Connect to `localhost:2113` (EventStore)
- Matches `docker-compose.yml` port mappings

---

### 2. CI/CD with Custom Ports

```yaml
# .github/workflows/test.yml
env:
  NEO4J_TEST_PORT: 17688 # Avoid conflicts with other jobs
  EVENTSTORE_PORT: 12113

steps:
  - name: Start services
    run: |
      docker run -d -p 17688:7687 neo4j:5.22.0
      docker run -d -p 12113:2113 eventstore/eventstore:23.10.1

  - name: Run tests
    run: pytest tests/integration/
```

**What happens:**

- Tests use overridden ports
- No code changes needed
- Parallel CI jobs don't conflict

---

### 3. Testing Against Managed Services

```bash
# Point to Neo4j Aura instance
export NEO4J_TEST_URI="neo4j+s://xxxx.databases.neo4j.io:7687"
export NEO4J_TEST_USER="neo4j"
export NEO4J_TEST_PASSWORD="<aura-password>"

# Point to EventStore Cloud
export EVENTSTORE_TEST_CONNECTION_STRING="esdb+discover://xxxx.eventstore.cloud:2113?tls=true"

pytest tests/integration/
```

**What happens:**

- Tests run against cloud services
- Validates production-like environment
- No test code changes

---

### 4. Multiple Local Instances

```bash
# Terminal 1: Test against instance A
export NEO4J_TEST_PORT=7688
export EVENTSTORE_PORT=2113
pytest tests/integration/

# Terminal 2: Test against instance B
export NEO4J_TEST_PORT=7689
export EVENTSTORE_PORT=2114
pytest tests/integration/
```

**What happens:**

- Each terminal tests different infrastructure
- No port conflicts
- Parallel development testing

---

## Implementation Details

### Priority Order

Configuration follows this precedence (highest to lowest):

#### Neo4j:

1. `NEO4J_TEST_URI` (explicit full URI)
2. `NEO4J_TEST_HOST` + `NEO4J_TEST_PORT` (construct URI)
3. Environment-aware defaults (Docker vs Host)

#### EventStore:

1. `EVENTSTORE_TEST_CONNECTION_STRING` (explicit full string)
2. `EVENTSTORE_HOST` + `EVENTSTORE_PORT` (construct string)
3. Environment-aware defaults (Docker/CI vs Host)

### Code Examples

#### conftest.py - Neo4j Configuration

```python
neo4j_test_uri = os.getenv("NEO4J_TEST_URI")
if not neo4j_test_uri:
    if environment == "docker":
        neo4j_host = os.getenv("NEO4J_TEST_HOST", "neo4j-test")
        neo4j_port = os.getenv("NEO4J_TEST_PORT", "7687")
    else:
        neo4j_host = os.getenv("NEO4J_TEST_HOST", "localhost")
        neo4j_port = os.getenv("NEO4J_TEST_PORT", "7688")
    neo4j_test_uri = f"bolt://{neo4j_host}:{neo4j_port}"
```

#### conftest.py - EventStore Configuration

```python
connection_string = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")

if not connection_string:
    environment = detect_environment()
    if environment in ("docker", "ci"):
        host = os.getenv("EVENTSTORE_HOST", "eventstore")
        port = os.getenv("EVENTSTORE_PORT", "2113")
    else:
        host = os.getenv("EVENTSTORE_HOST", "localhost")
        port = os.getenv("EVENTSTORE_PORT", "2113")

    connection_string = f"esdb://{host}:{port}?tls=false"
```

#### test_e2e_file_processing.py - EventStore in Tests

```python
esdb_connection = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")
if not esdb_connection:
    environment = detect_environment()
    host = os.getenv("EVENTSTORE_HOST",
                     "eventstore" if environment in ("docker", "ci") else "localhost")
    port = os.getenv("EVENTSTORE_PORT", "2113")
    esdb_connection = f"esdb://{host}:{port}?tls=false"
```

---

## Backwards Compatibility

✅ **100% backwards compatible** - All existing tests work without any environment variables set.

**Defaults match docker-compose.yml:**

```yaml
# docker-compose.yml
neo4j-test:
  ports:
    - "7688:7687" # ← Default NEO4J_TEST_PORT=7688 on host

eventstore:
  ports:
    - "2113:2113" # ← Default EVENTSTORE_PORT=2113
```

---

## Testing the Configuration

### Verify Environment Detection

```bash
pytest tests/integration/ -v -k "test_environment" --log-cli-level=INFO
```

**Expected output:**

```
Test environment configured for docker context
  Neo4j Test URI: bolt://neo4j-test:7687
```

### Verify Override Works

```bash
export NEO4J_TEST_PORT=9999
pytest tests/integration/test_neo4j_connection_reliability.py -v --log-cli-level=INFO
```

**Expected output:**

```
Test environment configured for host context
  Neo4j Test URI: bolt://localhost:9999
```

---

## Migration from Hardcoded

### What Changed

**Before:**

```python
# ❌ Hardcoded
if environment == "docker":
    uri = "bolt://neo4j-test:7687"
else:
    uri = "bolt://localhost:7688"
```

**After:**

```python
# ✅ Configurable
uri = os.getenv("NEO4J_TEST_URI")
if not uri:
    host = os.getenv("NEO4J_TEST_HOST",
                     "neo4j-test" if environment == "docker" else "localhost")
    port = os.getenv("NEO4J_TEST_PORT",
                     "7687" if environment == "docker" else "7688")
    uri = f"bolt://{host}:{port}"
```

### Files Modified

1. **`tests/integration/conftest.py`**

   - `setup_test_environment` fixture (Neo4j config)
   - `event_store_client` fixture (EventStore config)

2. **`tests/integration/test_e2e_file_processing.py`**

   - All E2E tests now use configurable EventStore

3. **`tests/integration/test_performance.py`**
   - Performance tests now use configurable EventStore

---

## Architectural Benefits

### 1. **Separation of Concerns**

- ✅ Infrastructure config separate from test logic
- ✅ Tests focus on behavior, not connection details

### 2. **12-Factor App Compliance**

- ✅ Config in environment, not code
- ✅ Same code runs in all environments

### 3. **Flexibility**

- ✅ Test against any infrastructure
- ✅ CI/CD uses different ports
- ✅ Developers customize per machine

### 4. **Maintainability**

- ✅ Change `docker-compose.yml` without touching tests
- ✅ Single source of truth for defaults
- ✅ Self-documenting via variable names

### 5. **Scalability**

- ✅ Parallel test runs with different ports
- ✅ Multiple environments simultaneously
- ✅ Production-like testing on demand

---

## Best Practices

### 1. **Use Defaults for Development**

```bash
# Don't set any env vars - defaults work
make db-test-up
pytest tests/integration/
```

### 2. **Document Custom Configurations**

```bash
# Create .env.test for your custom setup
# tests/.env.test
NEO4J_TEST_PORT=17688
EVENTSTORE_PORT=12113
```

### 3. **CI/CD: Set Explicitly**

```yaml
# Don't rely on defaults in CI
env:
  NEO4J_TEST_URI: "bolt://neo4j-ci:7687"
  EVENTSTORE_HOST: "eventstore-ci"
```

### 4. **Managed Services: Full URIs**

```bash
# Use complete connection strings for clarity
export NEO4J_TEST_URI="neo4j+s://prod.databases.neo4j.io:7687"
export EVENTSTORE_TEST_CONNECTION_STRING="esdb+discover://prod.eventstore.cloud:2113?tls=true"
```

---

## Troubleshooting

### Issue: Tests Can't Connect

**Check environment detection:**

```bash
python -c "from src.utils.environment import detect_environment; print(detect_environment())"
```

**Check what URIs are being used:**

```bash
pytest tests/integration/ -v -s -k "test_environment" --capture=no
```

**Verify services are actually running:**

```bash
# Neo4j
docker ps | grep neo4j-test
curl -I http://localhost:7474  # HTTP interface

# EventStore
docker ps | grep eventstore
curl -I http://localhost:2113/health/live
```

### Issue: Wrong Port in Docker

**Ensure you're using service names, not localhost:**

```bash
# Inside Docker container
export NEO4J_TEST_HOST=neo4j-test  # ← Service name
export EVENTSTORE_HOST=eventstore  # ← Service name

# NOT:
export NEO4J_TEST_HOST=localhost  # ❌ Won't work in Docker
```

---

## Summary

✅ **All test infrastructure is now configurable**
✅ **Defaults match docker-compose.yml**
✅ **100% backwards compatible**
✅ **Follows 12-Factor App principles**
✅ **Enables flexible testing scenarios**

**No hardcoded values** = Better architecture + More flexibility + Easier maintenance

---

**Related Documentation:**

- [Environment-Aware Testing](./ENVIRONMENT_AWARE_TESTING.md)
- [Categories 2 & 3 Session Summary](./CATEGORIES_2_AND_3_SESSION_SUMMARY.md)
