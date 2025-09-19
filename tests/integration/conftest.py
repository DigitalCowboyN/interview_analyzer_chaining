"""
Configuration and fixtures for integration tests with environment-aware Neo4j management.
"""

import asyncio
import os
import subprocess
import time
from typing import AsyncGenerator

import pytest

from src.utils.environment import detect_environment, get_available_neo4j_config
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.fixture(scope="session")
def ensure_neo4j_test_service():
    """
    Session-scoped fixture to ensure Neo4j test service is available.

    This fixture handles the Docker service lifecycle and waits for readiness.
    """
    environment = detect_environment()
    print(f"\n=== Setting up Neo4j test environment (detected: {environment}) ===")

    # Start the Neo4j test service
    print("Starting neo4j-test container...")
    try:
        result = subprocess.run(["make", "db-test-up"], check=True, capture_output=True, timeout=120, text=True)
        print("Neo4j test container started successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Neo4j test container: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print("Timeout starting Neo4j test container")
        raise

    # Wait for Neo4j to become ready using our new connection manager
    print("Waiting for Neo4j test service to become ready...")
    ready = asyncio.run(Neo4jConnectionManager.wait_for_ready(timeout=60.0, test_mode=True))

    if not ready:
        # Attempt to get logs for debugging
        try:
            logs_result = subprocess.run(
                ["docker", "logs", "interview_analyzer_neo4j_test"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print("=== NEO4J TEST CONTAINER LOGS ===")
            print(logs_result.stdout)
            print(logs_result.stderr)
            print("================================")
        except Exception as log_e:
            print(f"Could not capture logs: {log_e}")

        pytest.fail("Neo4j test service did not become ready within timeout")

    print("Neo4j test service is ready for testing!")

    yield  # Tests run here

    # Cleanup
    print("\n=== Cleaning up Neo4j test environment ===")
    try:
        subprocess.run(["make", "db-test-down"], check=True, capture_output=True, timeout=30)
        print("Neo4j test container stopped successfully")
    except Exception as e:
        print(f"Warning: Failed to clean up Neo4j test container: {e}")


@pytest.fixture(scope="function")
async def clean_test_database(ensure_neo4j_test_service) -> AsyncGenerator[None, None]:
    """
    Function-scoped fixture that provides a clean Neo4j test database for each test.

    This fixture:
    1. Ensures we have a connection to the test database
    2. Clears all data before the test
    3. Yields control to the test
    4. Optionally cleans up after (currently just logs)
    """
    print("\n--- Setting up clean test database ---")

    try:
        # Get a fresh driver instance for the test database
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        # Clear the database
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

            # Verify the database is clean
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            node_count = record["count"] if record else -1

            if node_count == 0:
                print("Test database cleared successfully")
            else:
                pytest.fail(f"Failed to clear test database. Node count: {node_count}")

    except Exception as e:
        print(f"ERROR: Failed to set up clean test database: {e}")
        pytest.fail(f"Test database setup failed: {e}")

    # Test runs here
    yield

    # Optional: Log final state for debugging
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as final_count")
            record = await result.single()
            final_count = record["final_count"] if record else -1
            print(f"Test completed. Final node count: {final_count}")
    except Exception as e:
        print(f"Warning: Could not check final database state: {e}")


@pytest.fixture(scope="function")
def test_connection_config():
    """
    Provides the current test database connection configuration.

    This is useful for tests that need to know the connection details
    or create their own connections.
    """
    config = get_available_neo4j_config(test_mode=True)
    if not config:
        pytest.skip("No available Neo4j test configuration found")
    return config


@pytest.fixture(scope="function")
async def test_neo4j_session(clean_test_database):
    """
    Provides a ready-to-use Neo4j session for tests.

    This session is connected to the clean test database and
    automatically closed after the test.
    """
    driver = await Neo4jConnectionManager.get_driver(test_mode=True)
    async with driver.session() as session:
        yield session


@pytest.fixture(scope="function", autouse=True)
def setup_test_environment(monkeypatch):
    """
    Auto-use fixture that sets up the test environment variables.

    This ensures that our environment-aware connection logic
    can find the test database configuration.
    """
    environment = detect_environment()

    # Set test-specific environment variables based on detected environment
    if environment == "docker":
        # Inside Docker container - use service names
        monkeypatch.setenv("NEO4J_TEST_URI", "bolt://neo4j-test:7687")
    else:
        # Host or CI - use localhost with exposed port
        monkeypatch.setenv("NEO4J_TEST_URI", "bolt://localhost:7688")

    monkeypatch.setenv("NEO4J_TEST_USER", "neo4j")
    monkeypatch.setenv("NEO4J_TEST_PASSWORD", "testpassword")

    print(f"Test environment configured for {environment} context")

    # Force connection manager to reinitialize with new environment
    # We'll let the connection manager handle this naturally rather than forcing it
    Neo4jConnectionManager._driver = None

    yield  # Test runs here

    # Cleanup: Reset driver state for next test
    try:
        Neo4jConnectionManager._driver = None
    except Exception as e:
        print(f"Warning: Error resetting driver after test: {e}")


# Utility fixtures for specific test scenarios


@pytest.fixture
def skip_if_no_neo4j():
    """
    Fixture that skips the test if Neo4j is not available.

    Useful for optional integration tests that should be skipped
    in environments where Neo4j is not set up.
    """
    config = get_available_neo4j_config(test_mode=True)
    if not config:
        pytest.skip("Neo4j test database not available")


@pytest.fixture
async def neo4j_health_check():
    """
    Fixture that performs a health check and provides connection info.

    Returns a dict with connection status and performance metrics.
    """
    start_time = time.time()

    try:
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        await Neo4jConnectionManager.verify_connectivity()

        connection_time = time.time() - start_time

        return {"status": "healthy", "connection_time": connection_time, "driver_initialized": True}
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection_time": time.time() - start_time,
            "driver_initialized": False,
        }


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "neo4j: marks tests as requiring Neo4j database")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
