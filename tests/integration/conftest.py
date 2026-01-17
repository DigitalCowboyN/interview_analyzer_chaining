"""
Configuration and fixtures for integration tests with environment-aware Neo4j management.
"""

import asyncio
import os
import subprocess
import time
from typing import AsyncGenerator

import pytest
import requests

from src.events.store import EventStoreClient
from src.utils.environment import (
    check_eventstore_ready,
    check_neo4j_test_ready,
    detect_environment,
    get_available_neo4j_config,
    is_eventstore_externally_managed,
    is_service_externally_managed,
)
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.fixture(scope="session")
def ensure_neo4j_test_service():
    """
    Session-scoped fixture to ensure Neo4j test service is available.

    This fixture is environment-aware:
    - In Docker/CI: Assumes service is externally managed (e.g., by docker-compose)
    - On Host: Starts and stops the service using Makefile targets

    This prevents "docker command not found" errors inside containers.
    """
    environment = detect_environment()
    externally_managed = is_service_externally_managed("neo4j-test", 7687)

    print("\n=== Setting up Neo4j test environment ===")
    print(f"Environment: {environment}")
    print(f"Externally managed: {externally_managed}")

    service_started_by_fixture = False

    if externally_managed:
        # Service is managed externally (e.g., by docker-compose)
        print("Service is externally managed - verifying availability...")

        # Just verify the service is accessible
        if not check_neo4j_test_ready(timeout=10.0):
            pytest.fail(
                "Neo4j test service is not accessible. "
                "Ensure docker-compose services are running (make db-test-up from host)."
            )
        print("Neo4j test service verified and ready!")

    else:
        # We need to start the service ourselves (host environment)
        print("Starting neo4j-test container...")
        try:
            subprocess.run(["make", "db-test-up"], check=True, capture_output=True, timeout=120, text=True)
            print("Neo4j test container started successfully")
            service_started_by_fixture = True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start Neo4j test container: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print("Timeout starting Neo4j test container")
            raise
        except FileNotFoundError:
            pytest.fail("Could not run 'make db-test-up'. " "Ensure you have make and docker installed.")

        # Wait for Neo4j to become ready
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

    # Cleanup - only if we started the service
    if service_started_by_fixture:
        print("\n=== Cleaning up Neo4j test environment ===")
        try:
            subprocess.run(["make", "db-test-down"], check=True, capture_output=True, timeout=30)
            print("Neo4j test container stopped successfully")
        except Exception as e:
            print(f"Warning: Failed to clean up Neo4j test container: {e}")
    else:
        print("\n=== Neo4j test service is externally managed - skipping cleanup ===")


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

    Environment variables can be overridden:
    - NEO4J_TEST_URI (full URI)
    - NEO4J_TEST_HOST, NEO4J_TEST_PORT (for URI construction)
    - NEO4J_TEST_USER
    - NEO4J_TEST_PASSWORD
    - EVENTSTORE_HOST, EVENTSTORE_PORT (for connection string)
    """
    environment = detect_environment()

    # Neo4j Test Configuration
    # Allow override of test URI, otherwise construct from host/port
    neo4j_test_uri = os.getenv("NEO4J_TEST_URI")
    if not neo4j_test_uri:
        if environment == "docker":
            # Inside Docker container - use service names
            neo4j_host = os.getenv("NEO4J_TEST_HOST", "neo4j-test")
            neo4j_port = os.getenv("NEO4J_TEST_PORT", "7687")
        else:
            # Host or CI - use localhost with exposed port (from docker-compose.yml)
            neo4j_host = os.getenv("NEO4J_TEST_HOST", "localhost")
            neo4j_port = os.getenv("NEO4J_TEST_PORT", "7688")
        neo4j_test_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    monkeypatch.setenv("NEO4J_TEST_URI", neo4j_test_uri)
    monkeypatch.setenv("NEO4J_TEST_USER", os.getenv("NEO4J_TEST_USER", "neo4j"))
    monkeypatch.setenv("NEO4J_TEST_PASSWORD", os.getenv("NEO4J_TEST_PASSWORD", "testpassword"))

    print(f"Test environment configured for {environment} context")
    print(f"  Neo4j Test URI: {neo4j_test_uri}")

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
        await Neo4jConnectionManager.get_driver(test_mode=True)
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


# EventStoreDB fixtures for M2.7 testing


@pytest.fixture(scope="session")
def ensure_eventstore_service():
    """
    Session-scoped fixture to ensure EventStoreDB service is available.

    This fixture is environment-aware:
    - In Docker/CI: Assumes service is externally managed (e.g., by docker-compose)
    - On Host: Starts and stops the service using Makefile targets

    This prevents "docker command not found" errors inside containers.
    """
    environment = detect_environment()
    externally_managed = is_eventstore_externally_managed("eventstore", 2113)

    print("\n=== Setting up EventStoreDB environment ===")
    print(f"Environment: {environment}")
    print(f"Externally managed: {externally_managed}")

    service_started_by_fixture = False

    if externally_managed:
        # Service is managed externally (e.g., by docker-compose)
        print("Service is externally managed - verifying availability...")

        # Just verify the service is accessible
        if not check_eventstore_ready(timeout=10.0):
            pytest.fail(
                "EventStoreDB service is not accessible. "
                "Ensure docker-compose services are running (make eventstore-up from host)."
            )
        print("EventStoreDB service verified and ready!")

    else:
        # We need to start the service ourselves (host environment)
        print("Starting EventStoreDB container...")
        try:
            subprocess.run(["make", "eventstore-up"], check=True, capture_output=True, timeout=120, text=True)
            print("EventStoreDB container started successfully")
            service_started_by_fixture = True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start EventStoreDB container: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print("Timeout starting EventStoreDB container")
            raise
        except FileNotFoundError:
            pytest.fail("Could not run 'make eventstore-up'. " "Ensure you have make and docker installed.")

        # Wait for EventStoreDB to become ready
        print("Waiting for EventStoreDB to become ready...")
        max_retries = 30
        retry_delay = 2

        for attempt in range(max_retries):
            if check_eventstore_ready(timeout=3.0):
                print("EventStoreDB service is ready for testing!")
                break

            if attempt < max_retries - 1:
                print(f"Waiting for EventStoreDB... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
        else:
            # Attempt to get logs for debugging
            try:
                logs_result = subprocess.run(
                    ["docker", "logs", "interview_analyzer_eventstore"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                print("=== EVENTSTORE CONTAINER LOGS ===")
                print(logs_result.stdout)
                print(logs_result.stderr)
                print("================================")
            except Exception as log_e:
                print(f"Could not capture logs: {log_e}")

            pytest.fail("EventStoreDB did not become healthy within timeout")

    yield  # Tests run here

    # Cleanup - only if we started the service
    if service_started_by_fixture:
        print("\n=== Cleaning up EventStoreDB environment ===")
        try:
            subprocess.run(["make", "eventstore-down"], check=True, capture_output=True, timeout=30)
            print("EventStoreDB container stopped successfully")
        except Exception as e:
            print(f"Warning: Failed to clean up EventStoreDB container: {e}")
    else:
        print("\n=== EventStoreDB service is externally managed - skipping cleanup ===")


@pytest.fixture(scope="function")
async def event_store_client(ensure_eventstore_service):
    """
    Provides an EventStoreDB client for testing.

    Connection string is environment-aware (eventstore in Docker, localhost on host).
    Can be overridden with EVENTSTORE_TEST_CONNECTION_STRING, or constructed from:
    - EVENTSTORE_HOST
    - EVENTSTORE_PORT
    """
    from src.events.store import EventStoreClient as ESClient

    # Check for explicit connection string override
    connection_string = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")

    if not connection_string:
        # Use environment-aware defaults, but allow port/host override
        environment = detect_environment()
        if environment in ("docker", "ci"):
            host = os.getenv("EVENTSTORE_HOST", "eventstore")
            port = os.getenv("EVENTSTORE_PORT", "2113")
        else:
            host = os.getenv("EVENTSTORE_HOST", "localhost")
            port = os.getenv("EVENTSTORE_PORT", "2113")

        connection_string = f"esdb://{host}:{port}?tls=false"

    client = ESClient(connection_string)
    await client.connect()

    yield client

    await client.disconnect()


@pytest.fixture(scope="function")
async def clean_event_store(event_store_client):
    """
    Clears test event streams before each test.

    Deletes known test streams to ensure test isolation. This is safe because:
    1. Tests use deterministic UUIDs based on known test filenames
    2. We can calculate which streams will be created
    3. We only delete those specific test streams

    Note: In production, you would never delete streams. This is only for testing.
    """
    import uuid
    from esdbclient import StreamState

    # Collect all test streams to clean up
    test_streams = []

    # Pattern 1: Common test file name used in test fixtures
    test_filenames = [
        "test_interview.txt",
        # Pattern 2: Concurrent test files (test_performance.py)
        *[f"concurrent_test_{i}.txt" for i in range(20)],
        # Pattern 3: E2E test files
        "e2e_test_file.txt",
        "test_file_1.txt",
        "test_file_2.txt",
        "test_file_3.txt",
        # Pattern 4: Large file tests
        "large_test_file.txt",
        # Pattern 5: Error recovery tests
        "error_test_file.txt",
    ]

    for test_filename in test_filenames:
        # Calculate deterministic interview_id (matches pipeline logic)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{test_filename}"))
        test_streams.append(f"Interview-{interview_id}")

        # Calculate sentence IDs (tests typically have up to 100 sentences for large files)
        for i in range(100):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            test_streams.append(f"Sentence-{sentence_id}")

    def delete_streams(streams):
        """Delete streams, ignoring errors for non-existent streams."""
        for stream_name in streams:
            try:
                event_store_client._client.delete_stream(
                    stream_name,
                    current_version=StreamState.ANY
                )
            except Exception:
                # Stream doesn't exist or other errors - ignore
                pass

    # Delete test streams before test runs
    delete_streams(test_streams)

    yield event_store_client

    # Clean up after test as well (for extra safety)
    delete_streams(test_streams)


@pytest.fixture(scope="function")
def sample_interview_file(tmp_path):
    """
    Creates a sample interview text file for testing.

    Returns the path to a temporary file with test content.
    """
    content = """This is a test interview.
It contains multiple sentences.
Each sentence will be analyzed.
This helps us test the pipeline."""

    file_path = tmp_path / "test_interview.txt"
    file_path.write_text(content)

    return file_path


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "neo4j: marks tests as requiring Neo4j database")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "eventstore: marks tests as requiring EventStoreDB")
