"""
Configuration and fixtures for integration tests.
"""

import asyncio  # Import asyncio for await
import os
import socket  # Import socket module
import subprocess
import time  # Import time module

import pytest

from src.utils.neo4j_driver import Neo4jConnectionManager

# Environment variables for the test database
# Use the service name and internal port for container-to-container communication
TEST_NEO4J_URI = "bolt://neo4j-test:7687"  # Reverted from localhost
TEST_NEO4J_USER = "neo4j"
TEST_NEO4J_PASS = "testpassword"


# --- Helper Function ---
def wait_for_port(
    host: str, port: int, timeout: float = 30.0, retry_interval: float = 1.0
):
    """Waits for a network port to become available on a host."""
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection((host, port), timeout=retry_interval):
                print(f"Successfully connected to {host}:{port}")
                return True
        except (socket.timeout, ConnectionRefusedError, socket.gaierror) as e:
            print(f"Waiting for {host}:{port}... ({type(e).__name__})")  # DEBUG
            if time.monotonic() - start_time >= timeout:
                print(f"Timeout waiting for {host}:{port} after {timeout} seconds.")
                raise TimeoutError(
                    f"Port {port} on host {host} did not become available within {timeout} seconds."
                ) from e
            time.sleep(retry_interval)


# --- End Helper Function ---


@pytest.fixture(scope="session")
def manage_test_db():
    """
    Session-scoped fixture to start/stop the neo4j-test container.
    NOTE: Removed worker_id check for simplicity. Assumes serial execution or
    that parallel runners handle external resource management appropriately.
    """
    print("\nStarting neo4j-test container...")
    subprocess.run(["make", "db-test-up"], check=True, capture_output=True, timeout=150)

    # Wait for the Neo4j Bolt port to be available
    print("Waiting for neo4j-test service port 7687...")
    try:
        wait_for_port("neo4j-test", 7687, timeout=60.0)  # Increased timeout
        print("neo4j-test container should be ready and port 7687 is open.")
    except TimeoutError as e:
        print(f"ERROR: Neo4j service did not become available: {e}")
        # Attempt to capture logs before teardown if port wait fails
        try:
            logs_result = subprocess.run(
                ["docker", "logs", "interview_analyzer_neo4j_test"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print("--- NEO4J LOGS ON PORT WAIT TIMEOUT ---")
            print(logs_result.stdout)
            print(logs_result.stderr)
            print("-----------------------------------------")
        except Exception as log_e:
            print(f"Could not capture logs after port wait timeout: {log_e}")
        raise  # Re-raise the TimeoutError to fail the setup

    # Pre-initialize the driver using test env vars
    print("Pre-initializing Neo4j driver for tests...")
    try:
        asyncio.run(
            Neo4jConnectionManager.get_driver()
        )  # Use asyncio.run for top-level await
        print("Neo4j driver pre-initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to pre-initialize Neo4j driver: {e}")
        # Optionally raise an error here to fail fast if pre-init fails
        # raise

    yield
    # --- Restore teardown logic ---
    print("\nStopping neo4j-test container...")
    # Use asyncio.run to ensure driver is closed cleanly in async context
    try:
        asyncio.run(Neo4jConnectionManager.close_driver())
    except Exception as e:
        print(f"Warning: Error closing Neo4j driver during teardown: {e}")

    subprocess.run(
        ["make", "db-test-down"], check=True, capture_output=True, timeout=30
    )
    print("neo4j-test container stopped.")


@pytest.fixture(scope="function")
async def clear_test_db(manage_test_db, set_test_db_env_vars):
    """
    Async function-scoped fixture to clear the test database using the driver.
    Depends on manage_test_db (container running) and set_test_db_env_vars (env vars set).
    """
    print("Clearing neo4j-test database using driver (async fixture)...")
    cleared_successfully = False
    try:
        # Now we can use await directly
        driver = await Neo4jConnectionManager.get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            if record and record["count"] == 0:
                cleared_successfully = True
                print("Database cleared successfully (verified count is 0).")
            else:
                print(
                    f"WARNING: Database clear verification failed. Count: {record['count'] if record else 'N/A'}"
                )

    except Exception as e:
        print(f"ERROR: Failed to clear neo4j-test database using driver: {e}")
        pytest.fail(f"Failed to clear test database: {e}")

    if not cleared_successfully:
        pytest.fail("Database clear could not be verified.")

    yield  # Test runs here
    print("Finished test, DB will be cleared again on next run.")


@pytest.fixture(scope="function")
def test_db_manager() -> Neo4jConnectionManager:
    """
    Provides a Neo4jConnectionManager instance configured for the TEST database.
    Used by tests to run verification queries.
    IMPORTANT: This does NOT affect the singleton used by the application code.
    """
    # This manager is separate from the application's singleton.
    # It's used *by the tests* to verify the data written by the app.
    # We assume the app's singleton correctly uses the test env vars.
    test_manager = Neo4jConnectionManager()  # Instantiate directly
    # Manually configure its internal driver state for the test DB
    # This relies on internal implementation details - less ideal but avoids
    # complex patching if the singleton init relies only on env vars.
    # A better approach might be to have get_driver accept optional args.

    # For simplicity now, assume tests can just create their own driver instance
    # for verification, bypassing the singleton manager.
    # Alternatively, configure the singleton via environment variables before test run.

    # Let's return a configured instance, assuming test env vars are set
    # when pytest is run.
    # We need to manage its lifecycle if we create it manually.
    # TODO: Revisit lifecycle management if needed.
    return test_manager


# Fixture to set environment variables for the test session
# Use function scope because monkeypatch is function-scoped.
# Use autouse=True to ensure it runs before each test.


@pytest.fixture(scope="function", autouse=True)
def set_test_db_env_vars(monkeypatch):
    """Sets environment variables for the test Neo4j database before each test function."""
    print("\nSetting test Neo4j env vars for function...")

    # Force close existing driver to ensure re-initialization with new env vars
    # Need to run this synchronously within the fixture setup
    try:
        # Check if driver exists before attempting async close
        if Neo4jConnectionManager._driver is not None:
            print("Closing existing Neo4j driver before test...")
            asyncio.run(Neo4jConnectionManager.close_driver())  # Use asyncio.run
            print("Existing Neo4j driver closed.")
        else:
            print("No existing Neo4j driver to close.")
            # Explicitly set _driver to None in case initialization failed previously
            Neo4jConnectionManager._driver = None
    except Exception as e:
        print(f"Warning: Error force-closing Neo4j driver: {e}")
        Neo4jConnectionManager._driver = None  # Ensure it's None if close fails

    monkeypatch.setenv("NEO4J_URI", TEST_NEO4J_URI)
    monkeypatch.setenv("NEO4J_USER", TEST_NEO4J_USER)
    monkeypatch.setenv("NEO4J_PASSWORD", TEST_NEO4J_PASS)
    print(f"Set NEO4J_URI={os.getenv('NEO4J_URI')}")
    # No need to reload config object explicitly, as Neo4jConnectionManager now reads env vars.
    yield  # Let the function run
    print("\nTest function finished. Env vars automatically cleaned up by monkeypatch.")


# Commented out original attempt
# @pytest.fixture(scope='session', autouse=True)
# def set_test_db_env_vars(monkeypatch):
#     """Sets environment variables for the test Neo4j database."""
#     print("Setting test Neo4j env vars...")
#     monkeypatch.setenv("NEO4J_URI", TEST_NEO4J_URI)
#     monkeypatch.setenv("NEO4J_USER", TEST_NEO4J_USER)
#     monkeypatch.setenv("NEO4J_PASSWORD", TEST_NEO4J_PASS)
#     # Ensure config reloads if it caches values
#     from src import config
#     config.reload_config() # Assuming a reload method exists
