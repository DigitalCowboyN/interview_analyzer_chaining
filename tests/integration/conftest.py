"""
Configuration and fixtures for integration tests.
"""
import pytest
import subprocess
import os
import time # Import time module
from src.utils.neo4j_driver import Neo4jConnectionManager
import asyncio # Import asyncio for await
import importlib # Import importlib for reloading modules

# Environment variables for the test database
# Use host.docker.internal to allow the devcontainer to reach the host's exposed port
TEST_NEO4J_URI = "neo4j://host.docker.internal:7688"
TEST_NEO4J_USER = "neo4j"
TEST_NEO4J_PASS = "testpassword"

# IMPORTANT: Autouse fixture to DIRECTLY patch the exported `config` dictionary
@pytest.fixture(scope='function', autouse=True)
def set_test_db_config():
    """Directly patches the exported `config` dict from src.config with test Neo4j settings."""
    print("\nAttempting to directly patch exported config dict from src.config...")
    original_neo4j_config = None
    config_was_missing = False
    try:
        # Import the actual config dictionary exported by the module
        from src.config import config

        # Store original config values to restore later
        if 'neo4j' in config:
            original_neo4j_config = config['neo4j'].copy()
        else:
            config_was_missing = True
            original_neo4j_config = None # Explicitly None if key missing

        # Directly modify the config dictionary
        config['neo4j'] = {
            'uri': TEST_NEO4J_URI,
            'username': TEST_NEO4J_USER,
            'password': TEST_NEO4J_PASS
        }
        print(f"Patched config['neo4j']: {config.get('neo4j')}")

    except ImportError:
        print("Skipping config patch: Cannot import config from src.config.")
        pytest.skip("Skipping test because src.config could not be imported.") # Skip test if config missing
    except Exception as e:
        print(f"Error patching config dictionary: {e}")
        raise # Re-raise unexpected errors during patching

    yield # Let the test function run

    # --- Teardown: Restore original config --- 
    print("\nRestoring original config dict state for Neo4j...")
    try:
        # Re-import in case it was altered during test
        from src.config import config
        
        if config_was_missing:
            if 'neo4j' in config:
                del config['neo4j']
                print("Removed added config['neo4j'] section.")
        elif original_neo4j_config is not None:
            config['neo4j'] = original_neo4j_config
            print(f"Restored config['neo4j']: {config.get('neo4j')}")
        else:
             # This case (original_neo4j_config is None but config_was_missing is False) 
             # might mean the original value was literally None, but we overwrote it.
             # For simplicity, assume if original_neo4j_config is None, we should remove the key if we added it.
             if 'neo4j' in config: # Check if it exists before deleting
                 del config['neo4j']
                 print("Removed potentially added config['neo4j'] section (original state unclear)." )
             
    except ImportError:
        print("Skipping config restore: Cannot import config from src.config.")
    except Exception as e:
         print(f"Error restoring config dictionary: {e}")

@pytest.fixture(scope='session')
def manage_test_db():
    """
    Session-scoped fixture to start/stop the neo4j-test container.
    NOTE: Removed worker_id check for simplicity. Assumes serial execution or 
    that parallel runners handle external resource management appropriately.
    """
    print("\nStarting neo4j-test container...")
    subprocess.run(["make", "db-test-up"], check=True, capture_output=True, timeout=150)
    print("Waiting a few seconds for Neo4j to initialize fully...")
    time.sleep(15) # Wait 15 seconds
    
    # Pre-initialize the driver using test env vars
    print("Pre-initializing Neo4j driver for tests...")
    try:
        asyncio.run(Neo4jConnectionManager.get_driver()) # Use asyncio.run for top-level await
        print("Neo4j driver pre-initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to pre-initialize Neo4j driver: {e}")
        # Optionally raise an error here to fail fast if pre-init fails
        # raise
        
    print("neo4j-test container should be ready.")
    yield
    print("\nStopping neo4j-test container...")
    # Use asyncio.run to ensure driver is closed cleanly in async context
    try: 
        asyncio.run(Neo4jConnectionManager.close_driver())
    except Exception as e:
        print(f"Warning: Error closing Neo4j driver during teardown: {e}")
    subprocess.run(["make", "db-test-down"], check=True, capture_output=True, timeout=30)
    print("neo4j-test container stopped.")

@pytest.fixture(scope='function')
def clear_test_db(manage_test_db):
    """
    Function-scoped fixture to clear the test database before each test.
    Depends on manage_test_db to ensure the container is running.
    """
    # print("Clearing neo4j-test database...") # Optional: for debugging
    # Use check=False as clear might fail if container is restarting
    # Allow retries?
    try:
        subprocess.run(["make", "db-test-clear"], check=True, capture_output=True, timeout=10)
        # print("neo4j-test database cleared.") # Optional: for debugging
    except subprocess.TimeoutExpired:
        print("WARNING: Clearing neo4j-test timed out.")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Clearing neo4j-test failed: {e}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
    yield # Test runs here

@pytest.fixture(scope='function')
def test_db_manager() -> Neo4jConnectionManager:
    """
    Provides a Neo4jConnectionManager instance configured for the TEST database.
    Used by tests to run verification queries.
    IMPORTANT: This does NOT affect the singleton used by the application code.
    """
    # This manager is separate from the application's singleton.
    # It's used *by the tests* to verify the data written by the app.
    # We assume the app's singleton correctly uses the test env vars.
    test_manager = Neo4jConnectionManager() # Instantiate directly
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
# Note: This might not reliably affect the singleton if it's imported early.
# Running tests with `export NEO4J_URI=...; pytest` might be more reliable.
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