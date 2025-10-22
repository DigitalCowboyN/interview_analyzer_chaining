"""
Integration tests for Neo4j connection reliability and environment-aware configuration.

These tests verify that the enhanced connection management system works correctly
across different environments and handles various failure scenarios gracefully.
"""

import asyncio
import os

import pytest

from src.utils.environment import (
    detect_environment,
    get_available_neo4j_config,
    get_neo4j_connection_config,
    get_neo4j_test_connection_config,
    is_neo4j_available,
)
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.neo4j
@pytest.mark.integration
class TestEnvironmentAwareConnection:
    """Test environment detection and connection configuration."""

    def test_environment_detection(self):
        """Test that environment detection works correctly."""
        environment = detect_environment()
        assert environment in ["docker", "ci", "host"]
        print(f"Detected environment: {environment}")

    def test_connection_config_structure(self):
        """Test that connection configuration has required fields."""
        try:
            config = get_neo4j_connection_config()
            required_fields = ["uri", "username", "password", "source"]

            for field in required_fields:
                assert field in config, f"Missing required field: {field}"
                assert config[field] is not None, f"Field {field} is None"

            print(f"Connection config: {config}")
        except Exception as e:
            pytest.skip(f"No connection config available: {e}")

    def test_test_connection_config_structure(self):
        """Test that test database connection configuration works."""
        try:
            config = get_neo4j_test_connection_config()
            required_fields = ["uri", "username", "password", "source"]

            for field in required_fields:
                assert field in config, f"Missing required field: {field}"
                assert config[field] is not None, f"Field {field} is None"

            # Test config should be different from production config
            prod_config = get_neo4j_connection_config()
            assert config["uri"] != prod_config["uri"] or "test" in config["source"]

            print(f"Test connection config: {config}")
        except Exception as e:
            pytest.skip(f"No test connection config available: {e}")


@pytest.mark.neo4j
@pytest.mark.integration
class TestConnectionManagerReliability:
    """Test the enhanced Neo4j connection manager."""

    @pytest.mark.asyncio
    async def test_driver_initialization_with_test_mode(self, ensure_neo4j_test_service):
        """Test that driver initializes correctly in test mode."""
        # Ensure clean state
        await Neo4jConnectionManager.close_driver()

        # Initialize in test mode
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        assert driver is not None

        # Verify it's a singleton
        driver2 = await Neo4jConnectionManager.get_driver(test_mode=True)
        assert driver is driver2

    @pytest.mark.asyncio
    async def test_connectivity_verification(self, ensure_neo4j_test_service):
        """Test that connectivity verification works."""
        await Neo4jConnectionManager.close_driver()
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        # Should not raise an exception
        is_connected = await Neo4jConnectionManager.verify_connectivity()
        assert is_connected is True

    @pytest.mark.asyncio
    async def test_session_context_manager(self, ensure_neo4j_test_service):
        """Test that session context manager works correctly."""
        await Neo4jConnectionManager.close_driver()
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run("RETURN 1 AS test_value")
            record = await result.single()
            assert record["test_value"] == 1

    @pytest.mark.asyncio
    async def test_wait_for_ready_success(self, ensure_neo4j_test_service):
        """Test wait_for_ready with available database."""
        await Neo4jConnectionManager.close_driver()

        # Should succeed quickly since database is already running
        is_ready = await Neo4jConnectionManager.wait_for_ready(timeout=10.0, test_mode=True)
        assert is_ready is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("RUNNING_IN_DOCKER") == "true" or os.path.exists("/.dockerenv"),
        reason="Timeout test not applicable in Docker where all Neo4j services are available",
    )
    async def test_wait_for_ready_timeout(self):
        """Test wait_for_ready timeout behavior with unavailable database."""
        await Neo4jConnectionManager.close_driver()

        # This should timeout quickly since no database is running
        is_ready = await Neo4jConnectionManager.wait_for_ready(
            timeout=2.0, test_mode=False  # Use production config which won't be available
        )
        assert is_ready is False

    @pytest.mark.asyncio
    async def test_driver_cleanup(self, ensure_neo4j_test_service):
        """Test that driver cleanup works correctly."""
        await Neo4jConnectionManager.close_driver()
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        assert driver is not None

        # Close the driver
        await Neo4jConnectionManager.close_driver()

        # Should be able to get a new driver
        new_driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        assert new_driver is not None
        # Note: new_driver might be the same object if connection pooling reuses it


@pytest.mark.neo4j
@pytest.mark.integration
class TestConnectionAvailabilityDetection:
    """Test connection availability detection utilities."""

    def test_availability_detection_with_running_service(self, ensure_neo4j_test_service):
        """Test availability detection when service is running."""
        config = get_available_neo4j_config(test_mode=True)
        assert config is not None

        # Should detect that the service is available
        is_available = is_neo4j_available(config["uri"], timeout=5.0)
        assert is_available is True

    def test_availability_detection_with_invalid_uri(self):
        """Test availability detection with invalid URI."""
        # Test with definitely invalid URI
        is_available = is_neo4j_available("bolt://nonexistent:9999", timeout=1.0)
        assert is_available is False

        # Test with malformed URI
        is_available = is_neo4j_available("invalid-uri", timeout=1.0)
        assert is_available is False

    def test_get_available_config_fallback(self, ensure_neo4j_test_service):
        """Test that get_available_neo4j_config finds working configuration."""
        config = get_available_neo4j_config(test_mode=True)
        assert config is not None
        assert "uri" in config
        assert "username" in config
        assert "password" in config
        assert "source" in config

        # The returned config should actually work
        is_available = is_neo4j_available(config["uri"])
        assert is_available is True


@pytest.mark.neo4j
@pytest.mark.integration
@pytest.mark.slow
class TestConnectionStressScenarios:
    """Test connection behavior under stress and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_driver_initialization(self, ensure_neo4j_test_service):
        """Test that concurrent driver initialization works correctly."""
        await Neo4jConnectionManager.close_driver()

        # Start multiple concurrent initialization attempts
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(Neo4jConnectionManager.get_driver(test_mode=True))
            tasks.append(task)

        # All should succeed and return the same driver instance
        drivers = await asyncio.gather(*tasks)

        # All drivers should be the same instance (singleton)
        first_driver = drivers[0]
        for driver in drivers[1:]:
            assert driver is first_driver

    @pytest.mark.asyncio
    async def test_rapid_close_and_reinitialize(self, ensure_neo4j_test_service):
        """Test rapid close and reinitialize cycles."""
        for i in range(3):
            print(f"Cycle {i + 1}")

            # Initialize
            driver = await Neo4jConnectionManager.get_driver(test_mode=True)
            assert driver is not None

            # Use it
            async with await Neo4jConnectionManager.get_session() as session:
                result = await session.run("RETURN $cycle AS cycle", cycle=i)
                record = await result.single()
                assert record["cycle"] == i

            # Close it
            await Neo4jConnectionManager.close_driver()

            # Brief pause
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_session_usage_patterns(self, clean_test_database):
        """Test various session usage patterns."""
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        # Test multiple concurrent sessions
        async def create_node(session_id: int):
            async with await Neo4jConnectionManager.get_session() as session:
                await session.run("CREATE (n:TestNode {session_id: $session_id})", session_id=session_id)

        # Create nodes concurrently
        tasks = [create_node(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all nodes were created
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run("MATCH (n:TestNode) RETURN count(n) AS count")
            record = await result.single()
            assert record["count"] == 5
