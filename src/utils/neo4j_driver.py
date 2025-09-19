"""
src/utils/neo4j_driver.py

Utility for managing the Neo4j asynchronous driver instance with environment-aware configuration.
"""

import asyncio
import os
from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import config  # Import the global config object
from src.utils.environment import detect_environment, get_available_neo4j_config
from src.utils.logger import get_logger

logger = get_logger()


class Neo4jConnectionManager:
    _driver: Optional[AsyncDriver] = None
    _lock = asyncio.Lock()  # Lock to prevent race conditions during initialization

    @classmethod
    async def get_driver(cls, test_mode: bool = False) -> AsyncDriver:
        """
        Gets the singleton Neo4j AsyncDriver instance with environment-aware configuration.

        Initializes the driver on first call using intelligent environment detection
        and connection availability testing. Supports both production and test modes.

        Args:
            test_mode: If True, use test database configuration

        Returns:
            AsyncDriver: The initialized Neo4j driver instance.

        Raises:
            ValueError: If no valid Neo4j configuration is found.
            Exception: Propagates errors from driver initialization.
        """
        if cls._driver is None:
            async with cls._lock:
                # Double-check if another coroutine initialized it while waiting for the lock
                if cls._driver is None:
                    environment = detect_environment()
                    logger.info(f"Initializing Neo4j Async Driver in {environment} environment...")

                    # Get environment-aware configuration
                    connection_config = get_available_neo4j_config(test_mode=test_mode)

                    if not connection_config:
                        error_msg = (
                            f"No available Neo4j configuration found for {environment} environment. "
                            f"Please ensure Neo4j is running and accessible."
                        )
                        logger.critical(error_msg)
                        raise ValueError(error_msg)

                    try:
                        uri = connection_config["uri"]
                        username = connection_config["username"]
                        password = connection_config["password"]
                        config_source = connection_config["source"]
                    except KeyError as e:
                        error_msg = f"Incomplete Neo4j configuration: missing {e}"
                        logger.critical(error_msg)
                        raise ValueError(f"Neo4j driver initialization failed: {error_msg}")

                    try:
                        auth = (username, password)
                        cls._driver = AsyncGraphDatabase.driver(
                            uri,
                            auth=auth,
                            max_connection_lifetime=3600,  # 1 hour
                            max_connection_pool_size=10,  # Reasonable pool size
                            connection_acquisition_timeout=30,  # 30 second timeout
                        )

                        logger.info(
                            f"Neo4j Async Driver initialized for URI: {uri} "
                            f"(source: {config_source}, environment: {environment})"
                        )

                        # Verify connectivity on initialization
                        await cls.verify_connectivity()
                        logger.info("Neo4j connectivity verified successfully.")

                    except Exception as e:
                        logger.critical(
                            f"Failed to initialize Neo4j driver for {uri}: {e}",
                            exc_info=True,
                        )
                        cls._driver = None  # Ensure driver remains None on failure
                        raise ValueError(f"Neo4j driver initialization failed: {e}")  # Wrap in ValueError
        return cls._driver

    @classmethod
    async def verify_connectivity(cls, timeout: float = 10.0) -> bool:
        """
        Verify Neo4j driver connectivity with a simple query.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            bool: True if connection is successful

        Raises:
            Exception: If connection verification fails
        """
        if cls._driver is None:
            raise ValueError("Driver not initialized. Call get_driver() first.")

        try:
            async with cls._driver.session() as session:
                # Simple connectivity test
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if record and record["test"] == 1:
                    return True
                else:
                    raise Exception("Unexpected result from connectivity test")
        except Exception as e:
            logger.error(f"Neo4j connectivity verification failed: {e}")
            raise

    @classmethod
    async def wait_for_ready(cls, timeout: float = 30.0, test_mode: bool = False) -> bool:
        """
        Wait for Neo4j to become ready with retry logic.

        Args:
            timeout: Maximum time to wait in seconds
            test_mode: If True, use test database configuration

        Returns:
            bool: True if Neo4j becomes ready within timeout
        """
        import time

        start_time = time.time()
        retry_interval = 2.0

        logger.info(f"Waiting for Neo4j to become ready (timeout: {timeout}s)...")

        while time.time() - start_time < timeout:
            try:
                # Force re-initialization to test current availability
                if cls._driver:
                    await cls.close_driver()

                await cls.get_driver(test_mode=test_mode)
                logger.info("Neo4j is ready!")
                return True

            except Exception as e:
                logger.debug(f"Neo4j not ready yet: {e}")
                await asyncio.sleep(retry_interval)

        logger.error(f"Neo4j did not become ready within {timeout} seconds")
        return False

    @classmethod
    async def close_driver(cls):
        """Closes the singleton Neo4j driver instance if it exists."""
        async with cls._lock:
            if cls._driver:
                logger.info("Closing Neo4j Async Driver...")
                await cls._driver.close()
                cls._driver = None
                logger.info("Neo4j Async Driver closed.")

    @classmethod
    async def get_session(cls, database: Optional[str] = None):
        """
        Provides an async context manager for a Neo4j session.

        Example usage:
            async with await Neo4jConnectionManager.get_session() as session:
                result = await session.run(...)

        Args:
            database (Optional[str]): The specific database to connect to (defaults to Neo4j default).

        Returns:
            An async context manager yielding an AsyncSession.
        """
        driver = await cls.get_driver()
        # The session itself is the async context manager
        return driver.session(database=database)


# Optional: Define functions to be called on FastAPI startup/shutdown
async def initialize_neo4j_driver():
    await Neo4jConnectionManager.get_driver()


async def shutdown_neo4j_driver():
    await Neo4jConnectionManager.close_driver()


# Example of how to use in an async function:
async def example_neo4j_query():
    try:
        async with await Neo4jConnectionManager.get_session() as session:
            # Example: Check connection by running a simple query
            result = await session.run("RETURN 1")
            record = await result.single()
            logger.info(f"Neo4j query result: {record[0] if record else 'No result'}")
    except Exception as e:
        logger.error(f"Neo4j example query failed: {e}", exc_info=True)


# To integrate with FastAPI, you might add lifespan events in main.py:
#
# from contextlib import asynccontextmanager
# from src.utils.neo4j_driver import initialize_neo4j_driver, shutdown_neo4j_driver
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Actions on startup
#     await initialize_neo4j_driver()
#     yield
#     # Actions on shutdown
#     await shutdown_neo4j_driver()
#
# app = FastAPI(lifespan=lifespan, ...)

# Create the singleton instance for easy import
connection_manager = Neo4jConnectionManager()
