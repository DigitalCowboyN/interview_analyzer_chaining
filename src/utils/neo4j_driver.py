"""
src/utils/neo4j_driver.py

Utility for managing the Neo4j asynchronous driver instance.
"""

import asyncio
import os  # <---- Added os import
from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import config  # Import the global config object
from src.utils.logger import get_logger

logger = get_logger()


class Neo4jConnectionManager:
    _driver: Optional[AsyncDriver] = None
    _lock = asyncio.Lock()  # Lock to prevent race conditions during initialization

    @classmethod
    async def get_driver(cls) -> AsyncDriver:
        """
        Gets the singleton Neo4j AsyncDriver instance.

        Initializes the driver on first call using configuration first from environment
        variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) and falling back to src.config.
        Uses an async lock to ensure thread-safe initialization.

        Returns:
            AsyncDriver: The initialized Neo4j driver instance.

        Raises:
            ValueError: If Neo4j configuration is missing or invalid from both env vars and src.config.
            Exception: Propagates errors from driver initialization (e.g., connection issues).
        """
        if cls._driver is None:
            async with cls._lock:
                # Double-check if another coroutine initialized it while waiting for the lock
                if cls._driver is None:
                    logger.info("Initializing Neo4j Async Driver...")
                    uri = None
                    username = None
                    password = None
                    config_source = "unknown"

                    try:
                        # Prioritize Environment Variables
                        uri = os.getenv("NEO4J_URI")
                        username = os.getenv("NEO4J_USER")
                        password = os.getenv(
                            "NEO4J_PASSWORD"
                        )  # Returns None if not set

                        if uri and username and password is not None:
                            config_source = "environment variables"
                            logger.info(
                                "Using Neo4j config from environment variables."
                            )
                        else:
                            # Fallback to global config object
                            logger.info(
                                "Attempting to use Neo4j config from global config object."
                            )
                            neo4j_config = config.get("neo4j")
                            if neo4j_config:
                                uri = neo4j_config.get("uri")
                                username = neo4j_config.get("username")
                                # Password should be loaded from env via config if set
                                password = neo4j_config.get("password")
                                config_source = "global config object"
                            else:
                                raise ValueError(
                                    "Neo4j configuration not found in environment variables or global config."
                                )

                        if (
                            not uri or not username or password is None
                        ):  # Final check after fallback attempt
                            raise ValueError(
                                f"Neo4j URI, username, or password missing in configuration "
                                f"(source: {config_source})."
                            )

                        auth = (username, password)
                        # max_connection_lifetime=3600 * 24 * 30, # Example: 30 days
                        # max_connection_pool_size=50,
                        # connection_acquisition_timeout=60
                        cls._driver = AsyncGraphDatabase.driver(uri, auth=auth)
                        logger.info(
                            f"Neo4j Async Driver initialized for URI: {uri} (from {config_source})"
                        )
                        # Optional: Verify connectivity on initialization
                        # await cls._driver.verify_connectivity()
                        # logger.info("Neo4j connectivity verified.")
                    except Exception as e:
                        logger.critical(
                            f"Failed to initialize Neo4j driver: {e}", exc_info=True
                        )
                        cls._driver = None  # Ensure driver remains None on failure
                        raise  # Re-raise the exception
        return cls._driver

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
