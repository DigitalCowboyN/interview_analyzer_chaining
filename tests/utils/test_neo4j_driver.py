"""
Tests for src/utils/neo4j_driver.py
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from neo4j import AsyncDriver # Import for type hinting

# Module to test
from src.utils.neo4j_driver import Neo4jConnectionManager

# Define mock config fixtures specifically for these tests
@pytest.fixture
def mock_neo4j_config_valid():
    return {
        "neo4j": {
            "uri": "bolt://mockneo4j:7687",
            "username": "mockuser",
            "password": "mockpass"
        }
    }

@pytest.fixture
def mock_neo4j_config_missing():
    return {}

@pytest.fixture
def mock_neo4j_config_incomplete():
    return {
        "neo4j": {
            "uri": "bolt://mockneo4j:7687",
            "username": "mockuser"
            # Missing password
        }
    }

@pytest.fixture(autouse=True)
async def reset_neo4j_manager():
    """Ensure the Neo4jConnectionManager is reset before/after each test."""
    # Reset before test
    Neo4jConnectionManager._driver = None
    yield
    # Close after test if driver was created
    if Neo4jConnectionManager._driver:
        await Neo4jConnectionManager.close_driver()
    Neo4jConnectionManager._driver = None

# --- Tests for get_driver ---

@pytest.mark.asyncio
async def test_get_driver_success_first_call(mock_neo4j_config_valid):
    """Test successful driver initialization on the first call."""
    mock_driver_instance = AsyncMock(spec=AsyncDriver)

    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver_instance) as mock_neo4j_driver_call, \
         patch('src.utils.neo4j_driver.logger') as mock_logger:

        driver = await Neo4jConnectionManager.get_driver()

        assert driver is mock_driver_instance
        mock_neo4j_driver_call.assert_called_once_with(
            mock_neo4j_config_valid['neo4j']['uri'],
            auth=("mockuser", "mockpass")
        )
        mock_logger.info.assert_any_call("Initializing Neo4j Async Driver...")
        mock_logger.info.assert_any_call(f"Neo4j Async Driver initialized for URI: {mock_neo4j_config_valid['neo4j']['uri']}")
        assert Neo4jConnectionManager._driver is mock_driver_instance

@pytest.mark.asyncio
async def test_get_driver_success_subsequent_call(mock_neo4j_config_valid):
    """Test that subsequent calls return the existing driver instance."""
    mock_driver_instance = AsyncMock(spec=AsyncDriver)

    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver_instance) as mock_neo4j_driver_call:

        # First call
        driver1 = await Neo4jConnectionManager.get_driver()
        assert driver1 is mock_driver_instance
        mock_neo4j_driver_call.assert_called_once() # Called only once

        # Second call
        driver2 = await Neo4jConnectionManager.get_driver()
        assert driver2 is mock_driver_instance
        mock_neo4j_driver_call.assert_called_once() # Still only called once

@pytest.mark.asyncio
async def test_get_driver_missing_config(mock_neo4j_config_missing):
    """Test get_driver raises ValueError if neo4j config section is missing."""
    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_missing), \
         patch('src.utils.neo4j_driver.logger') as mock_logger:

        with pytest.raises(ValueError, match="Neo4j configuration section is missing"):
            await Neo4jConnectionManager.get_driver()
        assert Neo4jConnectionManager._driver is None
        mock_logger.critical.assert_called_once()

@pytest.mark.asyncio
async def test_get_driver_incomplete_config(mock_neo4j_config_incomplete):
    """Test get_driver raises ValueError if uri/user/pass is missing."""
    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_incomplete), \
         patch('src.utils.neo4j_driver.logger') as mock_logger:

        with pytest.raises(ValueError,
                           match="Neo4j URI, username, or password missing"):
            await Neo4jConnectionManager.get_driver()
        assert Neo4jConnectionManager._driver is None
        mock_logger.critical.assert_called_once()


@pytest.mark.asyncio
async def test_get_driver_init_exception(mock_neo4j_config_valid):
    """Test get_driver handles exceptions during driver initialization."""
    init_error = ConnectionError("Failed to connect to mock DB")
    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', side_effect=init_error) as mock_neo4j_driver_call, \
         patch('src.utils.neo4j_driver.logger') as mock_logger:

        with pytest.raises(ConnectionError, match="Failed to connect to mock DB"):
            await Neo4jConnectionManager.get_driver()

        mock_neo4j_driver_call.assert_called_once()
        mock_logger.critical.assert_any_call(f"Failed to initialize Neo4j driver: {init_error}", exc_info=True)
        assert Neo4jConnectionManager._driver is None

# --- Tests for close_driver ---
@pytest.mark.asyncio
async def test_close_driver_success(mock_neo4j_config_valid):
    # Test closing an initialized driver.
    mock_driver_instance = AsyncMock(spec=AsyncDriver)
    mock_driver_instance.close = AsyncMock() # Mock the close method
    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver_instance), \
         patch('src.utils.neo4j_driver.logger') as mock_logger:
        # Initialize driver first
        await Neo4jConnectionManager.get_driver()
        assert Neo4jConnectionManager._driver is mock_driver_instance
        # Close it
        await Neo4jConnectionManager.close_driver()
        mock_driver_instance.close.assert_awaited_once()
        mock_logger.info.assert_any_call("Closing Neo4j Async Driver...")
        mock_logger.info.assert_any_call("Neo4j Async Driver closed.")
        assert Neo4jConnectionManager._driver is None

@pytest.mark.asyncio
async def test_close_driver_no_driver():
    # Test closing when the driver hasn't been initialized.
    with patch('src.utils.neo4j_driver.logger') as mock_logger:
        # Ensure driver is None
        assert Neo4jConnectionManager._driver is None
        # Attempt to close
        await Neo4jConnectionManager.close_driver()
        # Verify nothing happened
        mock_logger.info.assert_not_called()
        mock_logger.debug.assert_not_called()
        # Or check specific log levels if needed
        assert Neo4jConnectionManager._driver is None

# --- Tests for get_session ---
@pytest.mark.asyncio
async def test_get_session_success(mock_neo4j_config_valid):
    """Test getting a session from an initialized driver."""
    mock_session = MagicMock() # Can mock AsyncSession if needed, but MagicMock works
    mock_driver_instance = AsyncMock(spec=AsyncDriver)
    mock_driver_instance.session.return_value = mock_session

    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver_instance):
        
        # get_session implicitly calls get_driver
        session_context_manager = await Neo4jConnectionManager.get_session()
        
        assert session_context_manager is mock_session
        mock_driver_instance.session.assert_called_once_with(database=None) # Check default db

@pytest.mark.asyncio
async def test_get_session_with_database(mock_neo4j_config_valid):
    """Test getting a session for a specific database."""
    mock_session = MagicMock()
    mock_driver_instance = AsyncMock(spec=AsyncDriver)
    mock_driver_instance.session.return_value = mock_session

    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_valid), \
         patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver_instance):
        
        session_context_manager = await Neo4jConnectionManager.get_session(database="customdb")
        
        assert session_context_manager is mock_session
        mock_driver_instance.session.assert_called_once_with(database="customdb")

@pytest.mark.asyncio
async def test_get_session_init_error(mock_neo4j_config_missing):
    #Test get_session propagates errors from get_driver.
    with patch('src.utils.neo4j_driver.config', mock_neo4j_config_missing):
        
        with pytest.raises(ValueError, match="configuration section is missing"): 
             _ = await Neo4jConnectionManager.get_session() 