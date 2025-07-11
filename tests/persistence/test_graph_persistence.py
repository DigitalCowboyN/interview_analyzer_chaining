"""
Unit tests for src/persistence/graph_persistence.py
"""

import logging
from typing import Any, Dict
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest

from src.persistence.graph_persistence import save_analysis_to_graph
from src.utils.neo4j_driver import Neo4jConnectionManager

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# --- Fixtures ---


@pytest.fixture
def mock_connection_manager() -> MagicMock:
    """Provides a mock Neo4jConnectionManager with an async execute_query method."""
    manager = MagicMock(spec=Neo4jConnectionManager)
    manager.execute_query = AsyncMock()
    return manager


@pytest.fixture
def full_analysis_data() -> Dict[str, Any]:
    """Provides a sample analysis data dictionary with all fields populated."""
    return {
        "sentence_id": 1,
        "sequence_order": 1,
        "sentence": "This is the second sentence.",
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "persistence",
        "overall_keywords": ["test", "graph"],
        "domain_keywords": ["neo4j", "cypher"],
    }


@pytest.fixture
def partial_analysis_data() -> Dict[str, Any]:
    """Provides sample data missing some optional fields."""
    return {
        "sentence_id": 2,
        "sequence_order": 2,
        "sentence": "Another sentence.",
        "function_type": "declarative",
        # Missing structure_type, purpose
        "topic_level_1": "testing",
        # Missing topic_level_3
        "overall_keywords": ["test"],
        # Missing domain_keywords
    }


@pytest.fixture
def first_sentence_data() -> Dict[str, Any]:
    """Provides sample data for the first sentence (sequence_order 0)."""
    return {
        "sentence_id": 0,
        "sequence_order": 0,
        "sentence": "The very first sentence.",
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "introduction",
        "topic_level_1": "start",
        "topic_level_3": None,  # Explicitly None
        "overall_keywords": ["first", "start"],
        "domain_keywords": [],  # Empty list
    }


@pytest.fixture
def missing_core_data() -> Dict[str, Any]:
    """Provides sample data missing a core field (sentence_id)."""
    return {
        # 'sentence_id': None,
        "sequence_order": 3,
        "sentence": "Sentence with missing ID.",
        "function_type": "declarative",
    }


# --- Test Functions ---


async def test_save_analysis_success_full(
    full_analysis_data: Dict[str, Any],
    mock_connection_manager: MagicMock,  # Keep the mock manager fixture
):
    """Test successful save with all data fields present."""
    filename = "test_full.txt"

    # --- Setup Mocks ---
    # Mock the session object that the context manager will return
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()  # Mock the run method specifically

    # Mock the async context manager returned by get_session
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = (
        mock_session  # Return mock_session when entering
    )

    # Configure the connection manager mock
    mock_connection_manager.get_session.return_value = mock_session_cm

    # --- Execute ---
    await save_analysis_to_graph(full_analysis_data, filename, mock_connection_manager)

    # --- Assertions ---
    # Expected parameters based on full_analysis_data
    expected_params = {
        "filename": filename,
        "sentence_id": 1,
        "sequence_order": 1,
        "text": "This is the second sentence.",
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "to state a fact",
        "topic_level_1": "testing",
        "topic_level_3": "persistence",
        "overall_keywords": ["test", "graph"],
        "domain_keywords": ["neo4j", "cypher"],
    }

    # 1. Assert get_session was called
    mock_connection_manager.get_session.assert_awaited_once()

    # 2. Assert session.run calls
    calls = mock_session.run.await_args_list
    # Expected calls: Sentence(1) + Types(3) + Topics(1) + Keywords(2) + Follows(1) = 8
    assert (
        mock_session.run.await_count == 8
    ), f"Expected 8 session.run calls, got {mock_session.run.await_count}"

    # Check params on first call (Sentence Merge)
    # Note: The query itself isn't checked here, just the parameters passed.
    # Use ANY for more flexible matching in case of future changes
    assert calls[0].kwargs["parameters"] == expected_params

    # Verify that session.run was called with ANY query and the expected parameters
    mock_session.run.assert_any_call(ANY, parameters=expected_params)

    # More precise assertion using call object
    mock_session.run.assert_has_calls(
        [call(ANY, parameters=expected_params)], any_order=True
    )

    # Check params on a keyword call (Overall Keywords)
    # Find the call corresponding to the overall keywords query
    overall_kw_call = None
    for call_item in calls:
        # Check if the Cypher query string contains the relationship type
        if "MENTIONS_OVERALL_KEYWORD" in call_item.args[0]:
            overall_kw_call = call_item
            break
    assert overall_kw_call is not None, "Overall keyword query call not found"
    assert overall_kw_call.kwargs["parameters"] == expected_params

    # Verify that session.run was called with ANY query containing the keyword relationship
    mock_session.run.assert_any_call(ANY, parameters=expected_params)

    # More precise assertion using call object for keyword queries
    mock_session.run.assert_has_calls(
        [call(ANY, parameters=expected_params)], any_order=True
    )

    # Check params on the FOLLOWS call
    follows_call = None
    for call_item in calls:
        if "FOLLOWS" in call_item.args[0]:
            follows_call = call_item
            break
    assert follows_call is not None, "FOLLOWS query call not found"
    assert follows_call.kwargs["parameters"] == expected_params

    # Verify that session.run was called with ANY query containing the FOLLOWS relationship
    mock_session.run.assert_any_call(ANY, parameters=expected_params)

    # More precise assertion using call object for FOLLOWS query
    mock_session.run.assert_has_calls(
        [call(ANY, parameters=expected_params)], any_order=True
    )


async def test_save_analysis_success_partial(
    partial_analysis_data: Dict[str, Any],
    mock_connection_manager: MagicMock,  # Keep mock manager
):
    """Test successful save with missing optional data fields."""
    filename = "test_partial.txt"

    # --- Setup Mocks (same as full test) ---
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_connection_manager.get_session.return_value = mock_session_cm

    # --- Execute ---
    await save_analysis_to_graph(
        partial_analysis_data, filename, mock_connection_manager
    )

    # --- Assertions ---
    # 1. Assert get_session was called
    mock_connection_manager.get_session.assert_awaited_once()

    # 2. Check total calls: Sentence(1) + FuncType(1) + Topic1(1) + OverallKW(1) + Follows(1) = 5
    assert (
        mock_session.run.await_count == 5
    ), f"Expected 5 session.run calls, got {mock_session.run.await_count}"


async def test_save_analysis_first_sentence(
    first_sentence_data: Dict[str, Any],
    mock_connection_manager: MagicMock,  # Keep mock manager
):
    """Test saving the first sentence (sequence_order=0), :FOLLOWS should be skipped."""
    filename = "test_first.txt"

    # --- Setup Mocks (same as full test) ---
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_connection_manager.get_session.return_value = mock_session_cm

    # --- Execute ---
    await save_analysis_to_graph(first_sentence_data, filename, mock_connection_manager)

    # --- Assertions ---
    # 1. Assert get_session was called
    mock_connection_manager.get_session.assert_awaited_once()

    # 2. Check total calls: Sentence(1) + Types(3) + Topic1(1) + OverallKW(1) = 6
    # Note: Domain keywords list is empty. Follows is skipped. Topic_level_3 is None.
    assert (
        mock_session.run.await_count == 6
    ), f"Expected 6 session.run calls, got {mock_session.run.await_count}"


async def test_save_analysis_skips_missing_core(
    missing_core_data: Dict[str, Any], mock_connection_manager: MagicMock, caplog
):
    """Test that the function skips processing if core fields are missing."""
    filename = "test_missing.txt"
    with caplog.at_level(logging.WARNING):
        await save_analysis_to_graph(
            missing_core_data, filename, mock_connection_manager
        )

    # Assert no database calls were made
    mock_connection_manager.execute_query.assert_not_called()

    # Assert warning log was generated
    assert "Skipping graph save" in caplog.text
    assert "missing core fields" in caplog.text


async def test_save_analysis_db_error(
    full_analysis_data: Dict[str, Any],
    mock_connection_manager: MagicMock,  # Keep mock manager
    caplog,
):
    """Test that database errors during core save operations are propagated."""
    filename = "test_db_error.txt"
    db_error = RuntimeError("Neo4j connection failed")

    # --- Setup Mocks ---
    # Mock session.run to raise the error
    mock_session = AsyncMock()
    mock_session.run = AsyncMock(side_effect=db_error)  # Error on session.run
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_connection_manager.get_session.return_value = mock_session_cm

    # --- Execute & Assert ---
    with pytest.raises(RuntimeError, match="Neo4j connection failed"), caplog.at_level(
        logging.ERROR
    ):
        await save_analysis_to_graph(
            full_analysis_data, filename, mock_connection_manager
        )

    # Verify get_session was called
    mock_connection_manager.get_session.assert_awaited_once()
    # Verify the session.run was attempted
    mock_session.run.assert_awaited_once()
    # Verify error log
    assert (
        f"Failed during graph update for sentence {full_analysis_data['sentence_id']}"
        in caplog.text
    )


async def test_save_analysis_follows_error(
    full_analysis_data: Dict[str, Any],  # Use data where follows should run
    mock_connection_manager: MagicMock,  # Keep mock manager
    caplog,
):
    """Test that an error specifically during the :FOLLOWS query is warned but not propagated."""
    filename = "test_follows_error.txt"
    db_error = RuntimeError("FOLLOWS constraint violation")

    # --- Setup Mocks ---
    # Mock session.run to succeed initially, then fail on the FOLLOWS query
    mock_session = AsyncMock()
    num_expected_success_calls = 7  # Sentence(1) + Types(3) + Topics(1) + Keywords(2)
    side_effect_list = ([None] * num_expected_success_calls) + [db_error]
    mock_session.run = AsyncMock(side_effect=side_effect_list)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_connection_manager.get_session.return_value = mock_session_cm

    # --- Execute & Assert ---
    with caplog.at_level(logging.WARNING):
        # Should not raise an exception
        await save_analysis_to_graph(
            full_analysis_data, filename, mock_connection_manager
        )

    # Check get_session was called
    mock_connection_manager.get_session.assert_awaited_once()

    # Check all queries were attempted (session.run count should match side_effect list length)
    assert (
        mock_session.run.await_count == num_expected_success_calls + 1
    ), f"Expected {num_expected_success_calls + 1} session.run calls, got {mock_session.run.await_count}"

    # Check warning log for the FOLLOWS error
    assert (
        f"Could not create :FOLLOWS relationship for sentence {full_analysis_data['sentence_id']}"
        in caplog.text
    )
