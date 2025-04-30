"""
Unit tests for src/persistence/graph_persistence.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call, ANY
from typing import Dict, Any
from src.persistence.graph_persistence import save_analysis_to_graph
from src.utils.neo4j_driver import Neo4jConnectionManager
import logging

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
        'sentence_id': 1, 
        'sequence_order': 1, 
        'sentence': "This is the second sentence.",
        'function_type': 'declarative', 
        'structure_type': 'simple sentence', 
        'purpose': 'to state a fact',
        'topic_level_1': 'testing', 
        'topic_level_3': 'persistence', 
        'overall_keywords': ['test', 'graph'], 
        'domain_keywords': ['neo4j', 'cypher']
    }

@pytest.fixture
def partial_analysis_data() -> Dict[str, Any]:
    """Provides sample data missing some optional fields."""
    return {
        'sentence_id': 2, 
        'sequence_order': 2, 
        'sentence': "Another sentence.",
        'function_type': 'declarative', 
        # Missing structure_type, purpose
        'topic_level_1': 'testing', 
        # Missing topic_level_3
        'overall_keywords': ['test'], 
        # Missing domain_keywords
    }

@pytest.fixture
def first_sentence_data() -> Dict[str, Any]:
    """Provides sample data for the first sentence (sequence_order 0)."""
    return {
        'sentence_id': 0, 
        'sequence_order': 0, 
        'sentence': "The very first sentence.",
        'function_type': 'declarative', 
        'structure_type': 'simple sentence', 
        'purpose': 'introduction',
        'topic_level_1': 'start', 
        'topic_level_3': None, # Explicitly None
        'overall_keywords': ['first', 'start'], 
        'domain_keywords': [] # Empty list
    }

@pytest.fixture
def missing_core_data() -> Dict[str, Any]:
    """Provides sample data missing a core field (sentence_id)."""
    return {
        # 'sentence_id': None, 
        'sequence_order': 3, 
        'sentence': "Sentence with missing ID.",
        'function_type': 'declarative', 
    }

# --- Test Functions ---

async def test_save_analysis_success_full(
    full_analysis_data: Dict[str, Any], 
    mock_connection_manager: MagicMock
):
    """Test successful save with all data fields present."""
    filename = "test_full.txt"
    await save_analysis_to_graph(full_analysis_data, filename, mock_connection_manager)

    # Expected parameters based on full_analysis_data
    expected_params = {
        'filename': filename,
        'sentence_id': 1,
        'sequence_order': 1,
        'text': "This is the second sentence.",
        'function_type': 'declarative',
        'structure_type': 'simple sentence',
        'purpose': 'to state a fact',
        'topic_level_1': 'testing',
        'topic_level_3': 'persistence',
        'overall_keywords': ['test', 'graph'],
        'domain_keywords': ['neo4j', 'cypher']
    }

    # Check calls (order matters for some parts)
    calls = mock_connection_manager.execute_query.await_args_list
    
    # 1. Sentence Merge
    assert calls[0] == call(ANY, parameters=expected_params) # Check params on first call
    assert "MERGE (f:SourceFile {filename: $filename})" in calls[0].args[0]
    assert "MERGE (s:Sentence {sentence_id: $sentence_id, filename: $filename})" in calls[0].args[0]
    assert "MERGE (s)-[:PART_OF_FILE]->(f)" in calls[0].args[0]

    # 2. Types (FunctionType, StructureType, Purpose) - order might vary slightly
    type_calls = calls[1:4] # Expecting 3 calls for types
    type_names_seen = set()
    rel_types_seen = set()
    for c in type_calls:
        assert "MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})" in c.args[0]
        assert "MERGE (s)-[:HAS_" in c.args[0]
        if "MERGE (t:FunctionType {name: $function_type})" in c.args[0]:
            type_names_seen.add("FunctionType")
            rel_types_seen.add("HAS_FUNCTION_TYPE")
        elif "MERGE (t:StructureType {name: $structure_type})" in c.args[0]:
            type_names_seen.add("StructureType")
            rel_types_seen.add("HAS_STRUCTURE_TYPE")
        elif "MERGE (t:Purpose {name: $purpose})" in c.args[0]:
            type_names_seen.add("Purpose")
            rel_types_seen.add("HAS_PURPOSE")
        assert c.kwargs['parameters'] == expected_params
    assert type_names_seen == {"FunctionType", "StructureType", "Purpose"}
    assert rel_types_seen == {"HAS_FUNCTION_TYPE", "HAS_STRUCTURE_TYPE", "HAS_PURPOSE"}


    # 3. Topics
    assert calls[4] == call(ANY, parameters=expected_params)
    assert "MERGE (t1:Topic {name: $topic_level_1})" in calls[4].args[0]
    assert "MERGE (s)-[:HAS_TOPIC]->(t1)" in calls[4].args[0]
    assert "MERGE (t3:Topic {name: $topic_level_3})" in calls[4].args[0]
    assert "MERGE (s)-[:HAS_TOPIC]->(t3)" in calls[4].args[0]

    # 4. Keywords (Overall, Domain)
    keyword_calls = calls[5:7] # Expecting 2 calls for keywords
    rel_keywords_seen = set()
    for c in keyword_calls:
         assert "MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})" in c.args[0]
         assert "UNWIND $" in c.args[0]
         assert "MERGE (k:Keyword {text: keyword_text})" in c.args[0]
         if "UNWIND $overall_keywords" in c.args[0]:
             rel_keywords_seen.add("MENTIONS_OVERALL_KEYWORD")
         elif "UNWIND $domain_keywords" in c.args[0]:
              rel_keywords_seen.add("MENTIONS_DOMAIN_KEYWORD")
         assert c.kwargs['parameters'] == expected_params
    assert rel_keywords_seen == {"MENTIONS_OVERALL_KEYWORD", "MENTIONS_DOMAIN_KEYWORD"}

    # 5. Follows
    assert calls[7] == call(ANY, parameters=expected_params)
    assert "MATCH (s1:Sentence {sequence_order: $sequence_order - 1, filename: $filename})" in calls[7].args[0]
    assert "MERGE (s1)-[r:FOLLOWS]->(s2)" in calls[7].args[0]

    # Check total calls
    assert mock_connection_manager.execute_query.await_count == 8


async def test_save_analysis_success_partial(
    partial_analysis_data: Dict[str, Any], 
    mock_connection_manager: MagicMock
):
    """Test successful save with missing optional data fields."""
    filename = "test_partial.txt"
    await save_analysis_to_graph(partial_analysis_data, filename, mock_connection_manager)

    # Should have fewer calls than the full test
    calls = mock_connection_manager.execute_query.await_args_list
    
    # Check total calls: Sentence(1) + FuncType(1) + Topic1(1) + OverallKW(1) + Follows(1) = 5
    assert mock_connection_manager.execute_query.await_count == 5 
    
    # Verify certain relationships were NOT created
    query_strings = "\n".join([c.args[0] for c in calls])
    assert "StructureType" not in query_strings
    assert "Purpose" not in query_strings
    assert "domain_keywords" not in query_strings


async def test_save_analysis_first_sentence(
    first_sentence_data: Dict[str, Any], 
    mock_connection_manager: MagicMock
):
    """Test saving the first sentence (sequence_order=0), :FOLLOWS should be skipped."""
    filename = "test_first.txt"
    await save_analysis_to_graph(first_sentence_data, filename, mock_connection_manager)

    calls = mock_connection_manager.execute_query.await_args_list
    
    # Check total calls: Sentence(1) + Types(3) + Topic1(1) + OverallKW(1) = 6
    # Note: Domain keywords list is empty, so that query is skipped. Follows is skipped. Topic_level_3 is None.
    assert mock_connection_manager.execute_query.await_count == 6
    
    # Verify FOLLOWS query was not executed
    query_strings = "\n".join([c.args[0] for c in calls])
    assert "FOLLOWS" not in query_strings


async def test_save_analysis_skips_missing_core(
    missing_core_data: Dict[str, Any], 
    mock_connection_manager: MagicMock,
    caplog
):
    """Test that the function skips processing if core fields are missing."""
    filename = "test_missing.txt"
    with caplog.at_level(logging.WARNING):
        await save_analysis_to_graph(missing_core_data, filename, mock_connection_manager)

    # Assert no database calls were made
    mock_connection_manager.execute_query.assert_not_called()
    
    # Assert warning log was generated
    assert "Skipping graph save" in caplog.text
    assert "missing core fields" in caplog.text


async def test_save_analysis_db_error(
    full_analysis_data: Dict[str, Any], 
    mock_connection_manager: MagicMock,
    caplog
):
    """Test that database errors during core save operations are propagated."""
    filename = "test_db_error.txt"
    db_error = RuntimeError("Neo4j connection failed")
    mock_connection_manager.execute_query.side_effect = db_error

    with pytest.raises(RuntimeError, match="Neo4j connection failed"), caplog.at_level(logging.ERROR):
        await save_analysis_to_graph(full_analysis_data, filename, mock_connection_manager)

    # Assert first query attempt was made
    mock_connection_manager.execute_query.assert_awaited_once() 
    # Assert error log was generated
    assert f"Failed during graph update for sentence {full_analysis_data['sentence_id']}" in caplog.text
    assert "Neo4j connection failed" in caplog.text


async def test_save_analysis_follows_error(
    full_analysis_data: Dict[str, Any], # Use data where follows should run
    mock_connection_manager: MagicMock,
    caplog
):
    """Test that an error specifically during the :FOLLOWS query is warned but not propagated."""
    filename = "test_follows_error.txt"
    db_error = RuntimeError("FOLLOWS constraint violation")
    
    # Make previous queries succeed, only the last one (FOLLOWS) fails
    num_expected_success_calls = 7 # Sentence(1) + Types(3) + Topics(1) + Keywords(2)
    side_effect_list = ([None] * num_expected_success_calls) + [db_error]
    mock_connection_manager.execute_query.side_effect = side_effect_list

    with caplog.at_level(logging.WARNING):
         # Should not raise an exception
        await save_analysis_to_graph(full_analysis_data, filename, mock_connection_manager)

    # Check all queries were attempted
    assert mock_connection_manager.execute_query.await_count == num_expected_success_calls + 1
    
    # Assert warning log was generated for FOLLOWS failure
    assert "Could not create :FOLLOWS relationship" in caplog.text
    assert "FOLLOWS constraint violation" in caplog.text
    
    # Assert NO error log from the main try/except block
    error_logs = [rec.message for rec in caplog.records if rec.levelno == logging.ERROR]
    assert not any(f"Failed during graph update for sentence" in msg for msg in error_logs) 