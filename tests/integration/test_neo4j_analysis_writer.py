# tests/integration/test_neo4j_analysis_writer.py
"""
Integration tests for the Neo4j analysis writer implementation of SentenceAnalysisWriter protocol.
"""

import json
from typing import Any, Dict, List, Set

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.utils.neo4j_driver import Neo4jConnectionManager

# Mark all tests in this module as asyncio and require Neo4j
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# --- Tests for Neo4jAnalysisWriter ---


async def test_neo4j_analysis_writer_init() -> None:
    """Tests basic initialization of Neo4jAnalysisWriter."""
    project_id: str = "test-project-123"
    interview_id: str = "test-interview-456"

    writer = Neo4jAnalysisWriter(project_id, interview_id)

    assert writer.project_id == project_id
    assert writer.interview_id == interview_id
    assert writer.get_identifier() == interview_id


async def test_neo4j_analysis_writer_init_empty_ids() -> None:
    """Tests that initialization raises ValueError for empty IDs."""
    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jAnalysisWriter("", "interview-123")

    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jAnalysisWriter("project-123", "")

    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jAnalysisWriter("", "")


async def test_neo4j_analysis_writer_initialize_finalize(clean_test_database: Any) -> None:
    """Tests basic initialize and finalize operations."""
    project_id: str = "test-project-init"
    interview_id: str = "test-interview-init"
    writer = Neo4jAnalysisWriter(project_id, interview_id)

    # These should not raise exceptions
    await writer.initialize()
    await writer.finalize()

    # Verify identifier
    assert writer.get_identifier() == interview_id


async def test_neo4j_analysis_writer_write_basic_result(clean_test_database: Any) -> None:
    """Tests writing a basic analysis result with all dimensions."""
    project_id: str = "test-project-write"
    interview_id: str = "test-interview-write"

    # First, set up the prerequisite data using map storage
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Now test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Create a comprehensive analysis result
    analysis_result: Dict[str, Any] = {
        "sentence_id": 0,
        "function_type": "declarative",
        "structure_type": "simple",
        "purpose": "testing",
        "topics": ["software", "testing"],
        "overall_keywords": ["test", "sentence", "analysis"],
        "domain_keywords": ["neo4j", "database"],
    }

    # Write the result - this is the key test of our conversion
    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify the analysis was written by checking read_analysis_ids
    analysis_ids: Set[int] = await writer.read_analysis_ids()
    assert analysis_ids == {0}


async def test_neo4j_analysis_writer_missing_sentence_error(clean_test_database):
    """Tests that writing analysis for non-existent sentence raises ValueError."""
    project_id = "test-project-missing"
    interview_id = "test-interview-missing"

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Try to write analysis for a sentence that doesn't exist
    analysis_result = {"sentence_id": 999, "function_type": "declarative"}  # Non-existent sentence

    with pytest.raises(ValueError, match="Sentence node 999 not found"):
        await writer.write_result(analysis_result)


async def test_neo4j_analysis_writer_missing_sentence_id(clean_test_database):
    """Tests that analysis result without sentence_id is handled gracefully."""
    project_id = "test-project-no-id"
    interview_id = "test-interview-no-id"

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Analysis result without sentence_id
    analysis_result = {
        "function_type": "declarative"
        # Missing sentence_id
    }

    # Should not raise exception, just log and return
    await writer.write_result(analysis_result)
    await writer.finalize()

    # No analysis should be written
    analysis_ids = await writer.read_analysis_ids()
    assert analysis_ids == set()


# --- Tests for Dimension Relationship Handling ---


async def test_single_value_dimensions_basic(clean_test_database):
    """Tests basic single-value dimension handling (function, structure, purpose)."""
    project_id = "test-project-single"
    interview_id = "test-interview-single"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with single-value dimensions
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with all single-value dimensions
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "structure_type": "simple",
        "purpose": "testing",
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify the relationships were created
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function type relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "declarative"

        # Check structure type relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] == "simple"

        # Check purpose relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_PURPOSE]->(p:Purpose)
            RETURN p.name as purpose_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["purpose_name"] == "testing"


async def test_single_value_dimensions_overwrite(clean_test_database):
    """Tests that single-value dimensions are overwritten when analysis is rerun."""
    project_id = "test-project-overwrite"
    interview_id = "test-interview-overwrite"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "function_type": "declarative", "structure_type": "simple"}
    await writer.write_result(analysis_result_1)

    # Second analysis with different values
    analysis_result_2 = {"sentence_id": 0, "function_type": "interrogative", "structure_type": "complex"}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify only the latest values are present
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function type - should be updated
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "interrogative"

        # Check structure type - should be updated
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] == "complex"


async def test_multi_value_dimensions_basic(clean_test_database):
    """Tests basic multi-value dimension handling (keywords, topics, domain_keywords)."""
    project_id = "test-project-multi"
    interview_id = "test-interview-multi"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry(
        {"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence about software development."}
    )
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with multi-value dimensions
    analysis_result = {
        "sentence_id": 0,
        "topics": ["software", "development", "testing"],
        "overall_keywords": ["test", "sentence", "analysis"],  # Fixed: Use overall_keywords
        "domain_keywords": ["neo4j", "database", "graph"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify the relationships were created
    async with await Neo4jConnectionManager.get_session() as session:
        # Check topics - should be unlimited
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN t.name as topic_name
            ORDER BY t.name
        """
        )
        topics: List[str] = []
        async for record in result:
            topics.append(record["topic_name"])
        assert sorted(topics) == ["development", "software", "testing"]

        # Check keywords - should have default limit of 6
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text
            ORDER BY k.text
        """
        )
        keywords: List[str] = []
        async for record in result:
            keywords.append(record["keyword_text"])
        assert sorted(keywords) == ["analysis", "sentence", "test"]

        # Check domain keywords - should be unlimited
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN dk.text as domain_keyword_text
            ORDER BY dk.text
        """
        )
        domain_keywords = []
        async for record in result:
            domain_keywords.append(record["domain_keyword_text"])
        assert sorted(domain_keywords) == ["database", "graph", "neo4j"]


async def test_multi_value_dimensions_update_behavior(clean_test_database):
    """Tests that multi-value dimensions are properly updated when analysis is rerun."""
    project_id = "test-project-update"
    interview_id = "test-interview-update"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {
        "sentence_id": 0,
        "topics": ["software", "development"],
        "overall_keywords": ["test", "sentence"],
    }
    await writer.write_result(analysis_result_1)

    # Second analysis with different values
    analysis_result_2 = {
        "sentence_id": 0,
        "topics": ["testing", "automation"],  # Completely different topics
        "overall_keywords": ["test", "analysis", "neo4j"],  # Some overlap, some new
    }
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify only the latest values are present
    async with await Neo4jConnectionManager.get_session() as session:
        # Check topics - should be completely replaced
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN t.name as topic_name
            ORDER BY t.name
        """
        )
        topics = []
        async for record in result:
            topics.append(record["topic_name"])
        assert sorted(topics) == ["automation", "testing"]

        # Check keywords - should be completely replaced
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text
            ORDER BY k.text
        """
        )
        keywords = []
        async for record in result:
            keywords.append(record["keyword_text"])
        assert sorted(keywords) == ["analysis", "neo4j", "test"]


async def test_empty_dimension_values(clean_test_database):
    """Tests handling of empty or None dimension values."""
    project_id = "test-project-empty"
    interview_id = "test-interview-empty"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with empty/None values
    analysis_result = {
        "sentence_id": 0,
        "function_type": None,  # None single value
        "structure_type": "",  # Empty single value
        "purpose": "testing",  # Valid single value
        "topics": [],  # Empty list
        "overall_keywords": None,  # None list
        "domain_keywords": ["neo4j"],  # Valid list
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify only valid values created relationships
    async with await Neo4jConnectionManager.get_session() as session:
        # Check that no function type relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] is None

        # Check that no structure type relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] is None

        # Check that purpose relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_PURPOSE]->(p:Purpose)
            RETURN p.name as purpose_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["purpose_name"] == "testing"

        # Check that no topic relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 0

        # Check that no keyword relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 0

        # Check that domain keyword relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN dk.text as domain_keyword_text
        """
        )
        record = await result.single()
        assert record is not None
        assert record["domain_keyword_text"] == "neo4j"


async def test_dimension_node_properties(clean_test_database):
    """Tests that dimension nodes are created with correct properties."""
    project_id = "test-project-props"
    interview_id = "test-interview-props"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with various dimensions
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "topics": ["software"],
        "overall_keywords": ["test"],
        "domain_keywords": ["neo4j"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify node properties
    async with await Neo4jConnectionManager.get_session() as session:
        # Check FunctionType node properties
        result = await session.run(
            """
            MATCH (f:FunctionType {name: "declarative"})
            RETURN f.name as name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["name"] == "declarative"

        # Check Topic node properties
        result = await session.run(
            """
            MATCH (t:Topic {name: "software"})
            RETURN t.name as name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["name"] == "software"

        # Check Keyword node properties
        result = await session.run(
            """
            MATCH (k:Keyword {text: "test"})
            RETURN k.text as text
        """
        )
        record = await result.single()
        assert record is not None
        assert record["text"] == "test"

        # Check DomainKeyword node properties
        result = await session.run(
            """
            MATCH (dk:DomainKeyword {text: "neo4j"})
            RETURN dk.text as text, dk.is_custom as is_custom
        """
        )
        record = await result.single()
        assert record is not None
        assert record["text"] == "neo4j"
        assert record["is_custom"] is False


async def test_dimension_relationship_properties(clean_test_database):
    """Tests that dimension relationships are created with correct properties."""
    project_id = "test-project-rel-props"
    interview_id = "test-interview-rel-props"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis
    analysis_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test"]}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify relationship properties
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function relationship properties
        result = await session.run(
            """
            MATCH (a:Analysis)-[r:HAS_FUNCTION]->(f:FunctionType)
            RETURN r.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert record["is_edited"] is False

        # Check keyword relationship properties
        result = await session.run(
            """
            MATCH (a:Analysis)-[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN r.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert record["is_edited"] is False


# --- Tests for Cardinality Limits Enforcement ---


async def test_keyword_cardinality_limit_default(clean_test_database):
    """Tests that keywords respect the default cardinality limit of 6."""
    project_id = "test-project-keyword-limit"
    interview_id = "test-interview-keyword-limit"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with more keywords than the default limit (6)
    analysis_result: Dict[str, Any] = {
        "sentence_id": 0,
        "overall_keywords": [
            "keyword1",
            "keyword2",
            "keyword3",
            "keyword4",
            "keyword5",
            "keyword6",
            "keyword7",
            "keyword8",
        ],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify only 6 keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 6


async def test_topic_cardinality_unlimited(clean_test_database):
    """Tests that topics have unlimited cardinality (None limit)."""
    project_id = "test-project-topic-unlimited"
    interview_id = "test-interview-topic-unlimited"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with many topics (more than any reasonable limit)
    many_topics: List[str] = [f"topic{i}" for i in range(20)]
    analysis_result: Dict[str, Any] = {"sentence_id": 0, "topics": many_topics}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify all topics were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 20


async def test_domain_keyword_cardinality_unlimited(clean_test_database):
    """Tests that domain keywords have unlimited cardinality (None limit)."""
    project_id = "test-project-domain-unlimited"
    interview_id = "test-interview-domain-unlimited"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with many domain keywords
    many_domain_keywords = [f"domain{i}" for i in range(15)]
    analysis_result = {"sentence_id": 0, "domain_keywords": many_domain_keywords}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify all domain keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN count(dk) as domain_keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["domain_keyword_count"] == 15


async def test_single_value_cardinality_enforcement(clean_test_database):
    """Tests that single-value dimensions enforce cardinality of 1."""
    project_id = "test-project-single-cardinality"
    interview_id = "test-interview-single-cardinality"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "function_type": "declarative"}
    await writer.write_result(analysis_result_1)

    # Second analysis with different function type
    analysis_result_2 = {"sentence_id": 0, "function_type": "interrogative"}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify only one function type relationship exists
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN count(f) as function_count, f.name as function_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_count"] == 1
        assert record["function_name"] == "interrogative"  # Should be the latest value


async def test_cardinality_limit_with_duplicates(clean_test_database):
    """Tests that duplicate values don't count against cardinality limits."""
    project_id = "test-project-duplicates"
    interview_id = "test-interview-duplicates"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with duplicate keywords (should be deduplicated)
    analysis_result = {
        "sentence_id": 0,
        "overall_keywords": ["test", "test", "keyword", "keyword", "analysis", "analysis", "neo4j"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify only unique keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count, collect(k.text) as keywords
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 4  # Only unique values
        assert sorted(record["keywords"]) == ["analysis", "keyword", "neo4j", "test"]


async def test_cardinality_limit_order_preservation(clean_test_database):
    """Tests that when cardinality limits are enforced, the first N items are kept."""
    project_id = "test-project-order"
    interview_id = "test-interview-order"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with specific order of keywords
    analysis_result = {
        "sentence_id": 0,
        "overall_keywords": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify the first 6 keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN collect(k.text) as keywords
        """
        )
        record = await result.single()
        assert record is not None
        stored_keywords = record["keywords"]
        assert len(stored_keywords) == 6
        # Check that we have the first 6 keywords (order might vary due to set operations)
        expected_keywords = {"first", "second", "third", "fourth", "fifth", "sixth"}
        assert set(stored_keywords) == expected_keywords


async def test_zero_cardinality_limit(clean_test_database):
    """Tests behavior when cardinality limit is set to 0."""
    project_id = "test-project-zero-limit"
    interview_id = "test-interview-zero-limit"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create a project with zero limit for keywords
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_keywords_limit = 0
        """,
            project_id=project_id,
        )

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with keywords
    analysis_result = {"sentence_id": 0, "overall_keywords": ["test", "keyword", "analysis"]}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify no keywords were stored due to zero limit
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 0


# --- Tests for Edit Flag Protection ---


async def test_single_dimension_edit_protection(clean_test_database):
    """Tests that single-value dimensions with is_edited=true are protected from overwriting."""
    project_id = "test-project-edit-protection"
    interview_id = "test-interview-edit-protection"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "function_type": "declarative"}
    await writer.write_result(analysis_result_1)

    # Manually mark the relationship as edited
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:HAS_FUNCTION]->(f:FunctionType)
            SET r.is_edited = true
        """
        )

    # Second analysis with different function type
    analysis_result_2 = {"sentence_id": 0, "function_type": "interrogative"}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify the original function type is preserved (not overwritten)
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name, r.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "declarative"  # Original value preserved
        assert record["is_edited"] is True


async def test_multi_dimension_edit_protection(clean_test_database):
    """Tests that multi-value dimensions with is_edited=true are protected from removal."""
    project_id = "test-project-multi-edit"
    interview_id = "test-interview-multi-edit"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "overall_keywords": ["original", "keyword", "analysis"]}
    await writer.write_result(analysis_result_1)

    # Manually mark some relationships as edited
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            WHERE k.text IN ["original", "keyword"]
            SET r.is_edited = true
        """
        )

    # Second analysis with completely different keywords
    analysis_result_2 = {"sentence_id": 0, "overall_keywords": ["new", "different", "terms"]}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify edited keywords are preserved, unedited ones are replaced
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text, r.is_edited as is_edited
            ORDER BY k.text
        """
        )
        keywords = []
        async for record in result:
            keywords.append((record["keyword_text"], record["is_edited"]))

        # Should have the two edited keywords plus the new ones
        expected_keywords = [
            ("different", False),
            ("keyword", True),
            ("new", False),
            ("original", True),
            ("terms", False),
        ]
        assert sorted(keywords) == expected_keywords


async def test_mixed_edit_protection(clean_test_database):
    """Tests edit protection with a mix of edited and unedited relationships."""
    project_id = "test-project-mixed-edit"
    interview_id = "test-interview-mixed-edit"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis with multiple dimensions
    analysis_result_1 = {
        "sentence_id": 0,
        "function_type": "declarative",
        "overall_keywords": ["first", "second", "third"],
        "topics": ["topic1", "topic2"],
    }
    await writer.write_result(analysis_result_1)

    # Mark some relationships as edited
    async with await Neo4jConnectionManager.get_session() as session:
        # Mark function as edited
        await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:HAS_FUNCTION]->(f:FunctionType)
            SET r.is_edited = true
        """
        )

        # Mark one keyword as edited
        await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword {text: "second"})
            SET r.is_edited = true
        """
        )

    # Second analysis with different values
    analysis_result_2 = {
        "sentence_id": 0,
        "function_type": "interrogative",  # Should be ignored (protected)
        "overall_keywords": ["new", "keywords"],  # Should replace unedited ones, keep edited
        "topics": ["newtopic"],  # Should replace all (none were edited)
    }
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify mixed protection behavior
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function type - should be protected
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name, r.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "declarative"  # Original protected
        assert record["is_edited"] is True

        # Check keywords - should have edited one plus new ones
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text, r.is_edited as is_edited
            ORDER BY k.text
        """
        )
        keywords = []
        async for record in result:
            keywords.append((record["keyword_text"], record["is_edited"]))

        # Should have the edited "second" plus new "new" and "keywords"
        expected_keywords = [("keywords", False), ("new", False), ("second", True)]
        assert sorted(keywords) == expected_keywords

        # Check topics - should be completely replaced (none were edited)
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_TOPIC]->(t:Topic)
            RETURN t.name as topic_name, r.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert record["topic_name"] == "newtopic"
        assert record["is_edited"] is False


async def test_edit_protection_with_cardinality_limits(clean_test_database):
    """Tests that edit protection works correctly with cardinality limits."""
    project_id = "test-project-edit-cardinality"
    interview_id = "test-interview-edit-cardinality"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First analysis with keywords up to the limit (6)
    analysis_result_1 = {"sentence_id": 0, "overall_keywords": ["k1", "k2", "k3", "k4", "k5", "k6"]}
    await writer.write_result(analysis_result_1)

    # Mark some keywords as edited
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            WHERE k.text IN ["k1", "k2", "k3"]
            SET r.is_edited = true
        """
        )

    # Second analysis with new keywords (should only fit 3 new ones due to limit)
    analysis_result_2 = {
        "sentence_id": 0,
        "overall_keywords": ["new1", "new2", "new3", "new4", "new5"],  # 5 new, but only 3 slots available
    }
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # Verify cardinality limit is respected while preserving edited relationships
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as total_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["total_count"] == 6  # Should still be at the limit

        # Check that edited keywords are preserved
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            WHERE r.is_edited = true
            RETURN collect(k.text) as edited_keywords
        """
        )
        record = await result.single()
        assert record is not None
        assert sorted(record["edited_keywords"]) == ["k1", "k2", "k3"]

        # Check that we have exactly 3 new keywords (filling the remaining slots)
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            WHERE r.is_edited = false
            RETURN count(k) as new_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["new_count"] == 3  # Only 3 slots were available


# --- Tests for Error Result Handling ---


async def test_error_result_basic_storage(clean_test_database):
    """Tests that error results are properly stored in Analysis nodes without processing dimensions."""
    project_id = "test-project-error"
    interview_id = "test-interview-error"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with error result
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Create an error result
    error_result = {
        "sentence_id": 0,
        "error": True,
        "error_message": "Analysis failed due to API timeout",
        "error_code": "TIMEOUT",
        "model_name": "gpt-4",
        "function_type": "declarative",  # These should be ignored
        "overall_keywords": ["test", "keyword"],  # These should be ignored
        "topics": ["ignored", "topic"],  # These should be ignored
    }

    await writer.write_result(error_result)
    await writer.finalize()

    # Verify the error was stored correctly
    async with await Neo4jConnectionManager.get_session() as session:
        # Check that Analysis node exists with error data
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            RETURN a.error_data as error_data, a.model_name as model_name
        """
        )
        record = await result.single()
        assert record is not None
        assert json.loads(record["error_data"]) == error_result
        assert record["model_name"] == "gpt-4"

        # Verify NO dimension relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[r]->(d)
            WHERE type(r) IN ['HAS_FUNCTION', 'HAS_STRUCTURE', 'HAS_PURPOSE',
                              'MENTIONS_OVERALL_KEYWORD', 'MENTIONS_TOPIC', 'MENTIONS_DOMAIN_KEYWORD']
            RETURN count(r) as dimension_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["dimension_count"] == 0


async def test_error_result_without_dimensions(clean_test_database):
    """Tests that error results work even without dimension data in the result."""
    project_id = "test-project-error-minimal"
    interview_id = "test-interview-error-minimal"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with minimal error result
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Create a minimal error result
    error_result = {"sentence_id": 0, "error": True, "error_message": "Model unavailable"}

    await writer.write_result(error_result)
    await writer.finalize()

    # Verify the error was stored correctly
    async with await Neo4jConnectionManager.get_session() as session:
        # Check that Analysis node exists with error data
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            RETURN a.error_data as error_data, a.is_edited as is_edited
        """
        )
        record = await result.single()
        assert record is not None
        assert json.loads(record["error_data"]) == error_result
        assert record["is_edited"] is False

        # Verify NO dimension relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[r]->(d)
            WHERE type(r) IN ['HAS_FUNCTION', 'HAS_STRUCTURE', 'HAS_PURPOSE',
                              'MENTIONS_OVERALL_KEYWORD', 'MENTIONS_TOPIC', 'MENTIONS_DOMAIN_KEYWORD']
            RETURN count(r) as dimension_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["dimension_count"] == 0


async def test_error_result_after_successful_analysis(clean_test_database):
    """Tests that error results can overwrite successful analysis results."""
    project_id = "test-project-error-overwrite"
    interview_id = "test-interview-error-overwrite"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First, write a successful analysis
    successful_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "structure_type": "simple",
        "purpose": "testing",
        "overall_keywords": ["test", "keyword"],
        "topics": ["software", "testing"],
    }

    await writer.write_result(successful_result)

    # Verify successful analysis was written
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[r:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN a.error_data as error_data, count(r) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["error_data"] is None  # No error initially
        assert record["keyword_count"] == 2  # Two keywords

    # Now write an error result to the same sentence
    error_result = {
        "sentence_id": 0,
        "error": True,
        "error_message": "Reanalysis failed",
        "error_code": "REANALYSIS_FAILED",
    }

    await writer.write_result(error_result)
    await writer.finalize()

    # Verify the error overwrote the successful analysis
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            RETURN a.error_data as error_data
        """
        )
        record = await result.single()
        assert record is not None
        assert json.loads(record["error_data"]) == error_result

        # Note: The old dimension relationships might still exist because
        # error results don't actively clean up existing relationships.
        # This is expected behavior - error results just store error info
        # without processing dimensions.


async def test_error_result_read_analysis_ids(clean_test_database):
    """Tests that error results are included in read_analysis_ids."""
    project_id = "test-project-error-read"
    interview_id = "test-interview-error-read"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.write_entry({"sentence_id": 1, "sequence_order": 1, "sentence": "This is another test sentence."})
    await map_storage.finalize()

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write successful analysis for sentence 0
    successful_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test"]}
    await writer.write_result(successful_result)

    # Write error result for sentence 1
    error_result = {"sentence_id": 1, "error": True, "error_message": "Analysis failed"}
    await writer.write_result(error_result)

    await writer.finalize()

    # Verify both sentences show up in read_analysis_ids
    analysis_ids = await writer.read_analysis_ids()
    assert analysis_ids == {0, 1}


async def test_error_result_with_complex_error_data(clean_test_database):
    """Tests that complex error data structures are properly stored."""
    project_id = "test-project-error-complex"
    interview_id = "test-interview-error-complex"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with complex error data
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Create an error result with complex nested data
    error_result = {
        "sentence_id": 0,
        "error": True,
        "error_message": "Complex analysis failure",
        "error_details": {
            "api_response": {"status_code": 500, "response_body": "Internal server error"},
            "retry_attempts": 3,
            "timestamps": ["2023-01-01T10:00:00Z", "2023-01-01T10:01:00Z", "2023-01-01T10:02:00Z"],
            "model_config": {"temperature": 0.7, "max_tokens": 1000},
        },
        "partial_results": {"function_type": "unknown", "confidence": 0.1},
    }

    await writer.write_result(error_result)
    await writer.finalize()

    # Verify the complex error data was stored correctly
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            RETURN a.error_data as error_data
        """
        )
        record = await result.single()
        assert record is not None
        stored_error_data = json.loads(record["error_data"])

        # Verify the complex structure was preserved
        assert stored_error_data["error_message"] == "Complex analysis failure"
        assert stored_error_data["error_details"]["api_response"]["status_code"] == 500
        assert stored_error_data["error_details"]["retry_attempts"] == 3
        assert len(stored_error_data["error_details"]["timestamps"]) == 3
        assert stored_error_data["partial_results"]["confidence"] == 0.1


async def test_error_result_false_flag(clean_test_database):
    """Tests that results with error=False are processed normally."""
    project_id = "test-project-error-false"
    interview_id = "test-interview-error-false"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with error=False
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Create a result with explicit error=False
    normal_result = {
        "sentence_id": 0,
        "error": False,  # Explicitly false
        "function_type": "declarative",
        "overall_keywords": ["test", "keyword"],
    }

    await writer.write_result(normal_result)
    await writer.finalize()

    # Verify normal processing occurred
    async with await Neo4jConnectionManager.get_session() as session:
        # Check that Analysis node exists without error data
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            RETURN a.error_data as error_data
        """
        )
        record = await result.single()
        assert record is not None
        assert record["error_data"] is None

        # Verify dimension relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "declarative"

        # Verify keywords were processed
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 2


# --- Tests for Database Error Handling ---


async def test_database_connection_error_write_result(clean_test_database):
    """Tests that connection errors during write_result are properly handled."""
    project_id = "test-project-db-error"
    interview_id = "test-interview-db-error"

    # Set up prerequisite data first
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session.run method to raise a ServiceUnavailable error
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import ServiceUnavailable

    mock_error = ServiceUnavailable("Database service unavailable")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_error
        mock_get_session.return_value.__aenter__.return_value = mock_session

        analysis_result = {"sentence_id": 0, "function_type": "declarative"}

        # Should raise the ServiceUnavailable error
        with pytest.raises(ServiceUnavailable):
            await writer.write_result(analysis_result)


async def test_database_connection_error_read_analysis_ids(clean_test_database):
    """Tests that connection errors during read_analysis_ids are properly handled."""
    project_id = "test-project-db-error-read"
    interview_id = "test-interview-db-error-read"

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session.run method to raise a ServiceUnavailable error
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import ServiceUnavailable

    mock_error = ServiceUnavailable("Database service unavailable")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_error
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Should raise the ServiceUnavailable error
        with pytest.raises(ServiceUnavailable):
            await writer.read_analysis_ids()


async def test_invalid_cypher_query_handling(clean_test_database):
    """Tests handling of invalid Cypher queries by patching session.run."""
    project_id = "test-project-invalid-query"
    interview_id = "test-interview-invalid-query"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session.run method to raise a CypherSyntaxError
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import CypherSyntaxError

    mock_error = CypherSyntaxError("Invalid syntax", "INVALID_SYNTAX", "Invalid query")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_error
        mock_get_session.return_value.__aenter__.return_value = mock_session

        analysis_result = {"sentence_id": 0, "function_type": "declarative"}

        # Should raise the CypherSyntaxError
        with pytest.raises(CypherSyntaxError):
            await writer.write_result(analysis_result)


async def test_constraint_violation_handling(clean_test_database):
    """Tests handling of constraint violations during write operations."""
    project_id = "test-project-constraint"
    interview_id = "test-interview-constraint"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session.run method to raise a ConstraintError
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import ConstraintError

    mock_error = ConstraintError("Constraint violation", "CONSTRAINT_VIOLATION", "Duplicate key")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_error
        mock_get_session.return_value.__aenter__.return_value = mock_session

        analysis_result = {"sentence_id": 0, "function_type": "declarative"}

        # Should raise the ConstraintError
        with pytest.raises(ConstraintError):
            await writer.write_result(analysis_result)


async def test_transaction_rollback_on_error(clean_test_database):
    """Tests that transactions are properly rolled back on errors."""
    project_id = "test-project-rollback"
    interview_id = "test-interview-rollback"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session to fail partway through the transaction
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import TransientError

    call_count = 0

    def mock_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # First call (sentence lookup) succeeds
            mock_result = AsyncMock()
            mock_result.single.return_value = {"s": "mock_sentence"}
            return mock_result
        elif call_count == 2:  # Second call (analysis merge) succeeds
            mock_result = AsyncMock()
            mock_result.single.return_value = {"a": AsyncMock(id=123)}
            return mock_result
        else:  # Third call (dimension handling) fails
            raise TransientError("Database temporarily unavailable", "TRANSIENT_ERROR", "Try again")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_run
        mock_get_session.return_value.__aenter__.return_value = mock_session

        analysis_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test"]}

        # Should raise the TransientError
        with pytest.raises(TransientError):
            await writer.write_result(analysis_result)

        # Verify the transaction was attempted but failed
        assert call_count >= 3  # At least 3 calls were made before failure


async def test_read_analysis_ids_database_error_handling(clean_test_database):
    """Tests that database errors during read_analysis_ids are properly logged and re-raised."""
    project_id = "test-project-read-error"
    interview_id = "test-interview-read-error"

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session.run method to raise a database error
    from unittest.mock import AsyncMock, patch

    from neo4j.exceptions import ServiceUnavailable

    mock_error = ServiceUnavailable("Database service unavailable")

    with patch.object(Neo4jConnectionManager, "get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_session.run.side_effect = mock_error
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Should raise the ServiceUnavailable error
        with pytest.raises(ServiceUnavailable):
            await writer.read_analysis_ids()

    # Restore connection for cleanup
    # The connection will be automatically restored by the test fixture


async def test_partial_write_failure_recovery(clean_test_database):
    """Tests behavior when some dimension writes succeed but others fail."""
    project_id = "test-project-partial-failure"
    interview_id = "test-interview-partial-failure"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the session to succeed for some operations but fail for others
    from unittest.mock import patch

    from neo4j.exceptions import ClientError

    # Mock only the specific method that should fail
    original_handle_dimension_list_link = writer._handle_dimension_list_link

    async def mock_handle_dimension_list_link(session, *, relationship_type, **kwargs):
        if relationship_type == "MENTIONS_OVERALL_KEYWORD":
            raise ClientError("Property constraint violation", "CLIENT_ERROR", "Invalid property")
        else:
            return await original_handle_dimension_list_link(session, relationship_type=relationship_type, **kwargs)

    with patch.object(writer, "_handle_dimension_list_link", side_effect=mock_handle_dimension_list_link):
        analysis_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test", "keyword"]}

        # Should raise the ClientError from the keyword dimension handling
        with pytest.raises(ClientError):
            await writer.write_result(analysis_result)


async def test_session_management_error_handling(clean_test_database):
    """Tests that session management errors are handled properly."""
    project_id = "test-project-session-error"
    interview_id = "test-interview-session-error"

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the get_session method to raise an error
    from unittest.mock import patch

    from neo4j.exceptions import AuthError

    mock_error = AuthError("Authentication failed")

    with patch.object(Neo4jConnectionManager, "get_session", side_effect=mock_error):
        analysis_result = {"sentence_id": 0, "function_type": "declarative"}

        # Should raise the AuthError
        with pytest.raises(AuthError):
            await writer.write_result(analysis_result)


async def test_graceful_degradation_on_non_critical_errors(clean_test_database):
    """Tests that the writer can continue operating when non-critical operations fail."""
    project_id = "test-project-graceful"
    interview_id = "test-interview-graceful"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Create writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Mock the _fetch_project_limits method to fail (non-critical)
    from unittest.mock import patch

    async def mock_fetch_project_limits(session, project_id):
        # Simulate failure in fetching project limits, but return empty dict
        # This simulates graceful degradation where we fall back to defaults
        return {}

    with patch.object(writer, "_fetch_project_limits", side_effect=mock_fetch_project_limits):
        analysis_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test"]}

        # Should succeed despite the project limits fetch "failing"
        await writer.write_result(analysis_result)
        await writer.finalize()

        # Verify the analysis was written
        analysis_ids = await writer.read_analysis_ids()
        assert analysis_ids == {0}


# --- Tests for Project Cardinality Overrides ---


async def test_project_keyword_limit_override(clean_test_database):
    """Tests that project-specific keyword limits override default limits."""
    project_id = "test-project-keyword-override"
    interview_id = "test-interview-keyword-override"

    # Set up prerequisite data with custom project limits
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set custom project limits on the Project node
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_keywords_limit = 3
        """,
            project_id=project_id,
        )

    # Test the analysis writer with more keywords than the custom limit
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with 5 keywords (exceeds custom limit of 3)
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "overall_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify only 3 keywords were stored (custom limit enforced)
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 3  # Custom limit enforced


async def test_project_topic_limit_override(clean_test_database):
    """Tests that project-specific topic limits override default unlimited behavior."""
    project_id = "test-project-topic-override"
    interview_id = "test-interview-topic-override"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set custom project limits on the Project node (topics normally unlimited)
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_topics_limit = 2
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with 4 topics (exceeds custom limit of 2)
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "topics": ["topic1", "topic2", "topic3", "topic4"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify only 2 topics were stored (custom limit enforced)
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 2  # Custom limit enforced


async def test_project_zero_limit_override(clean_test_database):
    """Tests that project-specific zero limits prevent relationship creation."""
    project_id = "test-project-zero-override"
    interview_id = "test-interview-zero-override"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set zero limit for domain keywords
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_domain_keywords_limit = 0
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with domain keywords (should be ignored due to zero limit)
    analysis_result = {"sentence_id": 0, "function_type": "declarative", "domain_keywords": ["domain1", "domain2"]}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify no domain keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_DOMAIN_KEYWORD]->(d:DomainKeyword)
            RETURN count(d) as domain_keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["domain_keyword_count"] == 0  # Zero limit enforced


async def test_project_single_dimension_limit_override(clean_test_database):
    """Tests that project-specific limits work for single-value dimensions."""
    project_id = "test-project-single-override"
    interview_id = "test-interview-single-override"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set zero limit for function types (should prevent function relationships)
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_functions_limit = 0
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with function type (should be ignored due to zero limit)
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "structure_type": "simple",  # This should still work
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify no function relationship was created but structure was
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(f:FunctionType)
            OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
            RETURN count(f) as function_count, count(st) as structure_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["function_count"] == 0  # Zero limit enforced
        assert record["structure_count"] == 1  # Default limit still applies


async def test_project_partial_limit_overrides(clean_test_database):
    """Tests that only specified project limits override defaults, others use defaults."""
    project_id = "test-project-partial-override"
    interview_id = "test-interview-partial-override"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set only keyword limit, leave others as defaults
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_keywords_limit = 2
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with both keywords and topics
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "overall_keywords": ["k1", "k2", "k3", "k4", "k5"],  # Should be limited to 2
        "topics": ["t1", "t2", "t3", "t4", "t5"],  # Should be unlimited (default)
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify keyword limit is enforced but topics are unlimited
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 0})
                  -[:HAS_ANALYSIS]->(a:Analysis)
            RETURN
                COUNT { (a)-[:MENTIONS_OVERALL_KEYWORD]->(:Keyword) } as keyword_count,
                COUNT { (a)-[:MENTIONS_TOPIC]->(:Topic) } as topic_count
        """,
            interview_id=interview_id,
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 2  # Custom limit enforced
        assert record["topic_count"] == 5  # Default unlimited behavior


async def test_project_limits_with_null_values(clean_test_database):
    """Tests that NULL project limits fall back to defaults correctly."""
    project_id = "test-project-null-limits"
    interview_id = "test-interview-null-limits"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Explicitly set some limits to NULL
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_keywords_limit = null, p.max_topics_limit = 3
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with both keywords and topics
    analysis_result = {
        "sentence_id": 0,
        "overall_keywords": ["k1", "k2", "k3", "k4", "k5", "k6", "k7"],  # Should use default limit (6)
        "topics": ["t1", "t2", "t3", "t4", "t5"],  # Should use project limit (3)
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify default limit is used for keywords, project limit for topics
    async with await Neo4jConnectionManager.get_session() as session:
        # Check keywords count
        result = await session.run(
            """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 0})
                  -[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """,
            interview_id=interview_id,
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 6  # Default limit applied

        # Check topics count
        result = await session.run(
            """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 0})
                  -[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
        """,
            interview_id=interview_id,
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 3  # Project limit applied


async def test_project_limits_all_dimensions(clean_test_database):
    """Tests that project limits work for all dimension types."""
    project_id = "test-project-all-limits"
    interview_id = "test-interview-all-limits"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set custom limits for all dimension types
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_functions_limit = 1,
                p.max_structures_limit = 1,
                p.max_purposes_limit = 1,
                p.max_keywords_limit = 2,
                p.max_topics_limit = 2,
                p.max_domain_keywords_limit = 2
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # Write analysis with all dimension types
    analysis_result = {
        "sentence_id": 0,
        "function_type": "declarative",
        "structure_type": "simple",
        "purpose": "testing",
        "overall_keywords": ["k1", "k2", "k3", "k4"],
        "topics": ["t1", "t2", "t3", "t4"],
        "domain_keywords": ["d1", "d2", "d3", "d4"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # Verify all limits are enforced
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 0})
                    -[:HAS_ANALYSIS]->(a:Analysis)
            RETURN
                COUNT { (a)-[:HAS_FUNCTION]->(:FunctionType) } as function_count,
                COUNT { (a)-[:HAS_STRUCTURE]->(:StructureType) } as structure_count,
                COUNT { (a)-[:HAS_PURPOSE]->(:Purpose) } as purpose_count,
                COUNT { (a)-[:MENTIONS_OVERALL_KEYWORD]->(:Keyword) } as keyword_count,
                COUNT { (a)-[:MENTIONS_TOPIC]->(:Topic) } as topic_count,
                COUNT { (a)-[:MENTIONS_DOMAIN_KEYWORD]->(:DomainKeyword) } as domain_keyword_count
        """,
            interview_id=interview_id,
        )
        record = await result.single()
        assert record is not None
        assert record["function_count"] == 1  # Single value dimension
        assert record["structure_count"] == 1  # Single value dimension
        assert record["purpose_count"] == 1  # Single value dimension
        assert record["keyword_count"] == 2  # Custom limit enforced
        assert record["topic_count"] == 2  # Custom limit enforced
        assert record["domain_keyword_count"] == 2  # Custom limit enforced


async def test_project_limits_with_existing_relationships(clean_test_database):
    """Tests that project limits work correctly when updating existing analyses."""
    project_id = "test-project-existing-limits"
    interview_id = "test-interview-existing-limits"

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Set custom keyword limit
    async with await Neo4jConnectionManager.get_session() as session:
        await session.run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.max_keywords_limit = 3
        """,
            project_id=project_id,
        )

    # Test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id)
    await writer.initialize()

    # First write with 2 keywords
    analysis_result1 = {"sentence_id": 0, "overall_keywords": ["keyword1", "keyword2"]}
    await writer.write_result(analysis_result1)

    # Second write with 4 keywords (should be limited to 3 total)
    analysis_result2 = {"sentence_id": 0, "overall_keywords": ["keyword3", "keyword4", "keyword5", "keyword6"]}
    await writer.write_result(analysis_result2)
    await writer.finalize()

    # Verify only 3 keywords total (limit enforced on update)
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: 0})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
        """
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 3  # Custom limit enforced
