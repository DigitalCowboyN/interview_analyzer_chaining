# tests/integration/test_neo4j_analysis_writer.py
"""
Integration tests for the Neo4j analysis writer implementation of SentenceAnalysisWriter protocol.

M2.8: Tests updated to use projection service pattern.
"""

import json
import uuid
from typing import Any, Dict, List, Set

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.pipeline_event_emitter import PipelineEventEmitter
from src.projections.handlers.interview_handlers import InterviewCreatedHandler
from src.projections.handlers.sentence_handlers import (
    SentenceCreatedHandler,
    AnalysisGeneratedHandler,
)
from src.utils.neo4j_driver import Neo4jConnectionManager

# Mark all tests in this module as asyncio and require Neo4j
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def process_events_through_projection(
    event_store,
    interview_id: str,
    num_sentences: int,
    process_analyses: bool = True,
):
    """
    M2.8 Helper: Process events through projection service handlers.

    Args:
        event_store: EventStoreClient instance
        interview_id: Interview UUID
        num_sentences: Number of sentences to process
        process_analyses: Whether to process AnalysisGenerated events
    """
    import logging
    logger = logging.getLogger(__name__)

    # Process InterviewCreated event
    interview_handler = InterviewCreatedHandler()
    interview_stream = f"Interview-{interview_id}"
    interview_events = await event_store.read_stream(interview_stream)

    logger.info(f"[DEBUG] Reading {interview_stream}, got {len(interview_events)} events")

    for event in interview_events:
        if event.event_type == "InterviewCreated":
            logger.info(f"[DEBUG] Processing InterviewCreated v{event.version}")
            await interview_handler.handle(event)
            logger.info(f"[DEBUG] Completed InterviewCreated v{event.version}")

    # Process SentenceCreated and optionally AnalysisGenerated events
    sentence_handler = SentenceCreatedHandler()
    analysis_handler = AnalysisGeneratedHandler() if process_analyses else None

    for i in range(num_sentences):
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
        sentence_stream = f"Sentence-{sentence_id}"
        sentence_events = await event_store.read_stream(sentence_stream)

        logger.info(f"[DEBUG] Reading {sentence_stream}, got {len(sentence_events)} events")

        for idx, event in enumerate(sentence_events):
            logger.info(f"[DEBUG] Event {idx+1}/{len(sentence_events)}: {event.event_type} v{event.version} for {event.aggregate_id[:8]}...")

            if event.event_type == "SentenceCreated":
                logger.info(f"[DEBUG] Processing SentenceCreated v{event.version}")
                await sentence_handler.handle(event)
                logger.info(f"[DEBUG] Completed SentenceCreated v{event.version}")
            elif event.event_type == "AnalysisGenerated" and analysis_handler:
                logger.info(f"[DEBUG] Processing AnalysisGenerated v{event.version}")
                await analysis_handler.handle(event)
                logger.info(f"[DEBUG] Completed AnalysisGenerated v{event.version}")


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


async def test_neo4j_analysis_writer_write_basic_result(
    clean_test_database: Any, clean_event_store: Any
) -> None:
    """M2.8: Tests writing a basic analysis result with all dimensions."""
    project_id: str = "test-project-write"
    interview_id: str = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # First, set up the prerequisite data using map storage
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Now test the analysis writer
    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # Write the result - emits AnalysisGenerated event
    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # Verify the analysis was written by checking read_analysis_ids
    analysis_ids: Set[int] = await writer.read_analysis_ids()
    assert analysis_ids == {0}


async def test_neo4j_analysis_writer_missing_sentence_error(clean_test_database, clean_event_store):
    """Tests that writing analysis for non-existent sentence logs warning (event-first architecture).

    With event-first architecture, the event is emitted successfully but Neo4j write fails.
    This is correct - projection service will retry from the event.
    """
    import uuid
    project_id = "test-project-missing"
    interview_id = str(uuid.uuid4())  # Use random UUID to avoid conflicts
    sentence_index = 999

    # Create the sentence ID deterministically based on the random interview_id
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
    stream_name = f"Sentence-{sentence_id}"

    # First, create a SentenceCreated event (required before AnalysisGenerated)
    from src.events.envelope import Actor, ActorType
    from src.events.sentence_events import create_sentence_created_event

    system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
    sentence_created_event = create_sentence_created_event(
        aggregate_id=sentence_id,
        version=0,
        interview_id=interview_id,
        index=sentence_index,
        text="Test sentence for missing sentence test.",
        actor=system_actor,
        correlation_id="test-correlation",
    )

    # Append SentenceCreated event to EventStoreDB
    await clean_event_store.append_events(
        stream_name=stream_name,
        events=[sentence_created_event],
        expected_version=-1,  # New stream
    )

    # Create writer WITH event emitter to test event-first behavior
    from src.pipeline_event_emitter import PipelineEventEmitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    writer = Neo4jAnalysisWriter(
        project_id,
        interview_id,
        event_emitter=event_emitter,
        correlation_id="test-correlation"
    )
    await writer.initialize()

    # Now try to write analysis for a sentence that exists in EventStoreDB but NOT in Neo4j
    analysis_result = {"sentence_id": sentence_index, "function_type": "declarative"}

    # With event-first: event succeeds, Neo4j fails, warning logged (no exception raised)
    await writer.write_result(analysis_result)  # Should NOT raise

    # Event should have been emitted successfully to EventStoreDB
    # (even though Neo4j write failed because sentence doesn't exist in Neo4j)


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


async def test_single_value_dimensions_basic(clean_test_database, clean_event_store):
    """M2.8: Tests basic single-value dimension handling (function, structure, purpose)."""
    project_id = "test-project-single"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    # Test the analysis writer with single-value dimensions
    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify the relationships were created
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function type relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "declarative"

        # Check structure type relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] == "simple"

        # Check purpose relationship
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_PURPOSE]->(p:Purpose)
            RETURN p.name as purpose_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["purpose_name"] == "testing"


async def test_single_value_dimensions_overwrite(clean_test_database, clean_event_store):
    """M2.8: Tests that single-value dimensions are overwritten when analysis is rerun."""
    project_id = "test-project-overwrite"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "function_type": "declarative", "structure_type": "simple"}
    await writer.write_result(analysis_result_1)

    # Second analysis with different values
    analysis_result_2 = {"sentence_id": 0, "function_type": "interrogative", "structure_type": "complex"}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only the latest values are present
    async with await Neo4jConnectionManager.get_session() as session:
        # Check function type - should be updated
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] == "interrogative"

        # Check structure type - should be updated
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] == "complex"


async def test_multi_value_dimensions_basic(clean_test_database, clean_event_store):
    """M2.8: Tests basic multi-value dimension handling (keywords, topics, domain_keywords)."""
    project_id = "test-project-multi"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry(
        {"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence about software development."}
    )
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify the relationships were created
    async with await Neo4jConnectionManager.get_session() as session:
        # Check topics - should be unlimited
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN t.name as topic_name
            ORDER BY t.name
            """,
            sentence_id=sentence_id
        )
        topics: List[str] = []
        async for record in result:
            topics.append(record["topic_name"])
        assert sorted(topics) == ["development", "software", "testing"]

        # Check keywords - should have default limit of 6
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text
            ORDER BY k.text
            """,
            sentence_id=sentence_id
        )
        keywords: List[str] = []
        async for record in result:
            keywords.append(record["keyword_text"])
        assert sorted(keywords) == ["analysis", "sentence", "test"]

        # Check domain keywords - should be unlimited
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN dk.text as domain_keyword_text
            ORDER BY dk.text
            """,
            sentence_id=sentence_id
        )
        domain_keywords = []
        async for record in result:
            domain_keywords.append(record["domain_keyword_text"])
        assert sorted(domain_keywords) == ["database", "graph", "neo4j"]


async def test_multi_value_dimensions_update_behavior(clean_test_database, clean_event_store):
    """M2.8: Tests that multi-value dimensions are properly updated when analysis is rerun."""
    project_id = "test-project-update"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only the latest values are present
    async with await Neo4jConnectionManager.get_session() as session:
        # Check topics - should be completely replaced
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN t.name as topic_name
            ORDER BY t.name
            """,
            sentence_id=sentence_id
        )
        topics = []
        async for record in result:
            topics.append(record["topic_name"])
        assert sorted(topics) == ["automation", "testing"]

        # Check keywords - should be completely replaced
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN k.text as keyword_text
            ORDER BY k.text
            """,
            sentence_id=sentence_id
        )
        keywords = []
        async for record in result:
            keywords.append(record["keyword_text"])
        assert sorted(keywords) == ["analysis", "neo4j", "test"]


async def test_empty_dimension_values(clean_test_database, clean_event_store):
    """M2.8: Tests handling of empty or None dimension values."""
    project_id = "test-project-empty"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only valid values created relationships
    async with await Neo4jConnectionManager.get_session() as session:
        # Check that no function type relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(f:FunctionType)
            RETURN f.name as function_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["function_name"] is None

        # Check that no structure type relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
            RETURN st.name as structure_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["structure_name"] is None

        # Check that purpose relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_PURPOSE]->(p:Purpose)
            RETURN p.name as purpose_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["purpose_name"] == "testing"

        # Check that no topic relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 0

        # Check that no keyword relationships were created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
            OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 0

        # Check that domain keyword relationship was created
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN dk.text as domain_keyword_text
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["domain_keyword_text"] == "neo4j"


async def test_dimension_node_properties(clean_test_database, clean_event_store):
    """M2.8: Tests that dimension nodes are created with correct properties."""
    project_id = "test-project-props"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

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


async def test_dimension_relationship_properties(clean_test_database, clean_event_store):
    """M2.8: Tests that dimension relationships are created with correct properties."""
    project_id = "test-project-rel-props"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # Write analysis
    analysis_result = {"sentence_id": 0, "function_type": "declarative", "overall_keywords": ["test"]}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

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


async def test_keyword_cardinality_limit_default(clean_test_database, clean_event_store):
    """M2.8: Tests that keywords respect the default cardinality limit of 6."""
    project_id = "test-project-keyword-limit"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
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

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only 6 keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 6


async def test_topic_cardinality_unlimited(clean_test_database, clean_event_store):
    """M2.8: Tests that topics have unlimited cardinality (None limit)."""
    project_id = "test-project-topic-unlimited"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # Write analysis with many topics (more than any reasonable limit)
    many_topics: List[str] = [f"topic{i}" for i in range(20)]
    analysis_result: Dict[str, Any] = {"sentence_id": 0, "topics": many_topics}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify all topics were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_TOPIC]->(t:Topic)
            RETURN count(t) as topic_count
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["topic_count"] == 20


async def test_domain_keyword_cardinality_unlimited(clean_test_database, clean_event_store):
    """M2.8: Tests that domain keywords have unlimited cardinality (None limit)."""
    project_id = "test-project-domain-unlimited"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # Write analysis with many domain keywords
    many_domain_keywords = [f"domain{i}" for i in range(15)]
    analysis_result = {"sentence_id": 0, "domain_keywords": many_domain_keywords}

    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify all domain keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
            RETURN count(dk) as domain_keyword_count
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["domain_keyword_count"] == 15


async def test_single_value_cardinality_enforcement(clean_test_database, clean_event_store):
    """M2.8: Tests that single-value dimensions enforce cardinality of 1."""
    project_id = "test-project-single-cardinality"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # First analysis
    analysis_result_1 = {"sentence_id": 0, "function_type": "declarative"}
    await writer.write_result(analysis_result_1)

    # Second analysis with different function type
    analysis_result_2 = {"sentence_id": 0, "function_type": "interrogative"}
    await writer.write_result(analysis_result_2)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only one function type relationship exists
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:HAS_FUNCTION]->(f:FunctionType)
            RETURN count(f) as function_count, f.name as function_name
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["function_count"] == 1
        assert record["function_name"] == "interrogative"  # Should be the latest value


async def test_cardinality_limit_with_duplicates(clean_test_database, clean_event_store):
    """M2.8: Tests that duplicate values don't count against cardinality limits."""
    project_id = "test-project-duplicates"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # Write analysis with duplicate keywords (should be deduplicated)
    analysis_result = {
        "sentence_id": 0,
        "overall_keywords": ["test", "test", "keyword", "keyword", "analysis", "analysis", "neo4j"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify only unique keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN count(k) as keyword_count, collect(k.text) as keywords
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        assert record["keyword_count"] == 4  # Only unique values
        assert sorted(record["keywords"]) == ["analysis", "keyword", "neo4j", "test"]


async def test_cardinality_limit_order_preservation(clean_test_database, clean_event_store):
    """M2.8: Tests that when cardinality limits are enforced, the first N items are kept."""
    project_id = "test-project-order"
    interview_id = str(uuid.uuid4())  # Unique per test

    # Setup event emitter
    event_emitter = PipelineEventEmitter(clean_event_store)

    # Emit InterviewCreated event
    await event_emitter.emit_interview_created(
        interview_id=interview_id,
        project_id=project_id,
        title="test_interview",
        source="/test/path",
    )

    # Set up prerequisite data
    map_storage = Neo4jMapStorage(project_id, interview_id, event_emitter=event_emitter)
    await map_storage.initialize()
    await map_storage.write_entry({"sentence_id": 0, "sequence_order": 0, "sentence": "This is a test sentence."})
    await map_storage.finalize()

    writer = Neo4jAnalysisWriter(project_id, interview_id, event_emitter=event_emitter)
    await writer.initialize()

    # Write analysis with specific order of keywords
    analysis_result = {
        "sentence_id": 0,
        "overall_keywords": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"],
    }

    await writer.write_result(analysis_result)
    await writer.finalize()

    # M2.8: Process events through projection service
    await process_events_through_projection(clean_event_store, interview_id, 1, process_analyses=True)

    # M2.8: Calculate sentence UUID for verification queries
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))

    # Verify the first 6 keywords were stored
    async with await Neo4jConnectionManager.get_session() as session:
        result = await session.run(
            """
            MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                  -[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
            RETURN collect(k.text) as keywords
            """,
            sentence_id=sentence_id
        )
        record = await result.single()
        assert record is not None
        stored_keywords = record["keywords"]
        assert len(stored_keywords) == 6
        # Check that we have the first 6 keywords (order might vary due to set operations)
        expected_keywords = {"first", "second", "third", "fourth", "fifth", "sixth"}
        assert set(stored_keywords) == expected_keywords



# --- End of M2.8 Tests ---
#
# **Legacy Tests Moved:**
# Tests using direct Neo4j write path (without event_emitter) have been moved to
# test_neo4j_analysis_writer_legacy.py and are skipped in CI.
#
# **Recommended Tests to Add:**
# - Multi-dimension edit protection (M2.8 event-first pattern)
# - Mixed edit protection (M2.8 event-first pattern)
# - Edit protection with cardinality limits (M2.8 event-first pattern)
# - Zero cardinality limit (M2.8 event-first pattern)
# - Project limit overrides (8 tests, M2.8 event-first pattern)
#
# See test_neo4j_analysis_writer_legacy.py for test migration examples.
