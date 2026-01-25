"""
Data integrity tests for Neo4j integration via projection handlers.

These tests validate data consistency, transaction integrity, and relationship
correctness when events are processed through projection handlers:
- Transaction atomicity via projection handlers
- Relationship integrity and constraint validation
- Data consistency across projection operations
- Graph structure validation and orphaned node detection
- Idempotency and duplicate event handling

**M3.0 Architecture Context:**
- OLD (M2.x): Pipeline -> Neo4j (direct write, immediate consistency)
- NEW (M3.0): Pipeline -> EventStoreDB -> Projection Service -> Neo4j

These tests verify that projection handlers correctly create Neo4j state
from events. Direct writes (Neo4jAnalysisWriter, Neo4jMapStorage) are deprecated.

**Why this matters after context compaction:**
The dual-write pattern was REMOVED in M3.0. The projection service is the SOLE
writer to Neo4j. All tests must use events + projection handlers, not direct writes.
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.events.envelope import ActorType, AggregateType, Actor, EventEnvelope
from src.projections.handlers.interview_handlers import InterviewCreatedHandler
from src.projections.handlers.sentence_handlers import (
    AnalysisGeneratedHandler,
    SentenceCreatedHandler,
)
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.integration,
]


def create_interview_created_event(interview_id: str, project_id: str = "test-project") -> EventEnvelope:
    """Helper to create InterviewCreated event."""
    return EventEnvelope(
        event_id=str(uuid.uuid4()),
        event_type="InterviewCreated",
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=interview_id,
        version=1,
        occurred_at=datetime.now(timezone.utc),
        data={
            "project_id": project_id,
            "title": "Test Interview",
            "source": "test_file.txt",
            "language": "en",
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        actor=Actor(actor_type=ActorType.SYSTEM, display="test"),
    )


def create_sentence_created_event(
    sentence_id: str,
    interview_id: str,
    index: int,
    text: str,
    speaker: str = None,
    version: int = 1,
) -> EventEnvelope:
    """Helper to create SentenceCreated event."""
    return EventEnvelope(
        event_id=str(uuid.uuid4()),
        event_type="SentenceCreated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=sentence_id,
        version=version,
        occurred_at=datetime.now(timezone.utc),
        data={
            "interview_id": interview_id,
            "index": index,
            "text": text,
            "speaker": speaker,
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        actor=Actor(actor_type=ActorType.SYSTEM, display="test"),
    )


def create_analysis_generated_event(
    sentence_id: str,
    function_type: str = "declarative",
    structure_type: str = "simple",
    purpose: str = "testing",
    keywords: list = None,
    topics: list = None,
    domain_keywords: list = None,
    version: int = 2,  # Default to 2 since it follows SentenceCreated (version 1)
) -> EventEnvelope:
    """Helper to create AnalysisGenerated event."""
    return EventEnvelope(
        event_id=str(uuid.uuid4()),
        event_type="AnalysisGenerated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=sentence_id,
        version=version,
        occurred_at=datetime.now(timezone.utc),
        data={
            "model": "test-model",
            "model_version": "1.0",
            "confidence": 0.95,
            "classification": {
                "function_type": function_type,
                "structure_type": structure_type,
                "purpose": purpose,
            },
            "keywords": keywords or [],
            "topics": topics or [],
            "domain_keywords": domain_keywords or [],
        },
        actor=Actor(actor_type=ActorType.SYSTEM, display="test"),
    )


@pytest.mark.neo4j
@pytest.mark.integration
class TestProjectionTransactionIntegrity:
    """Test transaction atomicity via projection handlers."""

    @pytest.mark.asyncio
    async def test_projection_creates_complete_graph_structure(self, clean_test_database):
        """Test that projection handlers create complete graph structure atomically."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        # Create events
        interview_event = create_interview_created_event(interview_id, project_id)
        sentence_event = create_sentence_created_event(
            sentence_id=sentence_id,
            interview_id=interview_id,
            index=0,
            text="Transaction atomicity test sentence.",
            speaker="test_speaker",
        )
        analysis_event = create_analysis_generated_event(
            sentence_id=sentence_id,
            function_type="declarative",
            structure_type="complex",
            purpose="integrity_testing",
            keywords=["transaction", "atomicity", "test"],
            topics=["data_integrity"],
            domain_keywords=["integrity", "testing"],
        )

        # Process events through handlers
        interview_handler = InterviewCreatedHandler()
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        await interview_handler.handle(interview_event)
        await sentence_handler.handle(sentence_event)
        await analysis_handler.handle(analysis_event)

        # Verify complete graph structure was created
        async with await Neo4jConnectionManager.get_session() as session:
            # Check Project -> Interview -> Sentence chain
            result = await session.run(
                """
                MATCH (p:Project {project_id: $project_id})
                -[:CONTAINS_INTERVIEW]->(i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                RETURN p, i, s
                """,
                project_id=project_id,
                interview_id=interview_id,
                sentence_id=sentence_id,
            )
            record = await result.single()
            assert record is not None, "Project -> Interview -> Sentence chain not created"

            # Check Sentence -> Analysis with all relationships
            result = await session.run(
                """
                MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(ft:FunctionType)
                OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
                OPTIONAL MATCH (a)-[:HAS_PURPOSE]->(p:Purpose)
                OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (a)-[:MENTIONS_TOPIC]->(t:Topic)
                OPTIONAL MATCH (a)-[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
                RETURN
                    a.analysis_id as analysis_id,
                    ft.name as function_type,
                    st.name as structure_type,
                    p.name as purpose,
                    count(DISTINCT k) as keyword_count,
                    count(DISTINCT t) as topic_count,
                    count(DISTINCT dk) as domain_keyword_count
                """,
                sentence_id=sentence_id,
            )
            record = await result.single()
            assert record is not None, "Analysis not created"
            assert record["function_type"] == "declarative"
            assert record["structure_type"] == "complex"
            assert record["purpose"] == "integrity_testing"
            assert record["keyword_count"] == 3
            assert record["topic_count"] == 1
            assert record["domain_keyword_count"] == 2

    @pytest.mark.asyncio
    async def test_projection_idempotency_same_event_twice(self, clean_test_database):
        """Test that processing the same event twice doesn't create duplicates."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create event
        interview_event = create_interview_created_event(interview_id, project_id)

        # Process the same event twice
        handler = InterviewCreatedHandler()
        await handler.handle(interview_event)
        await handler.handle(interview_event)

        # Verify only one Interview node exists
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN count(i) as count",
                interview_id=interview_id,
            )
            count = await result.single()
            assert count["count"] == 1, "Duplicate Interview nodes created"


@pytest.mark.neo4j
@pytest.mark.integration
class TestProjectionRelationshipIntegrity:
    """Test relationship consistency via projection handlers."""

    @pytest.mark.asyncio
    async def test_relationship_consistency_multiple_sentences(self, clean_test_database):
        """Test that relationships maintain integrity across multiple sentences."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create interview
        interview_event = create_interview_created_event(interview_id, project_id)
        interview_handler = InterviewCreatedHandler()
        await interview_handler.handle(interview_event)

        # Create multiple sentences with analysis
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        sentence_ids = []
        for i in range(5):
            sentence_id = str(uuid.uuid4())
            sentence_ids.append(sentence_id)

            sentence_event = create_sentence_created_event(
                sentence_id=sentence_id,
                interview_id=interview_id,
                index=i,
                text=f"Relationship integrity test sentence {i}",
            )
            await sentence_handler.handle(sentence_event)

            analysis_event = create_analysis_generated_event(
                sentence_id=sentence_id,
                function_type="declarative",
                structure_type="simple",
                purpose="relationship_testing",
            )
            await analysis_handler.handle(analysis_event)

        # Verify relationship integrity
        async with await Neo4jConnectionManager.get_session() as session:
            # Check all sentences linked to interview
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(s) as sentence_count
                """,
                interview_id=interview_id,
            )
            sentence_count = await result.single()
            assert sentence_count["sentence_count"] == 5

            # Check all sentences have analysis
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                -[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            analysis_count = await result.single()
            assert analysis_count["analysis_count"] == 5

            # Verify no orphaned analysis nodes
            result = await session.run(
                """
                MATCH (a:Analysis)
                WHERE NOT (a)<-[:HAS_ANALYSIS]-(:Sentence)
                RETURN count(a) as orphaned_count
                """
            )
            orphaned = await result.single()
            assert orphaned["orphaned_count"] == 0, "Orphaned Analysis nodes found"

    @pytest.mark.asyncio
    async def test_shared_dimension_nodes_reused(self, clean_test_database):
        """Test that shared dimension nodes (keywords, topics) are reused, not duplicated."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create interview
        interview_event = create_interview_created_event(interview_id, project_id)
        await InterviewCreatedHandler().handle(interview_event)

        # Create sentences that share keywords
        shared_keywords = ["shared", "integrity", "cascade"]
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        for i in range(3):
            sentence_id = str(uuid.uuid4())

            sentence_event = create_sentence_created_event(
                sentence_id=sentence_id,
                interview_id=interview_id,
                index=i,
                text=f"Cascade integrity test sentence {i}",
            )
            await sentence_handler.handle(sentence_event)

            # All sentences share the same keywords
            analysis_event = create_analysis_generated_event(
                sentence_id=sentence_id,
                keywords=shared_keywords + [f"unique_{i}"],
                domain_keywords=["cascade", "testing"],
            )
            await analysis_handler.handle(analysis_event)

        # Verify shared keywords are reused (not duplicated)
        async with await Neo4jConnectionManager.get_session() as session:
            for keyword in shared_keywords:
                result = await session.run(
                    "MATCH (k:Keyword {text: $keyword}) RETURN count(k) as count",
                    keyword=keyword,
                )
                count = await result.single()
                assert count["count"] == 1, f"Keyword '{keyword}' duplicated"

                # Verify multiple analyses link to same keyword
                result = await session.run(
                    """
                    MATCH (k:Keyword {text: $keyword})<-[:MENTIONS_OVERALL_KEYWORD]-(a:Analysis)
                    RETURN count(a) as analysis_count
                    """,
                    keyword=keyword,
                )
                analysis_count = await result.single()
                assert analysis_count["analysis_count"] == 3, f"Keyword '{keyword}' not linked to all 3 analyses"


@pytest.mark.neo4j
@pytest.mark.integration
class TestProjectionDataConsistency:
    """Test data consistency via projection handlers."""

    @pytest.mark.asyncio
    async def test_analysis_update_replaces_old_analysis(self, clean_test_database):
        """Test that a new AnalysisGenerated event replaces the old analysis."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        # Setup
        await InterviewCreatedHandler().handle(
            create_interview_created_event(interview_id, project_id)
        )
        await SentenceCreatedHandler().handle(
            create_sentence_created_event(sentence_id, interview_id, 0, "Original sentence")
        )

        # First analysis (version 2 follows SentenceCreated version 1)
        await AnalysisGeneratedHandler().handle(
            create_analysis_generated_event(
                sentence_id=sentence_id,
                function_type="declarative",
                purpose="original_purpose",
                keywords=["original", "test"],
                version=2,
            )
        )

        # Second analysis (version 3, should replace first)
        await AnalysisGeneratedHandler().handle(
            create_analysis_generated_event(
                sentence_id=sentence_id,
                function_type="interrogative",
                purpose="updated_purpose",
                keywords=["updated", "test", "modified"],
                version=3,
            )
        )

        # Verify only one analysis exists with updated values
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                """
                MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(ft:FunctionType)
                OPTIONAL MATCH (a)-[:HAS_PURPOSE]->(p:Purpose)
                OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN
                    count(DISTINCT a) as analysis_count,
                    ft.name as function_type,
                    p.name as purpose,
                    collect(DISTINCT k.text) as keywords
                """,
                sentence_id=sentence_id,
            )
            record = await result.single()

            assert record["analysis_count"] == 1, "Multiple Analysis nodes found"
            assert record["function_type"] == "interrogative", "Function type not updated"
            assert record["purpose"] == "updated_purpose", "Purpose not updated"
            assert "updated" in record["keywords"], "Keywords not updated"
            assert "modified" in record["keywords"], "Keywords not updated"

    @pytest.mark.asyncio
    async def test_no_orphaned_nodes_after_projection(self, clean_test_database):
        """Test that projection handlers don't leave orphaned nodes."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        # Create full structure
        await InterviewCreatedHandler().handle(
            create_interview_created_event(interview_id, project_id)
        )
        await SentenceCreatedHandler().handle(
            create_sentence_created_event(sentence_id, interview_id, 0, "Orphan test sentence")
        )
        await AnalysisGeneratedHandler().handle(
            create_analysis_generated_event(
                sentence_id=sentence_id,
                keywords=["orphan", "detection", "test"],
                topics=["data_integrity"],
                domain_keywords=["testing"],
            )
        )

        # Check for orphaned nodes
        async with await Neo4jConnectionManager.get_session() as session:
            # Analysis nodes should be linked to Sentence
            result = await session.run(
                """
                MATCH (a:Analysis)
                WHERE NOT (a)<-[:HAS_ANALYSIS]-(:Sentence)
                RETURN count(a) as orphaned_analysis
                """
            )
            orphaned_analysis = await result.single()
            assert orphaned_analysis["orphaned_analysis"] == 0, "Orphaned Analysis nodes found"

            # Sentence nodes should be linked to Interview
            result = await session.run(
                """
                MATCH (s:Sentence)
                WHERE NOT (s)<-[:HAS_SENTENCE]-(:Interview)
                RETURN count(s) as orphaned_sentences
                """
            )
            orphaned_sentences = await result.single()
            assert orphaned_sentences["orphaned_sentences"] == 0, "Orphaned Sentence nodes found"


@pytest.mark.neo4j
@pytest.mark.integration
class TestProjectionDataValidation:
    """Test data type and constraint validation via projection handlers."""

    @pytest.mark.asyncio
    async def test_data_types_preserved_through_projection(self, clean_test_database):
        """Test that data types are correctly preserved through projection."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid4())

        # Create structure
        await InterviewCreatedHandler().handle(
            create_interview_created_event(interview_id, project_id)
        )
        await SentenceCreatedHandler().handle(
            create_sentence_created_event(
                sentence_id=sentence_id,
                interview_id=interview_id,
                index=42,
                text="Data type consistency test",
                speaker="test_speaker",
            )
        )
        await AnalysisGeneratedHandler().handle(
            create_analysis_generated_event(
                sentence_id=sentence_id,
                keywords=["type", "consistency"],
            )
        )

        # Verify data types
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                """
                MATCH (s:Sentence {sentence_id: $sentence_id})
                RETURN
                    s.sequence_order as sequence_order,
                    s.text as text,
                    s.speaker as speaker
                """,
                sentence_id=sentence_id,
            )
            record = await result.single()

            # Verify types
            assert isinstance(record["sequence_order"], int), "sequence_order should be int"
            assert record["sequence_order"] == 42
            assert isinstance(record["text"], str), "text should be string"
            assert record["speaker"] == "test_speaker"

    @pytest.mark.asyncio
    async def test_unique_sentence_ids_enforced(self, clean_test_database):
        """Test that unique sentence IDs are maintained."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        await InterviewCreatedHandler().handle(
            create_interview_created_event(interview_id, project_id)
        )

        # Create multiple sentences with different IDs
        sentence_ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, sentence_id in enumerate(sentence_ids):
            await SentenceCreatedHandler().handle(
                create_sentence_created_event(sentence_id, interview_id, i, f"Sentence {i}")
            )
            await AnalysisGeneratedHandler().handle(
                create_analysis_generated_event(sentence_id, purpose=f"testing_{i}")
            )

        # Verify uniqueness
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(DISTINCT s.sentence_id) as unique_count, count(s) as total_count
                """,
                interview_id=interview_id,
            )
            counts = await result.single()
            assert counts["unique_count"] == counts["total_count"], "Duplicate sentence IDs found"
            assert counts["unique_count"] == 3

            # Each sentence should have exactly one analysis
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                -[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id, count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            async for record in result:
                assert record["analysis_count"] == 1, f"Sentence {record['sentence_id']} has {record['analysis_count']} analyses"
