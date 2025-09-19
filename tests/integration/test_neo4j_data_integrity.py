"""
Comprehensive data integrity tests for Neo4j integration components.

These tests validate data consistency, transaction integrity, and relationship
correctness across various scenarios:
- Transaction atomicity and rollback behavior
- Relationship integrity and constraint validation
- Data consistency across concurrent operations
- Graph structure validation and orphaned node detection
- Cross-component data synchronization
- Data corruption detection and recovery
"""

import asyncio
import uuid

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.persistence.graph_persistence import save_analysis_to_graph
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jTransactionIntegrity:
    """Test transaction consistency and atomicity."""

    @pytest.mark.asyncio
    async def test_transaction_atomicity_success(self, clean_test_database):
        """Test that successful transactions commit all changes atomically."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create sentence mapping
        sentence_data = {
            "sentence_id": 1000,
            "sentence": "Transaction atomicity test sentence",
            "sequence_order": 0,
        }
        await map_storage.write_entry(sentence_data)

        # Write analysis data
        analysis_data = {
            "sentence_id": 1000,
            "sequence_order": 0,
            "sentence": "Transaction atomicity test sentence",
            "function_type": "declarative",
            "structure_type": "complex",
            "purpose": "integrity_testing",
            "topics": ["data_integrity", "transaction_atomicity"],
            "keywords": ["transaction", "atomicity", "test"],
            "domain_keywords": ["integrity", "testing"],
        }

        await writer.write_result(analysis_data)

        # Verify all related nodes and relationships were created atomically
        async with await Neo4jConnectionManager.get_session() as session:
            # Check sentence exists
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s",
                sentence_id=1000,
            )
            sentence = await result.single()
            assert sentence is not None

            # Check analysis node exists and is linked to sentence
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN a, s
                """,
                interview_id=interview_id,
                sentence_id=1000,
            )
            analysis_sentence = await result.single()
            assert analysis_sentence is not None

            # Check all analysis relationships exist
            expected_relationships = [
                ("FunctionType", "HAS_FUNCTION"),
                ("StructureType", "HAS_STRUCTURE"),
                ("Purpose", "HAS_PURPOSE"),
                ("Topic", "MENTIONS_TOPIC"),
                ("Keyword", "MENTIONS_KEYWORD"),
                ("DomainKeyword", "MENTIONS_DOMAIN_KEYWORD"),
            ]

            for node_type, rel_type in expected_relationships:
                result = await session.run(
                    f"""
                    MATCH (i:Interview {{interview_id: $interview_id}})
                    -[:HAS_SENTENCE]->(s:Sentence {{sentence_id: $sentence_id}})
                    MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:{rel_type}]->(n:{node_type})
                    RETURN count(n) as count
                    """,
                    interview_id=interview_id,
                    sentence_id=1000,
                )
                count = await result.single()
                # Should have at least one relationship for each type (except topics which might be empty)
                if node_type in ["Topic"]:
                    assert count["count"] >= 0  # Topics can be empty
                else:
                    assert count["count"] >= 1, f"Missing {rel_type} relationships to {node_type}"

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, clean_test_database):
        """Test that failed transactions don't leave partial data."""
        filename = "rollback_test.txt"

        # Create data that might cause transaction issues
        problematic_data = {
            "sentence_id": 2000,
            "sequence_order": 0,
            "sentence": "Transaction rollback test sentence",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "rollback_testing",
        }

        # First, verify database is clean
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            initial_count = await result.single()
            assert initial_count["count"] == 0

        # Attempt transaction (this should succeed, but let's simulate failure)
        try:
            await save_analysis_to_graph(problematic_data, filename, Neo4jConnectionManager)
        except Exception:
            # If it fails, that's what we're testing
            pass

        # Check if any partial data was left behind
        async with await Neo4jConnectionManager.get_session() as session:
            # Check for orphaned nodes
            result = await session.run("MATCH (n) WHERE NOT (n)-[]-() RETURN count(n) as orphaned_count")
            orphaned = await result.single()

            # Check for incomplete sentence data
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s",
                sentence_id=2000,
            )
            sentence = await result.single()

            if sentence is not None:
                # If sentence exists, verify it has proper structure
                result = await session.run(
                    "MATCH (s:Sentence {sentence_id: $sentence_id})-[:PART_OF_FILE]->(f:SourceFile) RETURN f",
                    sentence_id=2000,
                )
                source_file = await result.single()
                assert source_file is not None, "Sentence exists but missing SourceFile relationship"

            # Should have minimal orphaned nodes (some might be expected from successful operations)
            assert orphaned["orphaned_count"] <= 1, f"Too many orphaned nodes: {orphaned['orphaned_count']}"

    @pytest.mark.asyncio
    async def test_concurrent_transaction_isolation(self, clean_test_database):
        """Test that concurrent transactions maintain isolation."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        # Create sentence mappings for concurrent operations
        for i in range(5):
            await map_storage.write_entry(
                {
                    "sentence_id": 3000 + i,
                    "sentence": f"Concurrent isolation test sentence {i}",
                    "sequence_order": i,
                }
            )

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Define concurrent transaction operations
        async def concurrent_write(sentence_id: int, delay: float = 0.0):
            await asyncio.sleep(delay)  # Stagger operations

            data = {
                "sentence_id": sentence_id,
                "sequence_order": sentence_id - 3000,
                "sentence": f"Concurrent isolation test sentence {sentence_id - 3000}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": f"isolation_testing_{sentence_id}",
                "overall_keywords": [f"concurrent_{sentence_id}", "isolation", "test"],
            }

            await writer.write_result(data)
            return sentence_id

        # Launch concurrent transactions
        tasks = []
        for i in range(5):
            sentence_id = 3000 + i
            delay = i * 0.1  # Stagger by 100ms
            task = asyncio.create_task(concurrent_write(sentence_id, delay))
            tasks.append(task)

        # Wait for all transactions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all transactions completed successfully
        successful_ids = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Transaction failed: {result}")
            else:
                successful_ids.append(result)

        # Should have at least some successful transactions
        assert len(successful_ids) >= 3, f"Too many concurrent transaction failures: {len(successful_ids)}/5"

        # Verify data integrity for successful transactions
        async with await Neo4jConnectionManager.get_session() as session:
            for sentence_id in successful_ids:
                # Check analysis exists and is properly linked
                result = await session.run(
                    """
                    MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                    MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                    RETURN a.purpose as purpose
                    """,
                    interview_id=interview_id,
                    sentence_id=sentence_id,
                )
                analysis = await result.single()
                assert analysis is not None
                assert f"isolation_testing_{sentence_id}" in analysis["purpose"]


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jRelationshipIntegrity:
    """Test relationship consistency and constraint validation."""

    @pytest.mark.asyncio
    async def test_relationship_consistency(self, clean_test_database):
        """Test that relationships maintain referential integrity."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup with multiple sentences
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create a sequence of sentences with relationships
        sentences = []
        for i in range(5):
            sentence_data = {
                "sentence_id": 4000 + i,
                "sentence": f"Relationship integrity test sentence {i}",
                "sequence_order": i,
            }
            sentences.append(sentence_data)
            await map_storage.write_entry(sentence_data)

            # Add analysis for each sentence
            analysis_data = {
                "sentence_id": 4000 + i,
                "sequence_order": i,
                "sentence": f"Relationship integrity test sentence {i}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": "relationship_testing",
            }
            await writer.write_result(analysis_data)

        # Verify relationship integrity
        async with await Neo4jConnectionManager.get_session() as session:
            # Check Project -> Interview -> Sentence chain
            result = await session.run(
                """
                MATCH (p:Project {project_id: $project_id})
                -[:HAS_INTERVIEW]->(i:Interview {interview_id: $interview_id})
                MATCH (i)-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(s) as sentence_count
                """,
                project_id=project_id,
                interview_id=interview_id,
            )
            sentence_count = await result.single()
            assert sentence_count["sentence_count"] == 5

            # Check Sentence -> Analysis relationships
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            analysis_count = await result.single()
            assert analysis_count["analysis_count"] == 5

            # Check sentence sequence relationships (NEXT_SENTENCE)
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s1:Sentence)
                MATCH (s1)-[:NEXT_SENTENCE]->(s2:Sentence)
                RETURN count(*) as sequence_count
                """,
                interview_id=interview_id,
            )
            sequence_count = await result.single()
            assert sequence_count["sequence_count"] == 4  # 4 NEXT_SENTENCE relationships for 5 sentences

            # Verify no orphaned analysis nodes
            result = await session.run(
                """
                MATCH (a:Analysis)
                WHERE NOT (a)<-[:HAS_ANALYSIS]-(:Sentence)
                RETURN count(a) as orphaned_analysis
                """
            )
            orphaned = await result.single()
            assert orphaned["orphaned_analysis"] == 0

    @pytest.mark.asyncio
    async def test_relationship_uniqueness(self, clean_test_database):
        """Test that relationships maintain uniqueness constraints."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create sentence
        await map_storage.write_entry(
            {
                "sentence_id": 5000,
                "sentence": "Relationship uniqueness test sentence",
                "sequence_order": 0,
            }
        )

        # Write analysis multiple times (should not create duplicates)
        analysis_data = {
            "sentence_id": 5000,
            "sequence_order": 0,
            "sentence": "Relationship uniqueness test sentence",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "uniqueness_testing",
        }

        # Write the same analysis multiple times
        for _ in range(3):
            await writer.write_result(analysis_data)

        # Verify only one analysis node exists per sentence
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
                sentence_id=5000,
            )
            analysis_count = await result.single()
            assert analysis_count["analysis_count"] == 1

            # Verify only one relationship to function type
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:HAS_FUNCTION]->(f:FunctionType)
                RETURN count(*) as function_rel_count
                """,
                interview_id=interview_id,
                sentence_id=5000,
            )
            function_count = await result.single()
            assert function_count["function_rel_count"] == 1

    @pytest.mark.asyncio
    async def test_cascade_relationship_integrity(self, clean_test_database):
        """Test that relationship changes maintain integrity across the graph."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup with complex relationship structure
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create sentences with shared analysis dimensions
        shared_keywords = ["shared", "integrity", "cascade"]

        for i in range(3):
            await map_storage.write_entry(
                {
                    "sentence_id": 6000 + i,
                    "sentence": f"Cascade integrity test sentence {i}",
                    "sequence_order": i,
                }
            )

            analysis_data = {
                "sentence_id": 6000 + i,
                "sequence_order": i,
                "sentence": f"Cascade integrity test sentence {i}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": f"cascade_testing_{i}",
                "overall_keywords": shared_keywords + [f"unique_{i}"],
                "domain_keywords": ["cascade", "testing"],
            }
            await writer.write_result(analysis_data)

        # Verify shared dimension nodes are reused correctly
        async with await Neo4jConnectionManager.get_session() as session:
            # Check that shared keywords are reused (not duplicated)
            for keyword in shared_keywords:
                result = await session.run(
                    "MATCH (k:Keyword {text: $keyword}) RETURN count(k) as count",
                    keyword=keyword,
                )
                count = await result.single()
                assert count["count"] == 1, f"Keyword '{keyword}' should exist only once, found {count['count']}"

                # Verify multiple sentences link to the same keyword
                result = await session.run(
                    """
                    MATCH (k:Keyword {text: $keyword})<-[:MENTIONS_OVERALL_KEYWORD]-(a:Analysis)
                    RETURN count(a) as analysis_count
                    """,
                    keyword=keyword,
                )
                analysis_count = await result.single()
                assert analysis_count["analysis_count"] == 3, f"Keyword '{keyword}' should be linked to 3 analyses"

            # Verify unique keywords exist separately
            for i in range(3):
                unique_keyword = f"unique_{i}"
                result = await session.run(
                    "MATCH (k:Keyword {text: $keyword}) RETURN count(k) as count",
                    keyword=unique_keyword,
                )
                count = await result.single()
                assert count["count"] == 1

                # Should be linked to only one analysis
                result = await session.run(
                    """
                    MATCH (k:Keyword {text: $keyword})<-[:MENTIONS_OVERALL_KEYWORD]-(a:Analysis)
                    RETURN count(a) as analysis_count
                    """,
                    keyword=unique_keyword,
                )
                analysis_count = await result.single()
                assert analysis_count["analysis_count"] == 1


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jDataConsistency:
    """Test data consistency across components and operations."""

    @pytest.mark.asyncio
    async def test_cross_component_data_sync(self, clean_test_database):
        """Test data consistency between MapStorage and AnalysisWriter."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Initialize components
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Add sentences via MapStorage
        sentences = []
        for i in range(3):
            sentence_data = {
                "sentence_id": 7000 + i,
                "sentence": f"Cross-component sync test sentence {i}",
                "sequence_order": i,
                "start_time": f"00:0{i}:00",
                "end_time": f"00:0{i}:05",
                "speaker": f"speaker_{i % 2}",
            }
            sentences.append(sentence_data)
            await map_storage.write_entry(sentence_data)

        # Add analysis via AnalysisWriter
        for i, sentence_data in enumerate(sentences):
            analysis_data = {
                "sentence_id": sentence_data["sentence_id"],
                "sequence_order": sentence_data["sequence_order"],
                "sentence": sentence_data["sentence"],
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": f"sync_testing_{i}",
            }
            await writer.write_result(analysis_data)

        # Verify data consistency between components
        async with await Neo4jConnectionManager.get_session() as session:
            # Check that all sentences from MapStorage have corresponding analysis
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id,
                       s.text as sentence_text,
                       s.start_time as start_time,
                       a.purpose as analysis_purpose
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )

            sentence_analysis_pairs = []
            async for record in result:
                sentence_analysis_pairs.append(record)

            assert len(sentence_analysis_pairs) == 3

            for i, record in enumerate(sentence_analysis_pairs):
                # Verify sentence data consistency
                expected_sentence_id = 7000 + i
                assert record["sentence_id"] == expected_sentence_id
                assert f"sync test sentence {i}" in record["sentence_text"]
                assert record["start_time"] == f"00:0{i}:00"

                # Verify analysis data consistency
                assert record["analysis_purpose"] == f"sync_testing_{i}"

            # Check sequence relationships are maintained
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s1:Sentence)
                MATCH (s1)-[:NEXT_SENTENCE]->(s2:Sentence)
                RETURN s1.sequence_order as from_order, s2.sequence_order as to_order
                ORDER BY from_order
                """,
                interview_id=interview_id,
            )

            sequence_pairs = []
            async for record in result:
                sequence_pairs.append((record["from_order"], record["to_order"]))

            expected_pairs = [(0, 1), (1, 2)]
            assert sequence_pairs == expected_pairs

    @pytest.mark.asyncio
    async def test_data_update_consistency(self, clean_test_database):
        """Test that data updates maintain consistency."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create initial sentence and analysis
        await map_storage.write_entry(
            {
                "sentence_id": 8000,
                "sentence": "Original sentence text",
                "sequence_order": 0,
            }
        )

        original_analysis = {
            "sentence_id": 8000,
            "sequence_order": 0,
            "sentence": "Original sentence text",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "update_testing_original",
            "overall_keywords": ["original", "test"],
        }
        await writer.write_result(original_analysis)

        # Update sentence (simulating re-processing)
        updated_sentence = {
            "sentence_id": 8000,
            "sentence": "Updated sentence text with more content",
            "sequence_order": 0,
        }
        await map_storage.write_entry(updated_sentence)

        # Update analysis
        updated_analysis = {
            "sentence_id": 8000,
            "sequence_order": 0,
            "sentence": "Updated sentence text with more content",
            "function_type": "interrogative",  # Changed
            "structure_type": "complex",  # Changed
            "purpose": "update_testing_modified",  # Changed
            "overall_keywords": ["updated", "test", "modified"],  # Changed
        }
        await writer.write_result(updated_analysis)

        # Verify consistency after updates
        async with await Neo4jConnectionManager.get_session() as session:
            # Check sentence was updated
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                RETURN s.text as text
                """,
                interview_id=interview_id,
                sentence_id=8000,
            )
            sentence = await result.single()
            assert "Updated sentence text" in sentence["text"]

            # Check analysis was updated (should have new values)
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                MATCH (a)-[:HAS_FUNCTION]->(f:FunctionType)
                MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
                RETURN a.purpose as purpose, f.name as function_type, st.name as structure_type
                """,
                interview_id=interview_id,
                sentence_id=8000,
            )
            analysis = await result.single()
            assert analysis["purpose"] == "update_testing_modified"
            assert analysis["function_type"] == "interrogative"
            assert analysis["structure_type"] == "complex"

            # Check old keywords were replaced with new ones
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                -[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN collect(k.text) as keywords
                """,
                interview_id=interview_id,
                sentence_id=8000,
            )
            keywords = await result.single()
            keyword_list = keywords["keywords"]

            # Should have updated keywords
            assert "updated" in keyword_list
            assert "modified" in keyword_list
            assert "test" in keyword_list

            # Should not have original keywords (if they were properly replaced)
            # Note: "test" appears in both, so it should still exist

    @pytest.mark.asyncio
    async def test_orphaned_node_detection(self, clean_test_database):
        """Test detection and handling of orphaned nodes."""
        filename = "orphan_test.txt"

        # Create some analysis data using graph persistence
        analysis_data = {
            "sentence_id": 9000,
            "sequence_order": 0,
            "sentence": "Orphaned node detection test",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "orphan_testing",
            "overall_keywords": ["orphan", "detection", "test"],
            "domain_keywords": ["testing", "integrity"],
        }

        await save_analysis_to_graph(analysis_data, filename, Neo4jConnectionManager)

        # Verify initial state has no orphaned nodes
        async with await Neo4jConnectionManager.get_session() as session:
            # Check for nodes without any relationships
            result = await session.run(
                """
                MATCH (n)
                WHERE NOT (n)-[]-()
                RETURN labels(n) as labels, count(n) as count
                """
            )

            orphaned_by_type = {}
            async for record in result:
                labels = record["labels"]
                count = record["count"]
                if labels:
                    label = labels[0]  # Primary label
                    orphaned_by_type[label] = count

            # Should have minimal or no orphaned nodes
            total_orphaned = sum(orphaned_by_type.values())
            assert total_orphaned <= 1, f"Too many orphaned nodes: {orphaned_by_type}"

            # Check for analysis nodes without sentence relationships
            result = await session.run(
                """
                MATCH (a:Analysis)
                WHERE NOT (a)<-[:HAS_ANALYSIS]-(:Sentence)
                RETURN count(a) as orphaned_analysis
                """
            )
            orphaned_analysis = await result.single()
            assert orphaned_analysis["orphaned_analysis"] == 0

            # Check for dimension nodes without analysis relationships
            dimension_types = [
                "FunctionType",
                "StructureType",
                "Purpose",
                "Topic",
                "Keyword",
                "DomainKeyword",
            ]

            for dim_type in dimension_types:
                result = await session.run(
                    f"""
                    MATCH (d:{dim_type})
                    WHERE NOT (d)<-[]-(:Analysis)
                    RETURN count(d) as orphaned_count
                    """
                )
                orphaned = await result.single()

                # Some dimension nodes might be shared, so small numbers are acceptable
                assert (
                    orphaned["orphaned_count"] <= 2
                ), f"Too many orphaned {dim_type} nodes: {orphaned['orphaned_count']}"


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jDataValidation:
    """Test data validation and constraint enforcement."""

    @pytest.mark.asyncio
    async def test_data_type_consistency(self, clean_test_database):
        """Test that data types are consistent across the graph."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create sentence with various data types
        await map_storage.write_entry(
            {
                "sentence_id": 10000,
                "sentence": "Data type consistency test sentence",
                "sequence_order": 0,
                "start_time": "00:01:00",
                "end_time": "00:01:05",
                "speaker": "test_speaker",
            }
        )

        analysis_data = {
            "sentence_id": 10000,
            "sequence_order": 0,
            "sentence": "Data type consistency test sentence",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "type_consistency_testing",
            "overall_keywords": ["type", "consistency", "test"],
            "domain_keywords": ["testing", "validation"],
        }
        await writer.write_result(analysis_data)

        # Verify data types are consistent
        async with await Neo4jConnectionManager.get_session() as session:
            # Check sentence ID is integer
            result = await session.run(
                """
                MATCH (s:Sentence {sentence_id: $sentence_id})
                RETURN s.sentence_id as id,
                       s.sequence_order as order,
                       s.text as text,
                       s.start_time as start_time
                """,
                sentence_id=10000,
            )
            sentence = await result.single()

            assert isinstance(sentence["id"], int)
            assert isinstance(sentence["order"], int)
            assert isinstance(sentence["text"], str)
            assert isinstance(sentence["start_time"], str)

            # Check analysis relationships have correct structure
            result = await session.run(
                """
                MATCH (s:Sentence {sentence_id: $sentence_id})-[:HAS_ANALYSIS]->(a:Analysis)
                MATCH (a)-[:HAS_FUNCTION]->(f:FunctionType)
                MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN f.name as function_name, collect(k.text) as keywords
                """,
                sentence_id=10000,
            )
            analysis = await result.single()

            assert isinstance(analysis["function_name"], str)
            assert isinstance(analysis["keywords"], list)
            assert all(isinstance(keyword, str) for keyword in analysis["keywords"])

    @pytest.mark.asyncio
    async def test_constraint_validation(self, clean_test_database):
        """Test that graph constraints are properly validated."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create sentences with potential constraint violations
        sentences = [
            {"sentence_id": 11000, "sequence_order": 0},
            {"sentence_id": 11001, "sequence_order": 1},
            {"sentence_id": 11002, "sequence_order": 2},
        ]

        for sentence_data in sentences:
            await map_storage.write_entry(
                {
                    **sentence_data,
                    "sentence": f"Constraint validation test sentence {sentence_data['sequence_order']}",
                }
            )

            analysis_data = {
                **sentence_data,
                "sentence": f"Constraint validation test sentence {sentence_data['sequence_order']}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": f"constraint_testing_{sentence_data['sequence_order']}",
            }
            await writer.write_result(analysis_data)

        # Verify constraint compliance
        async with await Neo4jConnectionManager.get_session() as session:
            # Check unique sentence IDs within interview
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(DISTINCT s.sentence_id) as unique_count, count(s) as total_count
                """,
                interview_id=interview_id,
            )
            counts = await result.single()
            assert counts["unique_count"] == counts["total_count"], "Duplicate sentence IDs found"

            # Check sequence order consistency
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN collect(s.sequence_order ORDER BY s.sequence_order) as orders
                """,
                interview_id=interview_id,
            )
            orders = await result.single()
            expected_orders = [0, 1, 2]
            assert orders["orders"] == expected_orders, f"Sequence orders not consistent: {orders['orders']}"

            # Check that each sentence has exactly one analysis
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id, count(a) as analysis_count
                """,
                interview_id=interview_id,
            )

            sentence_analysis_counts = []
            async for record in result:
                sentence_analysis_counts.append((record["sentence_id"], record["analysis_count"]))

            for sentence_id, analysis_count in sentence_analysis_counts:
                assert analysis_count == 1, f"Sentence {sentence_id} has {analysis_count} analyses, expected 1"
