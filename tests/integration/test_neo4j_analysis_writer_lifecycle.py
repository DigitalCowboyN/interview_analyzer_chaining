"""
Enhanced integration tests for Neo4jAnalysisWriter focusing on lifecycle management,
concurrent operations, and edge cases not covered in the main integration tests.

These tests complement the existing integration tests by focusing on:
- Writer lifecycle management (initialize, finalize, cleanup)
- Concurrent write operations and thread safety
- Resource management and connection pooling
- Error recovery scenarios
- Performance characteristics under load

**M2.8 STATUS**: DEPRECATED - Tests use direct Neo4j write pattern without event_emitter.
This functionality is deprecated in M2.8 and will be removed in M3.0.
See M2.8_MIGRATION_SUMMARY.md for migration to event-first dual-write pattern.
"""

import asyncio
import uuid

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.utils.neo4j_driver import Neo4jConnectionManager

# M2.8: Mark all tests in this file as skipped - deprecated direct-write pattern
pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.integration,
    pytest.mark.skip(
        reason="M2.8: Direct Neo4j writes without event_emitter are deprecated. "
               "These lifecycle tests exercise deprecated functionality and will be "
               "removed in M3.0. See M2.8_MIGRATION_SUMMARY.md for migration guidance."
    ),
]


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jAnalysisWriterLifecycle:
    """Test Neo4j Analysis Writer lifecycle management."""

    @pytest.fixture
    def test_project_interview_ids(self):
        """Generate unique project and interview IDs for each test."""
        return str(uuid.uuid4()), str(uuid.uuid4())

    @pytest.fixture
    def sample_analysis_result(self):
        """Provide a sample analysis result for testing."""
        return {
            "sentence_id": 100,
            "sequence_order": 0,
            "sentence": "This is a test sentence for lifecycle testing.",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "testing",
            "topics": ["software_testing", "integration_testing"],  # Combined topic levels
            "keywords": ["test", "lifecycle", "integration"],  # Was overall_keywords
            "domain_keywords": ["testing", "software"],
        }

    @pytest.fixture
    async def setup_project_interview_sentence(
        self, clean_test_database, test_project_interview_ids, sample_analysis_result
    ):
        """
        Ensures Project, Interview, and a Sentence node exist before tests that
        rely on Neo4jAnalysisWriter writing to an existing sentence.
        """
        from src.io.neo4j_map_storage import Neo4jMapStorage

        project_id, interview_id = test_project_interview_ids
        sentence_id = sample_analysis_result["sentence_id"]
        sentence_text = sample_analysis_result["sentence"]
        sequence_order = sample_analysis_result["sequence_order"]

        # Create Project and Interview structure using Neo4jMapStorage
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        # Create Sentence node linked to Interview
        sentence_data = {
            "sentence_id": sentence_id,
            "sentence": sentence_text,
            "sequence_order": sequence_order,
        }
        await map_storage.write_entry(sentence_data)

        yield project_id, interview_id, sentence_id

    async def create_sentence_node(self, project_id: str, interview_id: str, sentence_data: dict):
        """
        Helper method to create a sentence node for a given project/interview.
        """
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()  # Ensure Project/Interview exist
        await map_storage.write_entry(sentence_data)

    @pytest.mark.asyncio
    async def test_writer_initialization_and_cleanup(self, clean_test_database, test_project_interview_ids):
        """Test proper initialization and cleanup of Neo4j Analysis Writer."""
        project_id, interview_id = test_project_interview_ids

        # Test initialization
        writer = Neo4jAnalysisWriter(project_id, interview_id)
        assert writer.project_id == project_id
        assert writer.interview_id == interview_id
        assert writer.get_identifier() == interview_id

        # Initialize writer (should be safe to call)
        await writer.initialize()

        # Test that double initialization is safe
        await writer.initialize()  # Should not raise error

        # Test finalization
        await writer.finalize()
        # Note: Neo4jAnalysisWriter may not have explicit finalization state
        # but should handle finalize() calls gracefully

    @pytest.mark.asyncio
    async def test_write_without_explicit_initialization(
        self, setup_project_interview_sentence, sample_analysis_result
    ):
        """Test behavior when writing without explicit initialization."""
        project_id, interview_id, sentence_id = setup_project_interview_sentence
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Neo4jAnalysisWriter should work without explicit initialization
        # since initialize() is mostly a no-op for this implementation
        await writer.write_result(sample_analysis_result)

        # Verify the data was written
        analysis_ids = await writer.read_analysis_ids()
        assert sample_analysis_result["sentence_id"] in analysis_ids

    @pytest.mark.asyncio
    async def test_multiple_writers_same_interview(self, clean_test_database, test_project_interview_ids):
        """Test behavior with multiple writers for the same interview."""
        project_id, interview_id = test_project_interview_ids

        # Create sentence nodes first (use single MapStorage instance to avoid clearing)
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        sentence_data_1 = {"sentence_id": 200, "sequence_order": 0, "sentence": "First writer sentence."}
        sentence_data_2 = {"sentence_id": 201, "sequence_order": 1, "sentence": "Second writer sentence."}

        await map_storage.write_entry(sentence_data_1)
        await map_storage.write_entry(sentence_data_2)

        # Create two writers for the same interview
        writer1 = Neo4jAnalysisWriter(project_id, interview_id)
        writer2 = Neo4jAnalysisWriter(project_id, interview_id)

        await writer1.initialize()
        await writer2.initialize()

        # Write different results from each writer
        result1 = {
            "sentence_id": 200,
            "sequence_order": 0,
            "sentence": "First writer sentence.",
            "function_type": "declarative",
        }

        result2 = {
            "sentence_id": 201,
            "sequence_order": 1,
            "sentence": "Second writer sentence.",
            "function_type": "declarative",
        }

        await writer1.write_result(result1)
        await writer2.write_result(result2)

        # Both writers should see both results
        ids1 = await writer1.read_analysis_ids()
        ids2 = await writer2.read_analysis_ids()

        assert 200 in ids1
        assert 201 in ids1
        assert 200 in ids2
        assert 201 in ids2

        await writer1.finalize()
        await writer2.finalize()

    @pytest.mark.asyncio
    async def test_writer_resource_cleanup_on_error(self, clean_test_database, test_project_interview_ids):
        """Test that resources are properly cleaned up when errors occur."""
        project_id, interview_id = test_project_interview_ids
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        await writer.initialize()

        # Simulate an error during write operation
        malformed_result = {
            "sentence_id": "invalid_id",  # Should be int
            "sequence_order": "invalid_order",  # Should be int
            "sentence": None,  # Should be string
        }

        # Writer should handle malformed data gracefully
        try:
            await writer.write_result(malformed_result)
        except Exception:
            pass  # Expected to fail

        # Create sentence node for recovery test
        sentence_data = {
            "sentence_id": 300,
            "sentence": "Recovery test sentence.",
            "sequence_order": 0,
        }
        await self.create_sentence_node(project_id, interview_id, sentence_data)

        # Writer should still be functional after error
        valid_result = {
            "sentence_id": 300,
            "sequence_order": 0,
            "sentence": "Recovery test sentence.",
            "function_type": "declarative",
        }

        await writer.write_result(valid_result)

        # Verify the valid result was written
        ids = await writer.read_analysis_ids()
        assert 300 in ids

        await writer.finalize()


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jAnalysisWriterConcurrency:
    """Test concurrent operations with Neo4j Analysis Writer."""

    async def create_sentence_node(self, project_id: str, interview_id: str, sentence_data: dict):
        """
        Helper method to create a sentence node for a given project/interview.
        """
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()  # Ensure Project/Interview exist
        await map_storage.write_entry(sentence_data)

    @pytest.fixture
    def concurrent_test_data(self):
        """Generate test data for concurrent operations."""
        results = []
        for i in range(20):  # 20 concurrent writes
            results.append(
                {
                    "sentence_id": 1000 + i,
                    "sequence_order": i,
                    "sentence": f"Concurrent test sentence {i}.",
                    "function_type": "declarative" if i % 2 == 0 else "interrogative",
                    "structure_type": "simple" if i % 3 == 0 else "complex",
                    "purpose": f"testing_concurrency_{i % 5}",
                    "keywords": [f"concurrent_{i}", f"test_{i % 10}"],  # Fixed: was overall_keywords
                    "domain_keywords": [f"domain_{i % 7}"],
                }
            )
        return results

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_writer(self, clean_test_database, concurrent_test_data):
        """Test concurrent write operations using the same writer instance."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        await writer.initialize()

        # Create sentence nodes for all concurrent test data
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()  # Ensure Project/Interview exist

        for result in concurrent_test_data:
            sentence_data = {
                "sentence_id": result["sentence_id"],
                "sentence": result["sentence"],
                "sequence_order": result["sequence_order"],
            }
            await map_storage.write_entry(sentence_data)

        # Execute concurrent writes
        write_tasks = []
        for result in concurrent_test_data:
            task = asyncio.create_task(writer.write_result(result))
            write_tasks.append(task)

        # Wait for all writes to complete
        await asyncio.gather(*write_tasks)

        # Verify all results were written
        analysis_ids = await writer.read_analysis_ids()
        expected_ids = {result["sentence_id"] for result in concurrent_test_data}
        actual_ids = set(analysis_ids)

        assert actual_ids == expected_ids

        await writer.finalize()

    @pytest.mark.asyncio
    async def test_concurrent_writers_different_interviews(self, clean_test_database, concurrent_test_data):
        """Test concurrent writers for different interviews."""
        project_id = str(uuid.uuid4())

        # Create multiple writers for different interviews
        writers = []
        interview_ids = []
        for i in range(5):  # 5 concurrent writers
            interview_id = str(uuid.uuid4())
            interview_ids.append(interview_id)
            writer = Neo4jAnalysisWriter(project_id, interview_id)
            await writer.initialize()
            writers.append(writer)

        # Create sentence nodes for each interview with appropriate sentence IDs
        for i, interview_id in enumerate(interview_ids):
            map_storage = Neo4jMapStorage(project_id, interview_id)
            await map_storage.initialize()  # Ensure Project/Interview exist

            # Each writer gets 4 results (20 total / 5 writers)
            writer_data = concurrent_test_data[i * 4 : (i + 1) * 4]
            for result in writer_data:
                sentence_data = {
                    "sentence_id": result["sentence_id"],
                    "sentence": result["sentence"],
                    "sequence_order": result["sequence_order"] - (i * 4),  # Adjust sequence order per interview
                }
                await map_storage.write_entry(sentence_data)

        # Each writer writes a subset of data
        write_tasks = []
        for i, writer in enumerate(writers):
            # Each writer gets 4 results (20 total / 5 writers)
            writer_data = concurrent_test_data[i * 4 : (i + 1) * 4]
            for result in writer_data:
                task = asyncio.create_task(writer.write_result(result))
                write_tasks.append(task)

        # Wait for all concurrent writes
        await asyncio.gather(*write_tasks)

        # Verify each writer has the correct data
        for i, writer in enumerate(writers):
            analysis_ids = await writer.read_analysis_ids()
            expected_ids = {result["sentence_id"] for result in concurrent_test_data[i * 4 : (i + 1) * 4]}
            actual_ids = set(analysis_ids)
            assert actual_ids == expected_ids

        # Clean up
        for writer in writers:
            await writer.finalize()

    @pytest.mark.asyncio
    async def test_writer_thread_safety_stress(self, clean_test_database):
        """Stress test writer thread safety with rapid concurrent operations."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        await writer.initialize()

        # Create all sentence nodes first (required for Neo4jAnalysisWriter)
        # Use a single Neo4jMapStorage instance to avoid deleting previously created sentences
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()  # Initialize once

        for i in range(50):
            sentence_data = {
                "sentence_id": 2000 + i,
                "sentence": f"Stress test sentence {2000 + i}.",
                "sequence_order": 2000 + i,
            }
            await map_storage.write_entry(sentence_data)  # Use same instance

        # Rapid concurrent operations mixing reads and writes
        async def write_operation(sentence_id: int):
            result = {
                "sentence_id": sentence_id,
                "sequence_order": sentence_id,
                "sentence": f"Stress test sentence {sentence_id}.",
                "function_type": "declarative",
            }
            await writer.write_result(result)

        async def read_operation():
            return await writer.read_analysis_ids()

        # Mix of write and read operations
        tasks = []
        for i in range(50):  # 50 write operations
            tasks.append(asyncio.create_task(write_operation(2000 + i)))

        for _ in range(10):  # 10 read operations interspersed
            tasks.append(asyncio.create_task(read_operation()))

        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions during stress test: {exceptions}"

        # Verify final state
        final_ids = await writer.read_analysis_ids()
        expected_ids = set(range(2000, 2050))
        actual_ids = set(final_ids)
        assert actual_ids == expected_ids

        await writer.finalize()


@pytest.mark.neo4j
@pytest.mark.integration
@pytest.mark.slow
class TestNeo4jAnalysisWriterPerformance:
    """Test performance characteristics of Neo4j Analysis Writer."""

    @pytest.mark.asyncio
    async def test_bulk_write_performance(self, clean_test_database):
        """Test performance of bulk write operations."""
        import time

        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create required sentence nodes first (required for Neo4jAnalysisWriter)
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)
        await writer.initialize()

        # Generate large dataset
        bulk_data = []
        for i in range(100):  # 100 sentences
            bulk_data.append(
                {
                    "sentence_id": 3000 + i,
                    "sequence_order": i,
                    "sentence": f"Performance test sentence {i} with additional content for realistic size.",
                    "function_type": "declarative" if i % 2 == 0 else "interrogative",
                    "structure_type": "simple" if i % 3 == 0 else "complex",
                    "purpose": f"performance_testing_{i % 8}",
                    "topic_level_1": f"topic_l1_{i % 12}",
                    "topic_level_3": f"topic_l3_{i % 15}",
                    "overall_keywords": [f"perf_{i}", f"test_{i}", f"bulk_{i % 20}"],
                    "domain_keywords": [f"domain_{i % 10}", f"performance_{i % 5}"],
                }
            )

        # Create sentence nodes first (required for Neo4jAnalysisWriter)
        for result in bulk_data:
            await map_storage.write_entry(
                {
                    "sentence_id": result["sentence_id"],
                    "sentence": result["sentence"],
                    "sequence_order": result["sequence_order"],
                }
            )

        # Measure write performance
        start_time = time.time()

        for result in bulk_data:
            await writer.write_result(result)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Bulk write performance: {len(bulk_data)} sentences in {total_time:.2f}s")
        print(f"Average time per sentence: {total_time / len(bulk_data):.3f}s")

        # Verify all data was written
        analysis_ids = await writer.read_analysis_ids()
        expected_ids = {result["sentence_id"] for result in bulk_data}
        actual_ids = set(analysis_ids)
        assert actual_ids == expected_ids

        # Performance assertions (should be reasonable)
        assert total_time < 60.0  # Should complete within 1 minute
        assert total_time / len(bulk_data) < 0.5  # Should average less than 0.5s per sentence

        await writer.finalize()

    @pytest.mark.asyncio
    async def test_read_performance_large_dataset(self, clean_test_database):
        """Test read performance with large datasets."""
        import time

        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create required sentence nodes first (required for Neo4jAnalysisWriter)
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)
        await writer.initialize()

        # Create large dataset
        dataset_size = 200
        for i in range(dataset_size):
            # Create sentence node first
            await map_storage.write_entry(
                {
                    "sentence_id": 4000 + i,
                    "sentence": f"Large dataset sentence {i}.",
                    "sequence_order": i,
                }
            )

            result = {
                "sentence_id": 4000 + i,
                "sequence_order": i,
                "sentence": f"Large dataset sentence {i}.",
                "function_type": "declarative",
            }
            await writer.write_result(result)

        # Measure read performance
        start_time = time.time()
        analysis_ids = await writer.read_analysis_ids()
        end_time = time.time()

        read_time = end_time - start_time
        print(f"Read performance: {len(analysis_ids)} IDs in {read_time:.3f}s")

        # Verify correctness
        assert len(analysis_ids) == dataset_size
        expected_ids = set(range(4000, 4000 + dataset_size))
        actual_ids = set(analysis_ids)
        assert actual_ids == expected_ids

        # Performance assertion
        assert read_time < 5.0  # Should read within 5 seconds

        await writer.finalize()


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jAnalysisWriterErrorRecovery:
    """Test error recovery and resilience scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_loss_recovery(self, clean_test_database):
        """Test writer behavior when database connection is temporarily lost."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create required sentence nodes first (required for Neo4jAnalysisWriter)
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)
        await writer.initialize()

        # Create sentence nodes for the test
        sentence_data = [
            {"sentence_id": 5000, "sentence": "Initial sentence before connection loss.", "sequence_order": 0},
            {"sentence_id": 5001, "sentence": "Sentence during connection loss.", "sequence_order": 1},
            {"sentence_id": 5002, "sentence": "Sentence after connection recovery.", "sequence_order": 2},
        ]
        for data in sentence_data:
            await map_storage.write_entry(data)

        # Write some initial data
        initial_result = {
            "sentence_id": 5000,
            "sequence_order": 0,
            "sentence": "Initial sentence before connection loss.",
            "function_type": "declarative",
        }
        await writer.write_result(initial_result)

        # Simulate connection loss by temporarily closing the driver
        # Note: This is a simplified simulation - real connection loss scenarios
        # would require more sophisticated testing infrastructure
        original_driver = Neo4jConnectionManager._driver

        try:
            # Temporarily set driver to None to simulate connection loss
            Neo4jConnectionManager._driver = None

            # Attempt to write during "connection loss"
            connection_loss_result = {
                "sentence_id": 5001,
                "sequence_order": 1,
                "sentence": "Sentence during connection loss.",
                "function_type": "declarative",
            }

            # This should either succeed (auto-reconnect) or fail gracefully
            try:
                await writer.write_result(connection_loss_result)
                connection_loss_handled = True
            except Exception as e:
                connection_loss_handled = False
                print(f"Expected connection loss error: {e}")

        finally:
            # Restore connection
            Neo4jConnectionManager._driver = original_driver

        # Write data after connection restoration
        recovery_result = {
            "sentence_id": 5002,
            "sequence_order": 2,
            "sentence": "Sentence after connection recovery.",
            "function_type": "declarative",
        }
        await writer.write_result(recovery_result)

        # Verify data integrity
        analysis_ids = await writer.read_analysis_ids()
        assert 5000 in analysis_ids  # Initial data should be preserved
        assert 5002 in analysis_ids  # Recovery data should be written

        # Connection loss data may or may not be present depending on handling
        if connection_loss_handled:
            assert 5001 in analysis_ids

        await writer.finalize()

    @pytest.mark.asyncio
    async def test_partial_write_failure_handling(self, clean_test_database):
        """Test handling of partial write failures."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create required sentence nodes first (required for Neo4jAnalysisWriter)
        from src.io.neo4j_map_storage import Neo4jMapStorage

        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        writer = Neo4jAnalysisWriter(project_id, interview_id)
        await writer.initialize()

        # Create sentence nodes for the test (6000-6004 and 7000)
        sentence_data = []
        for i in range(5):
            sentence_data.append(
                {
                    "sentence_id": 6000 + i,
                    "sentence": f"Valid sentence {i}.",
                    "sequence_order": i,
                }
            )
        sentence_data.append(
            {
                "sentence_id": 6010,
                "sentence": "Post-failure recovery sentence.",
                "sequence_order": 10,
            }
        )
        for data in sentence_data:
            await map_storage.write_entry(data)

        # Write valid data first
        valid_results = []
        for i in range(5):
            result = {
                "sentence_id": 6000 + i,
                "sequence_order": i,
                "sentence": f"Valid sentence {i}.",
                "function_type": "declarative",
            }
            valid_results.append(result)
            await writer.write_result(result)

        # Attempt to write invalid data
        invalid_result = {
            "sentence_id": None,  # Invalid
            "sequence_order": "invalid",  # Invalid type
            "sentence": "",  # Empty string might be invalid
        }

        try:
            await writer.write_result(invalid_result)
            invalid_write_succeeded = True
        except Exception:
            invalid_write_succeeded = False

        # Continue writing valid data after failure
        post_failure_result = {
            "sentence_id": 6010,
            "sequence_order": 10,
            "sentence": "Post-failure recovery sentence.",
            "function_type": "declarative",
        }
        await writer.write_result(post_failure_result)

        # Verify data integrity
        analysis_ids = await writer.read_analysis_ids()

        # All valid data should be present
        for result in valid_results:
            assert result["sentence_id"] in analysis_ids

        assert 6010 in analysis_ids  # Post-failure data should be written

        # Invalid data should not corrupt the database
        if invalid_write_succeeded:
            # If invalid write succeeded, verify it didn't corrupt other data
            assert len(analysis_ids) >= len(valid_results) + 1
        else:
            # If invalid write failed, verify other data is intact
            assert len(analysis_ids) >= len(valid_results) + 1

        await writer.finalize()
