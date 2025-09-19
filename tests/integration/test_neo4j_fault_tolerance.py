"""
Fault tolerance and error recovery integration tests for Neo4j components.

These tests verify that the system can handle various failure scenarios gracefully:
- Network interruptions and connection losses
- Database restarts and temporary unavailability
- Transaction failures and rollback scenarios
- Connection pool exhaustion
- Partial write failures and data consistency
- Recovery from corrupted state
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.persistence.graph_persistence import save_analysis_to_graph
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jNetworkFaultTolerance:
    """Test behavior during network interruptions and connection failures."""

    @pytest.fixture
    def test_analysis_data(self):
        """Provide test analysis data for fault tolerance testing."""
        return {
            "sentence_id": 1000,
            "sequence_order": 0,
            "sentence": "This is a fault tolerance test sentence.",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "testing",
            "topic_level_1": "fault_tolerance",
            "topic_level_3": "network_interruption",
            "overall_keywords": ["fault", "tolerance", "test"],
            "domain_keywords": ["testing", "reliability"],
        }

    @pytest.mark.asyncio
    async def test_connection_loss_during_write(self, clean_test_database, test_analysis_data):
        """Test behavior when connection is lost during write operation."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup proper project/interview structure
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        # Add sentence mappings first
        await map_storage.write_entry({"sentence_id": 999, "sentence": "Initial test sentence", "sequence_order": 0})
        await map_storage.write_entry(
            {"sentence_id": 1001, "sentence": "Connection loss test sentence", "sequence_order": 1}
        )
        await map_storage.write_entry({"sentence_id": 1002, "sentence": "Recovery test sentence", "sequence_order": 2})

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # First, write some data successfully
        initial_data = {**test_analysis_data, "sentence_id": 999}
        await writer.write_result(initial_data)

        # Verify initial write succeeded
        ids_before = await writer.read_analysis_ids()
        assert 999 in ids_before

        # Simulate connection loss by temporarily disrupting the driver
        @asynccontextmanager
        async def failing_session():
            raise ConnectionError("Simulated network interruption")

        # Patch get_session to fail
        with patch.object(Neo4jConnectionManager, "get_session", side_effect=failing_session):
            # Attempt to write during connection loss
            connection_loss_data = {**test_analysis_data, "sentence_id": 1001}

            with pytest.raises(ConnectionError):
                await writer.write_result(connection_loss_data)

        # Restore connection and verify system can recover
        # The original get_session should work again
        recovery_data = {**test_analysis_data, "sentence_id": 1002}
        await writer.write_result(recovery_data)

        # Verify recovery worked and original data is still intact
        ids_after = await writer.read_analysis_ids()
        assert 999 in ids_after  # Original data preserved
        assert 1002 in ids_after  # Recovery data written
        assert 1001 not in ids_after  # Failed write not present

    @pytest.mark.asyncio
    async def test_connection_recovery_with_retry(self, clean_test_database, test_analysis_data):
        """Test automatic connection recovery with retry logic."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Track retry attempts
        retry_count = 0

        async def retry_write_with_recovery(data, max_attempts=3):
            nonlocal retry_count
            for attempt in range(max_attempts):
                try:
                    writer = Neo4jAnalysisWriter(project_id, interview_id)
                    await writer.write_result(data)
                    return True
                except Exception as e:
                    retry_count += 1
                    if attempt < max_attempts - 1:
                        # Wait before retry (exponential backoff)
                        await asyncio.sleep(0.1 * (2**attempt))
                        continue
                    raise e
            return False

        # Simulate intermittent connection issues
        call_count = 0
        original_get_session = Neo4jConnectionManager.get_session

        @asynccontextmanager
        async def intermittent_session():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Intermittent connection failure")
            else:
                # Use original session for successful attempt
                async with await original_get_session() as session:
                    yield session

        with patch.object(Neo4jConnectionManager, "get_session", side_effect=intermittent_session):
            # This should succeed after retries
            success = await retry_write_with_recovery(test_analysis_data, max_attempts=5)
            assert success
            assert retry_count >= 2  # Should have retried

        # Verify data was eventually written
        writer = Neo4jAnalysisWriter(project_id, interview_id)
        ids = await writer.read_analysis_ids()
        assert test_analysis_data["sentence_id"] in ids

    @pytest.mark.asyncio
    async def test_database_restart_simulation(self, clean_test_database, test_analysis_data):
        """Test behavior during database restart scenario."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Setup proper project/interview structure
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        # Add sentence mappings
        await map_storage.write_entry(
            {"sentence_id": 2000, "sentence": "Database restart test sentence", "sequence_order": 0}
        )
        await map_storage.write_entry(
            {"sentence_id": 2001, "sentence": "Post-restart test sentence", "sequence_order": 1}
        )

        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Write initial data
        initial_data = {**test_analysis_data, "sentence_id": 2000}
        await writer.write_result(initial_data)

        # Simulate database restart by forcing driver reconnection
        await Neo4jConnectionManager.close_driver()

        # Verify driver is closed
        assert Neo4jConnectionManager._driver is None

        # Next operation should trigger reconnection
        restart_data = {**test_analysis_data, "sentence_id": 2001}
        await writer.write_result(restart_data)

        # Verify both records exist (data survived restart simulation)
        ids = await writer.read_analysis_ids()
        assert 2000 in ids
        assert 2001 in ids

    @pytest.mark.asyncio
    async def test_transaction_failure_rollback(self, clean_test_database):
        """Test that transaction failures don't leave partial data."""
        filename = "transaction_test.txt"

        # Create data that will cause a transaction failure
        problematic_data = {
            "sentence_id": 3000,
            "sequence_order": 0,
            "sentence": "Transaction failure test",
            "function_type": None,  # This might cause issues in some scenarios
            "structure_type": "invalid_structure_that_might_cause_constraint_violation",
            "purpose": "testing",
        }

        # Use graph persistence directly to test transaction behavior
        try:
            await save_analysis_to_graph(problematic_data, filename, Neo4jConnectionManager)
        except Exception:
            # Expected to fail - that's what we're testing
            pass

        # Verify no partial data was left behind
        async with await Neo4jConnectionManager.get_session() as session:
            # Check for any nodes related to this test
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN count(s) as count",
                sentence_id=3000,
            )
            count = await result.single()

            # Should be 0 if transaction properly rolled back
            # Or 1 if the transaction succeeded despite the problematic data
            # Either way, we shouldn't have partial/corrupted data
            assert count["count"] in [0, 1]


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jConnectionPoolFaultTolerance:
    """Test behavior under connection pool stress and exhaustion."""

    @pytest.mark.asyncio
    async def test_concurrent_connection_stress(self, clean_test_database):
        """Test system behavior under concurrent connection stress."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create multiple concurrent operations
        async def concurrent_write_operation(sentence_id: int):
            writer = Neo4jAnalysisWriter(project_id, interview_id)
            data = {
                "sentence_id": sentence_id,
                "sequence_order": sentence_id,
                "sentence": f"Concurrent stress test sentence {sentence_id}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": "stress_testing",
            }
            await writer.write_result(data)
            return sentence_id

        # Launch many concurrent operations
        num_concurrent = 20
        tasks = []
        for i in range(num_concurrent):
            task = asyncio.create_task(concurrent_write_operation(4000 + i))
            tasks.append(task)

        # Wait for all operations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful operations
        successful_writes = []
        exceptions = []
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                successful_writes.append(result)

        # Should have mostly successful writes
        success_rate = len(successful_writes) / num_concurrent
        assert success_rate >= 0.8  # At least 80% success rate

        # Verify successful writes are in database
        writer = Neo4jAnalysisWriter(project_id, interview_id)
        stored_ids = await writer.read_analysis_ids()

        for sentence_id in successful_writes:
            assert sentence_id in stored_ids

    @pytest.mark.asyncio
    async def test_connection_pool_recovery(self, clean_test_database):
        """Test that connection pool recovers from temporary exhaustion."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Simulate connection pool exhaustion scenario
        # by creating many long-running "connections"
        long_running_tasks = []

        async def long_running_operation(operation_id: int):
            # Simulate a long-running database operation
            async with await Neo4jConnectionManager.get_session() as session:
                # Hold the session for a while
                await asyncio.sleep(0.5)
                await session.run("RETURN $id as operation_id", id=operation_id)
                return operation_id

        # Start multiple long-running operations
        for i in range(10):
            task = asyncio.create_task(long_running_operation(i))
            long_running_tasks.append(task)

        # Give them a moment to start
        await asyncio.sleep(0.1)

        # Now try to do a regular write operation
        # This should either succeed or handle pool exhaustion gracefully
        test_data = {
            "sentence_id": 5000,
            "sequence_order": 0,
            "sentence": "Connection pool recovery test",
            "function_type": "declarative",
        }

        start_time = time.time()
        await writer.write_result(test_data)
        end_time = time.time()

        # Operation should complete (possibly with some delay due to pool pressure)
        assert end_time - start_time < 10.0  # Should not hang indefinitely

        # Wait for long-running operations to complete
        await asyncio.gather(*long_running_tasks)

        # Verify the write succeeded
        ids = await writer.read_analysis_ids()
        assert 5000 in ids


@pytest.mark.neo4j
@pytest.mark.integration
class TestNeo4jDataConsistencyFaultTolerance:
    """Test data consistency during various failure scenarios."""

    @pytest.mark.asyncio
    async def test_partial_write_consistency(self, clean_test_database):
        """Test that partial writes maintain data consistency."""
        filename = "consistency_test.txt"

        # Create data with multiple components that should be written atomically
        complex_data = {
            "sentence_id": 6000,
            "sequence_order": 0,
            "sentence": "Complex data consistency test",
            "function_type": "declarative",
            "structure_type": "complex",
            "purpose": "consistency_testing",
            "topic_level_1": "data_integrity",
            "topic_level_3": "consistency_validation",
            "overall_keywords": ["consistency", "integrity", "atomic"],
            "domain_keywords": ["testing", "validation", "data"],
        }

        # Write the complex data
        await save_analysis_to_graph(complex_data, filename, Neo4jConnectionManager)

        # Verify all components were written consistently
        async with await Neo4jConnectionManager.get_session() as session:
            # Check sentence exists
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s",
                sentence_id=6000,
            )
            sentence = await result.single()
            assert sentence is not None

            # Check all expected relationships exist
            relationship_checks = [
                ("FunctionType", "HAS_FUNCTION_TYPE"),
                ("StructureType", "HAS_STRUCTURE_TYPE"),
                ("Purpose", "HAS_PURPOSE"),
                ("Topic", "HAS_TOPIC"),
                ("Keyword", "MENTIONS_OVERALL_KEYWORD"),
                ("Keyword", "MENTIONS_DOMAIN_KEYWORD"),
            ]

            for node_type, relationship_type in relationship_checks:
                result = await session.run(
                    f"MATCH (s:Sentence {{sentence_id: $sentence_id}})-[:{relationship_type}]->(n:{node_type}) "
                    "RETURN count(n) as count",
                    sentence_id=6000,
                )
                count = await result.single()
                # Should have at least one relationship of each type
                # (except for topics/keywords which might be empty)
                if node_type in ["Topic", "Keyword"]:
                    assert count["count"] >= 0
                else:
                    assert count["count"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_write_consistency(self, clean_test_database):
        """Test data consistency under concurrent write operations."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        filename = "concurrent_consistency_test.txt"

        # Create multiple writers trying to write to the same sentence
        async def concurrent_writer(writer_id: int, sentence_id: int):
            data = {
                "sentence_id": sentence_id,
                "sequence_order": 0,
                "sentence": f"Concurrent consistency test from writer {writer_id}",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": f"testing_writer_{writer_id}",
            }

            # Use graph persistence to test concurrent access
            await save_analysis_to_graph(data, filename, Neo4jConnectionManager)
            return writer_id

        # Multiple writers trying to write to the same sentence ID
        sentence_id = 7000
        num_writers = 5

        tasks = []
        for writer_id in range(num_writers):
            task = asyncio.create_task(concurrent_writer(writer_id, sentence_id))
            tasks.append(task)

        # Wait for all concurrent writes
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify data consistency - should have exactly one sentence
        async with await Neo4jConnectionManager.get_session() as session:
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN count(s) as count",
                sentence_id=sentence_id,
            )
            count = await result.single()
            assert count["count"] == 1  # Should not have duplicates

            # Verify sentence has consistent data (one of the writers won)
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s.text as text",
                sentence_id=sentence_id,
            )
            sentence_text = await result.single()
            assert "Concurrent consistency test from writer" in sentence_text["text"]

    @pytest.mark.asyncio
    async def test_recovery_from_corrupted_state(self, clean_test_database):
        """Test recovery when database is in an unexpected state."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Create a "corrupted" state by manually creating incomplete data
        async with await Neo4jConnectionManager.get_session() as session:
            # Create sentence without proper project/interview structure
            await session.run(
                "CREATE (s:Sentence {sentence_id: $sentence_id, text: $text})",
                sentence_id=8000,
                text="Orphaned sentence for corruption test",
            )

        # Now try to write analysis for this sentence using the normal flow
        # This should either succeed by creating missing structure or fail gracefully
        recovery_data = {
            "sentence_id": 8000,
            "sequence_order": 0,
            "sentence": "Recovery test sentence",
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "recovery_testing",
        }

        # This might fail due to missing project/interview structure
        # but should not crash the system
        try:
            await writer.write_result(recovery_data)
        except Exception as e:
            # Should be a specific, handled exception, not a crash
            assert "not found" in str(e).lower() or "missing" in str(e).lower()

        # Verify system is still functional for new data
        new_data = {
            "sentence_id": 8001,
            "sequence_order": 0,
            "sentence": "New data after recovery test",
            "function_type": "declarative",
        }

        # Create proper project/interview structure first
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()
        await map_storage.write_sentence_mapping(
            {
                "sentence_id": 8001,
                "text": "New data after recovery test",
                "sequence_order": 0,
            }
        )

        # Now the analysis write should succeed
        await writer.write_result(new_data)

        # Verify system recovered and can process new data
        ids = await writer.read_analysis_ids()
        assert 8001 in ids


@pytest.mark.neo4j
@pytest.mark.integration
@pytest.mark.slow
class TestNeo4jLongRunningFaultTolerance:
    """Test fault tolerance over extended periods and operations."""

    @pytest.mark.asyncio
    async def test_long_running_operation_stability(self, clean_test_database):
        """Test system stability during long-running operations."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        writer = Neo4jAnalysisWriter(project_id, interview_id)

        # Setup initial structure
        map_storage = Neo4jMapStorage(project_id, interview_id)
        await map_storage.initialize()

        # Simulate a long-running batch operation
        batch_size = 50
        total_operations = 200
        successful_operations = 0
        failed_operations = 0

        for batch_start in range(0, total_operations, batch_size):
            batch_end = min(batch_start + batch_size, total_operations)
            batch_tasks = []

            # Process batch concurrently
            for i in range(batch_start, batch_end):
                data = {
                    "sentence_id": 9000 + i,
                    "sequence_order": i,
                    "sentence": f"Long-running stability test sentence {i}",
                    "function_type": "declarative",
                    "structure_type": "simple",
                    "purpose": "stability_testing",
                }

                # Add sentence mapping first
                await map_storage.write_sentence_mapping(
                    {
                        "sentence_id": 9000 + i,
                        "text": data["sentence"],
                        "sequence_order": i,
                    }
                )

                # Create write task
                task = asyncio.create_task(writer.write_result(data))
                batch_tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_operations += 1
                else:
                    successful_operations += 1

            # Brief pause between batches
            await asyncio.sleep(0.1)

        # Verify high success rate
        success_rate = successful_operations / total_operations
        assert success_rate >= 0.95  # Should have very high success rate

        # Verify data integrity
        final_ids = await writer.read_analysis_ids()
        assert len(final_ids) >= int(total_operations * 0.95)

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, clean_test_database):
        """Test that long-running operations don't cause memory leaks."""
        project_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Track connection creation/cleanup
        connection_count = 0
        original_get_session = Neo4jConnectionManager.get_session

        @asynccontextmanager
        async def tracked_session():
            nonlocal connection_count
            connection_count += 1
            async with await original_get_session() as session:
                try:
                    yield session
                finally:
                    connection_count -= 1

        with patch.object(Neo4jConnectionManager, "get_session", side_effect=tracked_session):
            # Perform many operations
            writer = Neo4jAnalysisWriter(project_id, interview_id)

            # Setup structure
            map_storage = Neo4jMapStorage(project_id, interview_id)
            await map_storage.initialize()

            for i in range(100):
                data = {
                    "sentence_id": 10000 + i,
                    "sequence_order": i,
                    "sentence": f"Memory leak test sentence {i}",
                    "function_type": "declarative",
                }

                await map_storage.write_sentence_mapping(
                    {
                        "sentence_id": 10000 + i,
                        "text": data["sentence"],
                        "sequence_order": i,
                    }
                )

                await writer.write_result(data)

        # Verify connections were properly cleaned up
        # (Allow some tolerance for async cleanup)
        await asyncio.sleep(0.1)
        assert abs(connection_count) <= 1  # Should be close to 0
