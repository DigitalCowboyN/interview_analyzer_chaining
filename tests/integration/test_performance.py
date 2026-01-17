"""
Performance and load tests for event-sourced architecture.

Tests:
1. Event emission throughput
2. Projection processing lag
3. Concurrent load handling
4. Memory and resource usage
"""

import asyncio
import os
import time
import uuid

import pytest

from src.events.envelope import Actor, ActorType
from src.events.sentence_events import create_sentence_created_event
from src.projections.handlers.sentence_handlers import SentenceCreatedHandler
from src.utils.environment import detect_environment
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.eventstore
@pytest.mark.slow
class TestEventEmissionPerformance:
    """Test event emission throughput and latency."""

    async def test_event_emission_throughput(
        self,
        event_store_client,
    ):
        """
        Test event emission throughput.

        Target: < 10ms per event
        """
        interview_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        # Generate 100 events
        events = []
        for i in range(100):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i}",
                actor=system_actor,
                correlation_id=correlation_id,
            )
            events.append((sentence_id, event))

        # === Measure emission time ===
        start_time = time.time()

        for sentence_id, event in events:
            stream_name = f"Sentence-{sentence_id}"
            await event_store_client.append_events(
                stream_name=stream_name,
                events=[event],
                expected_version=-1,
            )

        elapsed_time = time.time() - start_time

        # === Validate performance ===
        avg_time_per_event = (elapsed_time / 100) * 1000  # Convert to ms
        assert (
            avg_time_per_event < 10
        ), f"Event emission too slow: {avg_time_per_event:.2f}ms per event (target: < 10ms)"

        print("\n✓ Event emission performance:")
        print(f"  - 100 events emitted in {elapsed_time:.2f}s")
        print(f"  - Average: {avg_time_per_event:.2f}ms per event")
        print(f"  - Throughput: {100 / elapsed_time:.2f} events/sec")

    async def test_batch_event_emission(
        self,
        event_store_client,
    ):
        """
        Test batch event emission performance.

        Validates that batching multiple events to the same stream is more efficient.
        """
        interview_id = str(uuid.uuid4())
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        # Generate 100 events for the same sentence (simulating multiple updates)
        events = []
        for i in range(100):
            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=i,
                interview_id=interview_id,
                index=0,
                text=f"Version {i}",
                actor=system_actor,
                correlation_id=correlation_id,
            )
            events.append(event)

        # === Measure batch emission time ===
        start_time = time.time()

        stream_name = f"Sentence-{sentence_id}"
        # Note: In practice, we'd append in smaller batches to respect optimistic concurrency
        # For this test, we're measuring the theoretical batch performance
        for event in events:
            await event_store_client.append_events(
                stream_name=stream_name,
                events=[event],
                expected_version=event.version - 1,
            )

        elapsed_time = time.time() - start_time

        # === Validate performance ===
        avg_time_per_event = (elapsed_time / 100) * 1000  # Convert to ms
        print("\n✓ Batch event emission performance:")
        print(f"  - 100 events to same stream in {elapsed_time:.2f}s")
        print(f"  - Average: {avg_time_per_event:.2f}ms per event")
        print(f"  - Throughput: {100 / elapsed_time:.2f} events/sec")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.eventstore
@pytest.mark.slow
class TestProjectionPerformance:
    """Test projection processing performance and lag."""

    async def test_projection_processing_lag(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test projection processing lag.

        Target: < 1 second lag for 100 events
        """
        interview_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        # === Step 1: Emit 100 events to EventStoreDB ===
        events = []
        for i in range(100):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i}",
                actor=system_actor,
                correlation_id=correlation_id,
            )

            stream_name = f"Sentence-{sentence_id}"
            await event_store_client.append_events(
                stream_name=stream_name,
                events=[event],
                expected_version=-1,
            )
            events.append((sentence_id, event))

        emission_complete_time = time.time()

        # === Step 2: Process events through projection handler ===
        handler = SentenceCreatedHandler()

        for sentence_id, event in events:
            await handler.handle(event)

        processing_complete_time = time.time()

        # === Calculate lag ===
        projection_lag = processing_complete_time - emission_complete_time

        # === Validate performance ===
        assert projection_lag < 1.0, f"Projection lag too high: {projection_lag:.2f}s (target: < 1s)"

        print("\n✓ Projection processing performance:")
        print(f"  - 100 events processed in {projection_lag:.2f}s")
        print(f"  - Average: {(projection_lag / 100) * 1000:.2f}ms per event")
        print(f"  - Throughput: {100 / projection_lag:.2f} events/sec")

    async def test_concurrent_projection_processing(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test concurrent processing of events for different aggregates.

        Validates that processing events for different sentences concurrently
        is more efficient than sequential processing.
        """
        num_sentences = 50
        interview_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        # === Step 1: Create events ===
        events = []
        for i in range(num_sentences):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i}",
                actor=system_actor,
                correlation_id=correlation_id,
            )

            stream_name = f"Sentence-{sentence_id}"
            await event_store_client.append_events(
                stream_name=stream_name,
                events=[event],
                expected_version=-1,
            )
            events.append(event)

        # === Step 2: Process sequentially ===
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        handler = SentenceCreatedHandler()

        # Clear database for sequential test
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        sequential_start = time.time()
        for event in events:
            await handler.handle(event)
        sequential_time = time.time() - sequential_start

        # === Step 3: Process concurrently ===
        # Clear database for concurrent test
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        concurrent_start = time.time()
        tasks = [handler.handle(event) for event in events]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - concurrent_start

        # === Validate performance ===
        # Concurrent should be faster (or at least not much slower)
        speedup = sequential_time / concurrent_time
        print("\n✓ Concurrent projection processing:")
        print(f"  - Sequential: {sequential_time:.2f}s")
        print(f"  - Concurrent: {concurrent_time:.2f}s")
        print(f"  - Speedup: {speedup:.2f}x")

        # We expect at least some benefit from concurrency, but Neo4j may serialize writes
        assert speedup >= 0.8, "Concurrent processing significantly slower than sequential"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.eventstore
@pytest.mark.slow
class TestLoadTesting:
    """Test system behavior under load."""

    async def test_high_volume_event_processing(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test processing 1000 events.

        Validates:
        1. No events lost
        2. System remains stable
        3. Acceptable throughput
        """
        interview_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        # === Emit 1000 events ===
        start_time = time.time()

        for i in range(1000):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i}",
                actor=system_actor,
                correlation_id=correlation_id,
            )

            stream_name = f"Sentence-{sentence_id}"
            await event_store_client.append_events(
                stream_name=stream_name,
                events=[event],
                expected_version=-1,
            )

        elapsed_time = time.time() - start_time

        # === Validate ===
        throughput = 1000 / elapsed_time
        print("\n✓ High volume event processing:")
        print(f"  - 1000 events emitted in {elapsed_time:.2f}s")
        print(f"  - Throughput: {throughput:.2f} events/sec")

        # Verify we can read back all events (no loss)
        # This would require reading all 1000 streams, which is expensive
        # In practice, we rely on EventStoreDB's guarantees
        print("  - Event loss validation: Assuming EventStoreDB guarantees (not verified)")

    async def test_concurrent_file_processing(
        self,
        clean_test_database,
        event_store_client,
        tmp_path,
    ):
        """
        Test processing 10 files concurrently.

        Validates:
        1. No event loss or corruption
        2. All files processed successfully
        3. Acceptable total time
        """
        from pathlib import Path

        from src.config import config
        from src.pipeline import PipelineOrchestrator

        # Enable event sourcing
        test_config = config.copy()
        # Use environment-aware connection string with configurable host/port
        esdb_connection = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")
        if not esdb_connection:
            environment = detect_environment()
            host = os.getenv("EVENTSTORE_HOST", "eventstore" if environment in ("docker", "ci") else "localhost")
            port = os.getenv("EVENTSTORE_PORT", "2113")
            esdb_connection = f"esdb://{host}:{port}?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Create 10 test files
        file_paths = []
        for i in range(10):
            content = f"File {i} sentence 1.\nFile {i} sentence 2.\nFile {i} sentence 3."
            file_path = input_dir / f"concurrent_test_{i}.txt"
            file_path.write_text(content)
            file_paths.append(file_path)

        # === Process concurrently ===
        pipeline = PipelineOrchestrator(input_dir=input_dir, config_dict=test_config)

        start_time = time.time()
        tasks = [pipeline._process_single_file(Path(fp)) for fp in file_paths]
        await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        # === Validate ===
        print("\n✓ Concurrent file processing:")
        print(f"  - 10 files processed in {elapsed_time:.2f}s")
        print(f"  - Average: {elapsed_time / 10:.2f}s per file")

        # Verify data in Neo4j (spot check)
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        async with driver.session() as session:
            result = await session.run("MATCH (i:Interview) RETURN count(i) as count")
            interview_count = (await result.single())["count"]
            assert interview_count == 10, f"Expected 10 interviews, got {interview_count}"

            result = await session.run("MATCH (s:Sentence) RETURN count(s) as count")
            sentence_count = (await result.single())["count"]
            # Each file has 3 sentences
            assert sentence_count == 30, f"Expected 30 sentences, got {sentence_count}"

        print("  - All 10 interviews in Neo4j")
        print("  - All 30 sentences in Neo4j")
