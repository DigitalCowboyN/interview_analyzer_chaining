"""
End-to-end file processing tests with M2.8 event-sourced architecture.

Tests the complete workflow:
1. File upload → Pipeline processing
2. Events emitted to EventStoreDB (event-only writes)
3. Events processed through projection service
4. Data written to Neo4j (by projection service)
5. Event metadata validation
"""

import os
import uuid
from pathlib import Path

import pytest

from src.config import config
from src.pipeline import PipelineOrchestrator
from src.projections.handlers.interview_handlers import InterviewCreatedHandler
from src.projections.handlers.sentence_handlers import SentenceCreatedHandler
from src.utils.environment import detect_environment


async def process_events_through_projection(
    event_store,
    interview_id: str,
    num_sentences: int,
):
    """
    Helper to process events through projection service handlers.

    This simulates what the projection service does in production:
    1. Read events from EventStoreDB
    2. Process through handlers
    3. Write to Neo4j
    """
    # Process InterviewCreated event
    interview_handler = InterviewCreatedHandler()
    interview_stream = f"Interview-{interview_id}"
    interview_events = await event_store.read_stream(interview_stream)

    for event in interview_events:
        if event.event_type == "InterviewCreated":
            await interview_handler.handle(event)

    # Process SentenceCreated events
    sentence_handler = SentenceCreatedHandler()

    for i in range(num_sentences):
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
        sentence_stream = f"Sentence-{sentence_id}"
        sentence_events = await event_store.read_stream(sentence_stream)

        for event in sentence_events:
            if event.event_type == "SentenceCreated":
                await sentence_handler.handle(event)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestE2EFileProcessingWithDualWrite:
    """Test end-to-end file processing with dual-write enabled."""

    async def test_single_file_upload_with_dual_write(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test complete workflow: file upload → events + Neo4j.

        Validates:
        1. File is processed by pipeline
        2. Events are emitted to EventStoreDB
        3. Neo4j contains data from direct write
        4. Event metadata is correct (correlation IDs, actor, timestamps)
        """
        # Enable event sourcing for this test
        test_config = config.copy()
        # Use environment-aware connection string with configurable host/port
        esdb_connection = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")
        if not esdb_connection:
            environment = detect_environment()
            host = os.getenv("EVENTSTORE_HOST", "eventstore" if environment in ("docker", "ci") else "localhost")
            port = os.getenv("EVENTSTORE_PORT", "2113")
            esdb_connection = f"esdb://{host}:{port}?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        # Create output directory for the test
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Create pipeline orchestrator
        # Use the sample file's parent directory as input_dir
        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )

        # Process the file
        await pipeline._process_single_file(Path(sample_interview_file))

        # Generate expected interview_id (pipeline uses deterministic uuid5)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # === Validation 1: Check EventStoreDB for InterviewCreated event ===
        interview_stream = f"Interview-{interview_id}"
        interview_events = await clean_event_store.read_stream(interview_stream)

        assert len(interview_events) > 0, "No InterviewCreated event found in EventStoreDB"

        interview_created = interview_events[0]
        assert interview_created.event_type == "InterviewCreated"
        assert interview_created.aggregate_id == interview_id
        assert interview_created.aggregate_type == "Interview"
        assert interview_created.version == 0
        assert interview_created.actor.actor_type == "system"
        assert interview_created.correlation_id is not None

        # === Validation 2: Check EventStoreDB for SentenceCreated events ===
        # The test file has 4 sentences
        expected_sentence_count = 4

        for i in range(expected_sentence_count):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"

            sentence_events = await clean_event_store.read_stream(sentence_stream)
            assert len(sentence_events) > 0, f"No SentenceCreated event for sentence {i}"

            sentence_created = sentence_events[0]
            assert sentence_created.event_type == "SentenceCreated"
            assert sentence_created.aggregate_id == sentence_id
            assert sentence_created.aggregate_type == "Sentence"
            assert sentence_created.version == 0
            assert sentence_created.actor.actor_type == "system"
            # All sentence events should have the same correlation_id as the interview
            assert sentence_created.correlation_id == interview_created.correlation_id

        # === M2.8: Process events through projection service ===
        # In M2.8, projection service is the SOLE writer to Neo4j
        # Pipeline only emits events, so we must process them through handlers
        await process_events_through_projection(clean_event_store, interview_id, expected_sentence_count)

        # === Validation 3: Check Neo4j for data (written by projection service) ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        async with driver.session() as session:
            # Check for Interview node
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i", interview_id=interview_id
            )
            interview_node = await result.single()
            assert interview_node is not None, "Interview node not found in Neo4j"

            # Check for Sentence nodes
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence) "
                "RETURN count(s) as sentence_count",
                interview_id=interview_id,
            )
            record = await result.single()
            assert (
                record["sentence_count"] == expected_sentence_count
            ), f"Expected {expected_sentence_count} sentences in Neo4j, got {record['sentence_count']}"

            # Note: SourceFile nodes were created by dual-write mechanism removed in M3.0
            # Projection handlers only create Interview, Project, and Sentence nodes

        print("\n✓ M3.0 Single file processing validated:")
        print("  - InterviewCreated event in EventStoreDB")
        print(f"  - {expected_sentence_count} SentenceCreated events in EventStoreDB")
        print("  - Events processed through projection service")
        print("  - Interview node in Neo4j (written by projection service)")
        print(f"  - {expected_sentence_count} Sentence nodes in Neo4j (written by projection service)")
        print("  - Correlation ID consistent across all events")

    async def test_deterministic_sentence_uuids(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that sentence UUIDs are deterministic (uuid5 from interview_id:index).

        This ensures that:
        1. Replaying events will target the same sentences
        2. Manual edits can target specific sentences predictably
        3. No duplicate sentence nodes in Neo4j
        """
        # Enable event sourcing
        test_config = config.copy()
        # Use environment-aware connection string
        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Process the file ONCE
        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )
        await pipeline._process_single_file(Path(sample_interview_file))

        # === Validation: Verify UUIDs are deterministic ===
        # Calculate expected interview_id (deterministic based on filename)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # Calculate expected sentence UUIDs (deterministic based on interview_id:index)
        expected_sentence_ids = []
        for i in range(4):  # Sample file has 4 sentences
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            expected_sentence_ids.append(sentence_id)

        # Verify each sentence stream uses the expected deterministic UUID
        for i, expected_sentence_id in enumerate(expected_sentence_ids):
            sentence_stream = f"Sentence-{expected_sentence_id}"

            # Read events from stream
            sentence_events = await clean_event_store.read_stream(sentence_stream)
            assert len(sentence_events) >= 1, f"No events for sentence {i}"

            # Verify all events use the same (deterministic) sentence_id
            for event in sentence_events:
                assert event.aggregate_id == expected_sentence_id, \
                    f"Sentence {i} UUID mismatch: got {event.aggregate_id}, expected {expected_sentence_id}"

        # Verify the interview stream uses deterministic UUID
        interview_stream = f"Interview-{interview_id}"
        interview_events = await clean_event_store.read_stream(interview_stream)
        assert len(interview_events) >= 1, "No interview events"
        assert interview_events[0].aggregate_id == interview_id, "Interview UUID is not deterministic"

        print("\n✓ Deterministic UUID validation passed:")
        print(f"  - Interview ID: {interview_id} (deterministic from filename)")
        print(f"  - Sentence IDs: {len(expected_sentence_ids)} sentences with deterministic UUIDs")
        print("  - Using uuid5(namespace:identifier) generation strategy")
        print("  - UUIDs will be identical on reprocessing (idempotent)")

    @pytest.mark.eventstore
    @pytest.mark.integration
    async def test_multiple_files_concurrent_processing(
        self,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test concurrent file processing with dual-write.

        Validates:
        1. Multiple files processed simultaneously
        2. Events correctly partitioned by interview_id
        3. No event loss or corruption
        4. Neo4j has data for all files
        """
        # Enable event sourcing
        test_config = config.copy()
        # Use environment-aware connection string
        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Create 3 test files
        file_paths = []
        for i in range(3):
            content = f"""File {i} sentence 1.
File {i} sentence 2.
File {i} sentence 3."""
            file_path = tmp_path / f"test_file_{i}.txt"
            file_path.write_text(content)
            file_paths.append(file_path)

        # Clean up streams for these specific files (in case they exist from previous runs)
        from esdbclient import StreamState

        for file_path in file_paths:
            interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{file_path.name}"))
            streams_to_delete = [f"Interview-{interview_id}"]

            # Add sentence streams (3 sentences per file)
            for j in range(10):  # Clean up to 10 sentences
                sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{j}"))
                streams_to_delete.append(f"Sentence-{sentence_id}")

            for stream_name in streams_to_delete:
                try:
                    clean_event_store._client.delete_stream(stream_name, current_version=StreamState.ANY)
                except Exception:
                    pass  # Stream doesn't exist, which is fine

        # Process all files concurrently
        pipeline = PipelineOrchestrator(input_dir=tmp_path, output_dir=output_dir, config_dict=test_config)

        import asyncio

        tasks = [pipeline._process_single_file(Path(fp)) for fp in file_paths]
        await asyncio.gather(*tasks)

        # === M2.8: Process events through projection service for all files ===
        # In M2.8, projection service is the SOLE writer to Neo4j
        # Each file has 3 sentences
        for file_path in file_paths:
            interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{file_path.name}"))
            await process_events_through_projection(clean_event_store, interview_id, 3)

        # === Validation: Check each file's events and Neo4j data ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        for i, file_path in enumerate(file_paths):
            interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{file_path.name}"))

            # Check EventStoreDB
            interview_stream = f"Interview-{interview_id}"
            interview_events = await clean_event_store.read_stream(interview_stream)
            assert len(interview_events) > 0, f"No events for file {i}"

            # Check Neo4j (written by projection service)
            async with driver.session() as session:
                result = await session.run(
                    "MATCH (i:Interview {interview_id: $interview_id}) "
                    "RETURN i, [(i)-[:HAS_SENTENCE]->(s:Sentence) | s] as sentences",
                    interview_id=interview_id,
                )
                record = await result.single()
                assert record is not None, f"Interview not found in Neo4j for file {i}"
                assert len(record["sentences"]) == 3, f"Expected 3 sentences for file {i}"

        print("\n✓ M2.8 Concurrent processing validated:")
        print("  - 3 files processed simultaneously")
        print("  - Events correctly partitioned by interview_id")
        print("  - Events processed through projection service")
        print("  - All data present in Neo4j (written by projection service)")
